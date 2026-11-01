#!/bin/bash
# Seslendirme hizmetlerini macOS servisi olarak çalıştırır: oturum açılınca kendiliğinden başlar, çökerse yeniden
# başlar. İkisi de yalnızca bu bilgisayardan erişilir:
#   uygulama (Streamlit)  http://localhost:8501
#   api      (FastAPI)    http://127.0.0.1:8000     belgeler: /docs
#
# Neden proje klasöründen çalışmıyor: macOS'un gizlilik koruması (TCC), launchd'nin başlattığı süreçlerin
# ~/Documents altını okumasına izin vermiyor ("Operation not permitted"). Bu yüzden servisler, gereken
# dosyaların bir kopyasıyla ~/Library/Application Support/ema-seslendirme altından çalışır. Kaynak yine bu
# projede; kod değişince ya da yeni bir ses eğitilince kopyayı "guncelle" yeniler.
# Uygulamanın belge çıktıları kalıcı ve kolay bulunur bir yerde durur: ~/Music/EMA Seslendirme
#
# Kullanım (Ema klasöründe):
#   bash servis/servisler.sh kur          # kopyala, Python ortamını kur, iki servisi başlat
#   bash servis/servisler.sh guncelle     # kod ya da ses dosyaları değişince: kopyayı yenile, ikisini yeniden başlat
#   bash servis/servisler.sh durum
#   bash servis/servisler.sh kayit [api|uygulama] [satır]
#   bash servis/servisler.sh durdur|baslat [api|uygulama]     # hizmet adı verilmezse ikisi
#   bash servis/servisler.sh kaldir       # servisleri ve çalışma kopyasını kaldır (belge çıktılarına dokunmaz)
#
# API ayarları: servis/api.env (isteğe bağlı) -> EMA_API_PORT, EMA_API_ANAHTARI, EMA_API_HOST, EMA_API_CORS.
# Ağa açmak için örnek api.env:   EMA_API_HOST=0.0.0.0   ve   EMA_API_ANAHTARI=uzun-gizli-bir-anahtar
set -euo pipefail

PROJE=$(cd "$(dirname "$0")/.." && pwd)
CALISMA="$HOME/Library/Application Support/ema-seslendirme"
ESKI_CALISMA="$HOME/Library/Application Support/ema-api"     # yalnızca API'nin olduğu eski kurulum
CIKTI="$HOME/Music/EMA Seslendirme"
ALAN="gui/$(id -u)"
HIZMETLER=(api uygulama)
# Proje ortamında test edilmiş sürümler; servisler aynılarını kullanır
PAKETLER=(torch ema-lightning normalizer-tr numpy huggingface-hub fastapi uvicorn pydantic starlette streamlit pypdf striprtf)

etiket() { echo "net.sofiatech.ema-$1"; }
plist() { echo "$HOME/Library/LaunchAgents/$(etiket "$1").plist"; }
kayit_dosyasi() { echo "$HOME/Library/Logs/ema-$1.log"; }
api_portu() {
  local p
  p=$(grep -E '^EMA_API_PORT=' "$CALISMA/env" 2> /dev/null | cut -d= -f2 | tr -d '"' || true)
  echo "${p:-8000}"
}
adres() { [ "$1" = api ] && echo "http://127.0.0.1:$(api_portu)" || echo "http://localhost:8501"; }
saglikli() {
  if [ "$1" = api ]; then curl -s -m 3 "$(adres api)/saglik" 2> /dev/null | grep -q hazir
  else curl -s -m 3 "$(adres uygulama)/_stcore/health" 2> /dev/null | grep -q ok; fi
}
secilen() { [ -n "${1:-}" ] && echo "$1" || echo "${HIZMETLER[@]}"; }

kopyala() {
  mkdir -p "$CALISMA/deneyler/akustik/agirliklar" "$CALISMA/deneyler/agirliklar" "$CALISMA/.streamlit"
  cp "$PROJE"/api.py "$PROJE"/app.py "$PROJE"/belge.py "$PROJE"/ses_kaydi.py "$PROJE"/ses_dosyasi.py "$CALISMA/"
  # Kişisel sesler isteğe bağlı (kisisel_sesler.ornek.py); yoksa eski kopya da kaldırılır
  if [ -f "$PROJE/kisisel_sesler.py" ]; then cp "$PROJE/kisisel_sesler.py" "$CALISMA/"; else rm -f "$CALISMA/kisisel_sesler.py"; fi
  cp "$PROJE/deneyler/akustik/ornekle.py" "$CALISMA/deneyler/akustik/"
  # Ses ağırlıkları: yalnızca bitmiş olanlar (.pt), eskiler silinir
  rsync -a --delete --include='ema_*.pt' --exclude='*' "$PROJE/deneyler/akustik/agirliklar/" "$CALISMA/deneyler/akustik/agirliklar/"
  rsync -a --delete --include='decoder_*.pt' --exclude='*' "$PROJE/deneyler/agirliklar/" "$CALISMA/deneyler/agirliklar/"

  cat > "$CALISMA/.streamlit/config.toml" <<'EOF'
[server]
headless = true
port = 8501
address = "127.0.0.1"       # yalnızca bu bilgisayar
maxUploadSize = 500         # MB
fileWatcherType = "none"    # servis kopyası değişmez; değişince "guncelle" yeniden başlatır

[browser]
gatherUsageStats = false
serverAddress = "localhost"
EOF

  # Ortak ortam: Homebrew'un yolu ffmpeg (MP3) için; API için yalnızca EMA_API_* değişkenleri
  {
    echo 'PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"'
    echo 'HF_HUB_OFFLINE=1'
    echo "EMA_CIKTI_KLASORU=\"$CIKTI\""
    [ -f "$PROJE/servis/api.env" ] && grep -E '^EMA_API_[A-Z_]+=' "$PROJE/servis/api.env" || true
  } > "$CALISMA/env"
  chmod 600 "$CALISMA/env"

  for h in "${HIZMETLER[@]}"; do
    {
      echo '#!/bin/bash'
      echo 'set -eu'
      echo 'CALISMA="$HOME/Library/Application Support/ema-seslendirme"'
      echo 'set -a; . "$CALISMA/env"; set +a'
      echo 'cd "$CALISMA"'
      if [ "$h" = api ]; then
        echo 'exec "$CALISMA/.venv/bin/python" "$CALISMA/api.py"'
      else
        # Çıktı klasörüne yazılamıyorsa (gizlilik koruması vb.) çalışma kopyasının içine yaz
        echo 'if ! { mkdir -p "$EMA_CIKTI_KLASORU" && touch "$EMA_CIKTI_KLASORU/.yazma_denemesi" && rm -f "$EMA_CIKTI_KLASORU/.yazma_denemesi"; } 2> /dev/null; then'
        echo '  echo "UYARI: $EMA_CIKTI_KLASORU yazılamıyor; çıktılar çalışma kopyasına yazılacak"'
        echo '  export EMA_CIKTI_KLASORU="$CALISMA/cikti/belgeler"'
        echo 'fi'
        echo 'exec "$CALISMA/.venv/bin/python" -m streamlit run "$CALISMA/app.py"'
      fi
    } > "$CALISMA/calistir_$h.sh"
    chmod +x "$CALISMA/calistir_$h.sh"
  done
  echo "Çalışma kopyası güncellendi: $CALISMA ($(ls "$CALISMA/deneyler/akustik/agirliklar" | wc -l | tr -d ' ') akustik model)"
}

ortam() {
  local surumler=()
  for p in "${PAKETLER[@]}"; do
    surumler+=("$(uv pip freeze --python "$PROJE/.venv/bin/python" | grep -iE "^$p==")")
  done
  [ -x "$CALISMA/.venv/bin/python" ] || uv venv --quiet --python 3.12 "$CALISMA/.venv"
  uv pip install --quiet --python "$CALISMA/.venv/bin/python" "${surumler[@]}"
  echo "Python ortamı hazır (${#surumler[@]} paket, proje ortamıyla aynı sürümler)"
}

plist_yaz() {
  local h=$1
  mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
  cat > "$(plist "$h")" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$(etiket "$h")</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$CALISMA/calistir_$h.sh</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$CALISMA</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <dict>
    <key>SuccessfulExit</key>
    <false/>
  </dict>
  <key>ThrottleInterval</key>
  <integer>15</integer>
  <key>ProcessType</key>
  <string>Interactive</string>
  <key>StandardOutPath</key>
  <string>$(kayit_dosyasi "$h")</string>
  <key>StandardErrorPath</key>
  <string>$(kayit_dosyasi "$h")</string>
</dict>
</plist>
EOF
  plutil -lint "$(plist "$h")" > /dev/null
}

hazir_bekle() {
  local h=$1
  for _ in $(seq 1 60); do
    saglikli "$h" && { echo "  $h hazır: $(adres "$h")"; return 0; }
    sleep 2
  done
  echo "  $h 2 dakikada hazır olmadı; günlük:"; tail -15 "$(kayit_dosyasi "$h")"; return 1
}

case "${1:-}" in
  kur)
    kopyala
    ortam
    for h in "${HIZMETLER[@]}"; do
      plist_yaz "$h"
      launchctl bootout "$ALAN/$(etiket "$h")" 2> /dev/null || true
      # bootout hemen bitmeyebilir; eski servis tamamen bırakılmadan yüklemek "Bootstrap failed: 5" verir
      for _ in $(seq 1 20); do launchctl print "$ALAN/$(etiket "$h")" > /dev/null 2>&1 || break; sleep 0.5; done
      launchctl bootstrap "$ALAN" "$(plist "$h")"
    done
    echo "Servisler kuruldu; modeller yükleniyor…"
    for h in "${HIZMETLER[@]}"; do hazir_bekle "$h"; done
    # Yalnızca API'nin olduğu eski kurulumun kopyası artık kullanılmıyor
    if [ -d "$ESKI_CALISMA" ]; then rm -rf "$ESKI_CALISMA" && echo "Eski çalışma kopyası kaldırıldı: $ESKI_CALISMA"; fi
    echo "Belge çıktıları: $CIKTI"
    ;;
  guncelle)
    kopyala
    for h in "${HIZMETLER[@]}"; do launchctl kickstart -k "$ALAN/$(etiket "$h")"; done
    echo "Servisler yeniden başlatıldı; modeller yükleniyor…"
    for h in "${HIZMETLER[@]}"; do hazir_bekle "$h"; done
    ;;
  durum)
    for h in "${HIZMETLER[@]}"; do
      if launchctl print "$ALAN/$(etiket "$h")" > /dev/null 2>&1; then
        bilgi=$(launchctl print "$ALAN/$(etiket "$h")" | grep -E "^\s+(state|pid|runs) =" | head -3 | tr -s ' \t' ' ' | tr '\n' ' ')
        echo "$h: $(saglikli "$h" && echo "yanıt veriyor" || echo "YANIT VERMİYOR") · $(adres "$h") ·$bilgi"
      else
        echo "$h: kurulu değil (kurmak için: bash servis/servisler.sh kur)"
      fi
    done
    echo "belge çıktıları: $CIKTI"
    ;;
  kayit) tail -n "${3:-30}" "$(kayit_dosyasi "${2:-uygulama}")" ;;
  durdur)
    for h in $(secilen "${2:-}"); do launchctl bootout "$ALAN/$(etiket "$h")" && echo "$h durduruldu"; done
    echo "(Oturum açılışında yine başlarlar; tamamen kaldırmak için: kaldir)"
    ;;
  baslat)
    for h in $(secilen "${2:-}"); do launchctl bootstrap "$ALAN" "$(plist "$h")" && hazir_bekle "$h"; done
    ;;
  kaldir)
    for h in "${HIZMETLER[@]}"; do
      launchctl bootout "$ALAN/$(etiket "$h")" 2> /dev/null || true
      rm -f "$(plist "$h")"
    done
    rm -rf "$CALISMA"
    echo "Servisler ve çalışma kopyası kaldırıldı. Belge çıktıları duruyor: $CIKTI"
    ;;
  *) sed -n '2,22p' "$0"; exit 1 ;;
esac

#!/bin/bash
# Seçilen bir sentetik sesin baştan sona hattı: VoxCPM2 veri üretimi -> Whisper denetimi -> çözücü uyarlaması
# -> eğitim verisi -> akustik model eğitimi -> ölçüm. Bitince Streamlit uygulaması sesi kendiliğinden bulur.
#
# Kullanım (Ema klasöründe, Mac uyumasın ve oturumdan bağımsız sürsün diye):
#   nohup caffeinate -is bash sesler/ses_hatti.sh kadin > sesler/kadin_hat.log 2>&1 &
# Kısa deneme:  ADET=60 ADIM=200 COZUCU_ADIM=100 DOGRULAMA=5 bash sesler/ses_hatti.sh kadin
# Her adım kaldığı yerden sürer; yarıda kalırsa aynı komutla yeniden başlatılabilir.
set -u
cd "$(dirname "$0")/.."
SES=$1
ADET=${ADET:-11000}
ADIM=${ADIM:-20000}
COZUCU_ADIM=${COZUCU_ADIM:-2000}
DOGRULAMA=${DOGRULAMA:-50}
VERI=deneyler/veri/$SES
export HF_HUB_OFFLINE=1 TQDM_DISABLE=1 PYTHONWARNINGS=ignore
adim() { echo "$(date '+%F %H:%M') $*"; }
dur() { adim "HATA: $1 (ayrıntı: $2)"; tail -5 "$2"; exit 1; }
mkdir -p "$VERI"

adim "1/6 veri üretimi ($ADET cümle, VoxCPM2)"
(cd sesler && .venv/bin/python uret_voxcpm.py --ses "$SES" --adet "$ADET") > "$VERI/uretim.log" 2>&1 || dur "veri üretimi" "$VERI/uretim.log"
adim "   $(grep -c . "$VERI/uretim.tsv") satır üretim kaydı"

adim "2/6 Whisper denetimi"
(cd deneyler && .venv/bin/python veri/denetle.py --veri "veri/$SES") > "$VERI/denetim.log" 2>&1 || dur "denetim" "$VERI/denetim.log"
grep -E "geçen|oranı" "$VERI/denetim.log" | sed 's/^/   /'

adim "3/6 çözücü uyarlaması"
# Denetimden geçen ilk 300 cümle tek bir kayda (yaklaşık 15-20 dk); öğretmen VoxCPM2'nin büyük çözücüsü
(cd deneyler && .venv/bin/python - "$SES" <<'EOF'
import csv, sys, numpy as np, soundfile as sf
ses = sys.argv[1]
gecen = [r["id"] for r in csv.DictReader(open(f"veri/{ses}/denetim.tsv", encoding="utf-8"), delimiter="\t",
                                          quoting=csv.QUOTE_NONE) if r["gecti"] == "1"][:300]
parcalar, sr = [], None
for k in gecen:
    x, sr = sf.read(f"veri/{ses}/wavs/{k}.wav", dtype="float32")
    parcalar += [x, np.zeros(int(0.3 * sr), np.float32)]
sf.write(f"veri/{ses}/cozucu_egitim.wav", np.concatenate(parcalar), sr, subtype="PCM_16")
print(f"{len(gecen)} cümle, {sum(len(p) for p in parcalar) / sr / 60:.1f} dk")
EOF
) > "$VERI/cozucu.log" 2>&1 || dur "çözücü verisi" "$VERI/cozucu.log"
(cd deneyler && .venv/bin/python cozucu_uyarla.py --egitim "veri/$SES/cozucu_egitim.wav" --dogrulama-sn 30 \
  --test "../sesler/secilen/$SES.wav" --adim "$COZUCU_ADIM" --ad "$SES") >> "$VERI/cozucu.log" 2>&1 || dur "çözücü uyarlaması" "$VERI/cozucu.log"
grep -E "^(ÖNCE|SONRA)" "$VERI/cozucu.log" | sed 's/^/   /'

adim "4/6 eğitim verisi (hizalama + latentler)"
rm -f "deneyler/akustik/veri_$SES"/parca_*.pt
(cd deneyler && .venv/bin/python akustik/veri_hazirla.py --veri "veri/$SES" --cikti "akustik/veri_$SES") \
  > "$VERI/hazirla.log" 2>&1 || dur "veri hazırlama" "$VERI/hazirla.log"
adim "   $(grep -v Warning "$VERI/hazirla.log" | tail -1)"

adim "5/6 akustik model eğitimi ($ADIM adım)"
(cd deneyler && .venv/bin/python akustik/egit.py --veri "akustik/veri_$SES" --adim "$ADIM" --olcum-araligi 2000 \
  --dogrulama "$DOGRULAMA" --ad "$SES" --cozucu "agirliklar/decoder_$SES.pt") > "$VERI/egitim.log" 2>&1 \
  || dur "eğitim" "$VERI/egitim.log"
grep -E "eğitim:|ÖNCE|SONRA" "$VERI/egitim.log" | sed 's/^/   /'

adim "6/6 ölçüm"
(cd deneyler && .venv/bin/python akustik/cfg_dene.py "akustik/agirliklar/ema_$SES.pt" --dogrulama "$DOGRULAMA" \
  --cozucu "agirliklar/decoder_$SES.pt" --veri "akustik/veri_$SES" --hedef "veri/$SES/wavs" \
  --ayarlar euler:32:2:0.7:-1 euler:16:2:0.7:-1 yeniden:4:1) > "$VERI/olcum.log" 2>&1 || dur "ölçüm" "$VERI/olcum.log"
grep -E "HEDEF|euler|yeniden" "$VERI/olcum.log" | sed 's/^/   /'
adim "TAMAMLANDI: Streamlit uygulamasını yeniden başlatınca '$SES' sesi listede görünür"

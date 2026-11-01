#!/bin/bash
# Whisper denetiminden sonraki tam hat: veri hazırlama -> akustik model eğitimi -> örnekleme ayarı ölçümü.
# Çözücü önceden o sese uyarlanmış olmalı: agirliklar/decoder_<ad>.pt (bkz. cozucu_uyarla.py).
#
# Kullanım (Ema/deneyler klasöründe, Mac uyumasın diye):
#   caffeinate -is bash akustik/egitim_hatti.sh <veri klasörü> <ad> [adım]
#   caffeinate -is bash akustik/egitim_hatti.sh veri/kayit_chatterbox benim_sesim 20000
# Çıktı: akustik/agirliklar/ema_<ad>.pt
set -u
cd "$(dirname "$0")/.."
VERI=${1:?"veri klasörü gerekli (örn. veri/kayit_chatterbox)"}
AD=${2:?"ses adı gerekli (çıktı: akustik/agirliklar/ema_<ad>.pt)"}
ADIM=${3:-20000}
export HF_HUB_OFFLINE=1
temiz() { grep -v "Warning\|warn\|WeightNorm" "$@"; }

echo "$(date +%H:%M) veri hazırlama başladı"
rm -f "akustik/veri_$AD"/parca_*.pt "akustik/veri_$AD/ozet.tsv"
.venv/bin/python akustik/veri_hazirla.py --veri "$VERI" --cikti "akustik/veri_$AD" > "akustik/veri_hazirla_$AD.log" 2>&1 \
  || { temiz "akustik/veri_hazirla_$AD.log" | tail -5; exit 1; }
echo "$(date +%H:%M) veri hazır: $(temiz "akustik/veri_hazirla_$AD.log" | tail -1)"

.venv/bin/python akustik/egit.py --veri "akustik/veri_$AD" --adim "$ADIM" --olcum-araligi 2000 --dogrulama 50 --ad "$AD" \
  --cozucu "agirliklar/decoder_$AD.pt" > "akustik/egit_$AD.log" 2>&1 || { temiz "akustik/egit_$AD.log" | tail -5; exit 1; }
echo "$(date +%H:%M) eğitim bitti"
grep -E "eğitim:|ÖNCE|SONRA|doğrulama \(" "akustik/egit_$AD.log"

.venv/bin/python akustik/cfg_dene.py "akustik/agirliklar/ema_$AD.pt" --dogrulama 50 --veri "akustik/veri_$AD" \
  --cozucu "agirliklar/decoder_$AD.pt" --hedef "$VERI/wavs" \
  --ayarlar yeniden:4:1 euler:16:2:0.7:-1 euler:32:2:0.7:-1 > "akustik/cfg_$AD.log" 2>&1 \
  || { temiz "akustik/cfg_$AD.log" | tail -5; exit 1; }
echo "$(date +%H:%M) ölçüm bitti"
grep -E "HEDEF|euler|yeniden" "akustik/cfg_$AD.log"

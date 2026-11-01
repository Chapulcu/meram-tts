#!/bin/bash
# Veri üretiminin ilerlemesini gösterir.  Kullanım (Ema/deneyler klasöründe): CIKTI=veri/kayit_chatterbox bash veri/durum.sh [toplam]
cd "$(dirname "$0")/.."
D=${CIKTI:?"üretim klasörü gerekli (örn. CIKTI=veri/kayit_chatterbox)"}
TOPLAM=${1:-11000}
biten=$(find "$D/wavs" -name "*.wav" | wc -l | tr -d ' ')
sure=$(awk -F'\t' 'NR > 1 { s += $2 } END { printf "%.1f", s / 3600 }' "$D/uretim.tsv")
echo "cümle: $biten / $TOPLAM  (%$((100 * biten / TOPLAM))), toplam ses: $sure saat"
# Hız, partinin başlangıcından (parti.log'un oluşturulma anı) beri üretilen dosyalardan
if [ -f "$D/loglar/parti.log" ]; then
  gecen_dk=$(( ($(date +%s) - $(stat -f %B "$D/loglar/parti.log")) / 60 ))
  yeni=$(find "$D/wavs" -name "*.wav" -newer "$D/loglar/parti.log" | wc -l | tr -d ' ')
  if [ "$gecen_dk" -ge 5 ] && [ "$yeni" -gt 0 ]; then
    kalan_dk=$(( (TOPLAM - biten) * gecen_dk / yeni ))
    echo "hız: saatte $((yeni * 60 / gecen_dk)) cümle; kalan süre ≈ $((kalan_dk / 60)) sa $((kalan_dk % 60)) dk"
  fi
fi
for c in mps cpu; do
  if pgrep -f "uret_chatterbox.py --cihaz $c" > /dev/null; then echo "$c işçisi: çalışıyor"; else echo "$c işçisi: durmuş"; fi
done
grep -h "HATA\|durdu\|bırakıldı" "$D"/loglar/*.log 2>/dev/null | tail -3
tail -2 "$D/loglar/parti.log" 2>/dev/null

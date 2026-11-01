#!/bin/bash
# Bir veri partisi üretir: GPU (MPS) ve işlemci aynı anda, ayrı cümle aralıklarında çalışır
# (birlikte tek başına GPU'dan ~1,6 kat hızlı). Çöken işçi kaldığı yerden yeniden başlar.
# Üretim bitince Whisper denetimi kendiliğinden çalışır.
#
# Gerekenler (ortam değişkeni): CHATTERBOX_PYTHON (chatterbox-tts kurulu Python), REF (sesinin 5-20 sn'lik
# temiz kaydı), CIKTI (çıktı klasörü). Başlatmak için (Ema/deneyler klasöründe), Mac uyumasın diye:
#   export CHATTERBOX_PYTHON=~/chatterbox/.venv/bin/python REF=~/sesim.wav CIKTI=veri/kayit_chatterbox
#   mkdir -p $CIKTI/loglar && nohup caffeinate -is bash veri/uret_parti.sh 11000 > $CIKTI/loglar/parti.log 2>&1 &
# İlerleme:  CIKTI=veri/kayit_chatterbox bash veri/durum.sh
# Durdurmak: pkill -f uret_parti.sh; pkill -f uret_chatterbox.py   (sonra aynı komutla kaldığı yerden sürer)
set -u
cd "$(dirname "$0")/.."
TOPLAM=${1:-11000}
MPS_PAYI=${2:-$((TOPLAM * 53 / 100))}   # paralel ölçümde GPU işin ~%53'ünü yapıyor
CB=${CHATTERBOX_PYTHON:?"chatterbox-tts kurulu Python'un yolu gerekli"}
REF=${REF:?"sesinin referans kaydı gerekli (5-20 sn)"}
CIKTI=${CIKTI:?"çıktı klasörü gerekli (örn. veri/kayit_chatterbox)"}
LOG="$CIKTI/loglar"
mkdir -p "$LOG"
export HF_HUB_OFFLINE=1 TQDM_DISABLE=1 PYTHONWARNINGS=ignore

isci() {  # cihaz başlangıç adet
  for deneme in $(seq 1 20); do
    "$CB" veri/uret_chatterbox.py --ref "$REF" --cikti "$CIKTI" --cihaz "$1" --baslangic "$2" --adet "$3" >> "$LOG/$1.log" 2>&1 && return 0
    echo "$(date '+%F %T') $1 işçisi durdu (deneme $deneme), 10 sn sonra kaldığı yerden devam" >> "$LOG/$1.log"
    sleep 10
  done
  echo "$(date '+%F %T') $1 işçisi 20 denemede bitiremedi, bırakıldı" >> "$LOG/$1.log"
}

echo "$(date '+%F %T') başladı: $TOPLAM cümle (GPU 0-$((MPS_PAYI - 1)), işlemci $MPS_PAYI-$((TOPLAM - 1)))"
isci mps 0 "$MPS_PAYI" &
isci cpu "$MPS_PAYI" "$((TOPLAM - MPS_PAYI))" &
wait
echo "$(date '+%F %T') üretim bitti, Whisper denetimi başlıyor"
.venv/bin/python veri/denetle.py --veri "$CIKTI" >> "$LOG/denetim.log" 2>&1
grep -E "denetlenen|geçen|oranı" "$LOG/denetim.log" | tail -4
echo "$(date '+%F %T') tamamlandı"

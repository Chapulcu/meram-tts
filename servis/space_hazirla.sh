#!/bin/bash
# Hugging Face Space klasörünü bu projeden üretir: uygulama kodu, seçilen seslerin ağırlıkları ve Docker ortamı.
# Klasör kendi başına çalışır ve olduğu gibi bir Hugging Face Docker Space'ine yüklenebilir.
#
# Kullanım (Ema klasöründe):
#   bash servis/space_hazirla.sh                          # huggingface_space/ , sesler: Elif, Kerem, Defne
#   bash servis/space_hazirla.sh kadin erkek kiz-cocuk ema
#   HEDEF=~/Desktop/ema-space bash servis/space_hazirla.sh kadin
# Sesler ses_kaydi.py'deki kimlikleriyle verilir; Space'te yalnızca bunlar, bu sırayla görünür (ilki varsayılan).
#
# Kaynak her zaman bu proje: kod ya da sesler değişince betiği yeniden çalıştırmak yeter. Hedef klasördeki eski
# dosyalar yenileriyle değiştirilir; .git klasörüne dokunulmaz (Space deposunu oraya klonladıysan bozulmaz).
# Şablonlar (Dockerfile, README, Streamlit ayarları): servis/space_sablon/
set -euo pipefail

PROJE=$(cd "$(dirname "$0")/.." && pwd)
HEDEF=${HEDEF:-$PROJE/huggingface_space}
SESLER=("$@")
[ ${#SESLER[@]} -eq 0 ] && SESLER=(kadin erkek kiz-cocuk)
# Proje ortamında test edilmiş sürümler aynen sabitlenir (torch Docker'da işlemci sürümünden kurulur)
PAKETLER=(torch ema-lightning normalizer-tr numpy huggingface-hub streamlit pypdf striprtf)

GECICI=$(mktemp -d)
trap 'rm -rf "$GECICI"' EXIT

chmod 755 "$GECICI"
cp -R "$PROJE/servis/space_sablon/." "$GECICI/"
cp "$PROJE"/app.py "$PROJE"/belge.py "$PROJE"/ses_dosyasi.py "$PROJE"/ses_kaydi.py "$PROJE/deneyler/akustik/ornekle.py" "$GECICI/"

mkdir -p "$GECICI/agirliklar"
for ses in "${SESLER[@]}"; do
  [ "$ses" = ema ] && continue   # EMA'nın kendi sesi dosya istemez; Hugging Face'ten iner
  dosya=${ses//-/_}
  akustik="$PROJE/deneyler/akustik/agirliklar/ema_$dosya.pt"
  cozucu="$PROJE/deneyler/agirliklar/decoder_$dosya.pt"
  if [ ! -f "$akustik" ] || [ ! -f "$cozucu" ]; then
    echo "HATA: '$ses' sesinin dosyaları yok: $akustik ve $cozucu" >&2
    exit 1
  fi
  cp "$akustik" "$cozucu" "$GECICI/agirliklar/"
done

for p in "${PAKETLER[@]}"; do
  uv pip freeze --python "$PROJE/.venv/bin/python" | grep -iE "^$p==" | sed 's/+.*//'
done > "$GECICI/requirements.txt"

# Space yalnızca seçilen sesleri, bu sırayla gösterir
LISTE=$(IFS=,; echo "${SESLER[*]}")
sed -i '' "s/{{SESLER_LISTESI}}/$LISTE/" "$GECICI/Dockerfile"

# README'deki ses listesi: ses_kaydi.py'deki adlar ve açıklamalardan
ADLAR=$(cd "$PROJE" && EMA_SESLER="$LISTE" EMA_AKUSTIK_KLASORU="$GECICI/agirliklar" EMA_COZUCU_KLASORU="$GECICI/agirliklar" \
  .venv/bin/python - "$GECICI/README.md" 2> /dev/null <<'EOF'
import sys
from pathlib import Path
import ses_kaydi
sesler = ses_kaydi.mevcut_sesler()
satirlar = [f"- **{s.ad}**: {s.aciklama}" for s in sesler]
yol = Path(sys.argv[1])
yol.write_text(yol.read_text(encoding="utf-8").replace("{{SESLER}}", "\n".join(satirlar)), encoding="utf-8")
print(", ".join(s.ad for s in sesler))
EOF
)

mkdir -p "$HEDEF"
rsync -a --delete --exclude .git "$GECICI/" "$HEDEF/"
echo "Space klasörü hazır: $HEDEF"
echo "  sesler: $ADLAR  ·  boyut: $(du -sh "$HEDEF" | cut -f1)"
echo "  gereksinimler: $(tr '\n' ' ' < "$HEDEF/requirements.txt")"

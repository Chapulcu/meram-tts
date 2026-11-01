#!/bin/bash
# GitHub'a yüklenecek depo klasörünü bu projeden üretir: yalnızca yayımlanacak dosyalar (kod, betikler, sentetik
# seslerin ağırlıkları, belgeler). Kişisel sesler, eğitim verisi, ortamlar ve çıktılar girmez; üretimden sonra
# bütün metin dosyaları kişisel izlere karşı taranır, bulunursa klasör yazılmaz.
#
# Kullanım (Ema klasöründe):
#   bash servis/github_hazirla.sh                      # github_repo/
#   HEDEF=~/Desktop/turkce-seslendirme bash servis/github_hazirla.sh
# Kaynak her zaman bu proje: değişiklikten sonra yeniden çalıştırmak yeter. Hedefteki .git klasörüne dokunulmaz.
set -euo pipefail

PROJE=$(cd "$(dirname "$0")/.." && pwd)
HEDEF=${HEDEF:-$PROJE/github_repo}
# Yayımlanan sesler (ses_kaydi.py'deki dosya adları); kişisel sesler (kisisel_sesler.py) hiçbir zaman eklenmez
SESLER=(kadin erkek kiz_cocuk)
# Kişisel iz sayılan kalıplar: bulunursa üretim durur
YASAK='talip|akhan|sailer99|/Users/|ARABISTAN|02_Calisma|storybook|@gmail'

cd "$PROJE"
DOSYALAR=(
  README.md LICENSE NOTICE .gitignore .python-version pyproject.toml uv.lock
  app.py api.py belge.py ses_kaydi.py ses_dosyasi.py main.py kisisel_sesler.ornek.py
  servis/servisler.sh servis/api_servisi.sh servis/space_hazirla.sh servis/github_hazirla.sh
  servis/space_sablon/Dockerfile servis/space_sablon/README.md servis/space_sablon/.streamlit/config.toml
  servis/space_sablon/.gitattributes servis/space_sablon/.dockerignore
  deneyler/requirements.txt deneyler/latent_uyumu.py deneyler/cozucu_uyarla.py deneyler/okuma_metni.txt
  deneyler/akustik/*.py deneyler/akustik/*.sh deneyler/veri/*.py deneyler/veri/*.sh
  sesler/requirements.txt sesler/*.py sesler/ses_hatti.sh sesler/dinleme_listesi.md
)
for s in "${SESLER[@]}"; do
  DOSYALAR+=("deneyler/akustik/agirliklar/ema_$s.pt" "deneyler/agirliklar/decoder_$s.pt"
             "sesler/secilen/$s.txt" "sesler/secilen/$s.wav")
done

GECICI=$(mktemp -d)
trap 'rm -rf "$GECICI"' EXIT
chmod 755 "$GECICI"
rsync -a --relative "${DOSYALAR[@]}" "$GECICI/"

# Güvenlik taraması: kişisel iz ya da gizli bilgi taşıyan dosya varsa dur (bu betiğin kendi kalıp satırı hariç)
if izler=$(grep -rIniE "$YASAK" "$GECICI" | grep -v "YASAK=" ); then
  echo "DURDU: yayımlanacak dosyalarda kişisel iz bulundu:" >&2
  echo "$izler" | sed "s|$GECICI/||" | cut -c1-160 >&2
  exit 1
fi
if buyuk=$(find "$GECICI" -type f -size +95M | sed "s|$GECICI/||"); [ -n "$buyuk" ]; then
  echo "DURDU: GitHub'ın 100 MB'lık dosya sınırına yakın dosyalar: $buyuk" >&2
  exit 1
fi

mkdir -p "$HEDEF"
rsync -a --delete --exclude .git "$GECICI/" "$HEDEF/"
echo "Depo klasörü hazır: $HEDEF"
echo "  $(find "$HEDEF" -type f -not -path '*/.git/*' | wc -l | tr -d ' ') dosya · $(du -sh --exclude=.git "$HEDEF" 2> /dev/null | cut -f1 || du -sh "$HEDEF" | cut -f1)"
echo "  sesler: ${SESLER[*]} · kişisel iz taraması: temiz"

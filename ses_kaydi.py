"""Seslerin ortak kaydı: Streamlit uygulaması da API sunucusu da sesleri buradan yükler.

Her ses aynı biçimde kullanılır:
    s = yukle("kadin")
    ses = s.seslendir("Merhaba", hiz=1.0, tohum=7, ornekleme=48000)      # float32 dizi
    for parca in s.parcalar("Uzun bir metin…", tohum=7):                 # akış: parça hazır oldukça
        ...

Yeni bir ses eklemek için SESLER listesine bir satır yeter. İnce ayarlı her sesin iki dosyası var:
deneyler/akustik/agirliklar/ema_<ad>.pt (akustik model) ve deneyler/agirliklar/decoder_<ad>.pt (o sese
uyarlanmış çözücü); sesler/ses_hatti.sh bir sesi bu adlarla yazar. Dosyaları olmayan ses listede görünmez,
dosyalar oluşunca uygulama ya da API yeniden başlatıldığında kendiliğinden belirir.
"""

import os
import random
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ema_lightning import EMA

KOK = Path(__file__).parent
sys.path.insert(0, str(KOK / "deneyler/akustik"))   # Hugging Face Space'te ornekle.py zaten kökte
import ornekle  # noqa: E402
from ornekle import Seslendirici  # noqa: E402

# Ağırlık klasörleri: projede deneyler/ altında; Hugging Face Space'te ortam değişkeniyle agirliklar/
AGIRLIK = Path(os.getenv("EMA_AKUSTIK_KLASORU", KOK / "deneyler/akustik/agirliklar"))
COZUCULER = Path(os.getenv("EMA_COZUCU_KLASORU", KOK / "deneyler/agirliklar"))
CIHAZ = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
# Yalnızca işlemcide (örn. Hugging Face'in ücretsiz donanımı): çözücü süreyi zaten belirliyor, 16 adım 32'ye çok
# yakın ve belirgin şekilde hızlı (2 çekirdekte 15x yerine 20x gerçek zaman)
EN_IYI = {**ornekle.EN_IYI, "adim": 16} if CIHAZ == "cpu" else dict(ornekle.EN_IYI)


@dataclass(frozen=True)
class Ses:
    kimlik: str                         # API'de kullanılan ad
    ad: str                             # arayüzde görünen ad
    akustik: Path | None = None         # None: paketin kendi EMA sesi
    cozucu: Path | None = None
    aciklama: str = ""

    @property
    def ozel(self):
        return self.akustik is not None

    @property
    def mevcut(self):
        return not self.ozel or (self.akustik.exists() and self.cozucu.exists())


def _ince_ayarli(kimlik, ad, aciklama, dosya_adi=None, akustik_adi=None):
    dosya_adi = dosya_adi or kimlik
    return Ses(kimlik, ad, AGIRLIK / f"ema_{akustik_adi or dosya_adi}.pt", COZUCULER / f"decoder_{dosya_adi}.pt",
               aciklama)


try:  # bu bilgisayara özel (kişisel) sesler; Hugging Face Space'e kopyalanmaz
    import kisisel_sesler
except ImportError:
    kisisel_sesler = None


def _kisisel(satirlar):
    return [_ince_ayarli(k, ad, aciklama, dosya, akustik) for k, ad, aciklama, dosya, akustik in satirlar]


# Sıra önemli: listede mevcut olan ilk ses varsayılandır. Sentetik seslerin adları (Elif, Kerem, Defne) arayüzde
# görünür; kimlikleri (kadin, erkek, kiz-cocuk) API'de kullanılır ve değişmez. bul() ikisini de tanır.
SESLER = [
    *(_kisisel(kisisel_sesler.ONDE) if kisisel_sesler else []),
    _ince_ayarli("kadin", "Elif", "Kadın sesi · otuzlu yaşlarda, sıcak ve sakin bir anlatıcı"),
    _ince_ayarli("erkek", "Kerem", "Erkek sesi · otuzlu yaşlarda, tok ve güven veren bir bariton"),
    _ince_ayarli("kiz-cocuk", "Defne", "Kız çocuk sesi · sekiz yaşında, neşeli ve tatlı", dosya_adi="kiz_cocuk"),
    Ses("ema", "EMA (orijinal)", aciklama="EMA Lightning'in kendi sesi"),
    *(_kisisel(kisisel_sesler.SONDA) if kisisel_sesler else []),
]
# Gösterilecek seslerin kimlikleri, sırasıyla (örn. Hugging Face Space'te "kadin,erkek,kiz-cocuk"); boşsa hepsi
SECILI = [k.strip() for k in os.getenv("EMA_SESLER", "").split(",") if k.strip()]


def mevcut_sesler():
    sesler = [s for s in SESLER if s.mevcut]
    if SECILI:
        sira = {k: i for i, k in enumerate(SECILI)}
        sesler = sorted((s for s in sesler if s.kimlik in sira), key=lambda s: sira[s.kimlik])
    return sesler


def varsayilan_ses():
    return mevcut_sesler()[0]


def bul(kimlik_ya_da_ad):
    """Kimliğe ya da görünen ada göre (büyük/küçük harf duyarsız) mevcut ses; yoksa None."""
    anahtar = kimlik_ya_da_ad.strip().lower().replace("_", "-")
    return next((s for s in mevcut_sesler() if anahtar in (s.kimlik, s.ad.lower())), None)


def rastgele_tohum():
    return random.SystemRandom().randrange(2**31)


class Sentezleyici:
    """Bir sesin motoru. İnce ayarlı sesler yönlendirmeli çok adımlı örnekleyiciyle (ornekle.py), EMA'nın kendi
    sesi paketin yoluyla çalışır. İnce ayarlı motor aynı anda tek isteği işler (kilit); EMA'nınki kendi
    zamanlayıcısıyla eşzamanlı istekleri birlikte yürütür."""

    def __init__(self, ses, cihaz=CIHAZ):
        self.ses = ses
        self.kilit = threading.Lock()
        if ses.ozel:
            self.motor = Seslendirici(ses.akustik, ses.cozucu, cihaz)
            self.motor("Isınma.", **EN_IYI)
        else:
            self.motor = EMA(device=cihaz)
            self.motor.say("Isınma.")  # ilk üretim GPU'yu ısıtır; ilk istek beklemesin

    def parcalar(self, metin, hiz=1.0, tohum=0, ornekleme=48000, adim=None, yonlendirme=None, sicaklik=None):
        """Metnin sesi, parça hazır oldukça float32 diziler halinde (akış)."""
        if not self.ses.ozel:
            yield from self.motor.stream(metin, speed=hiz, seed=tohum, sample_rate=ornekleme)
            return
        ayar = dict(EN_IYI)
        ayar.update({k: v for k, v in (("adim", adim), ("yonlendirme", yonlendirme), ("sicaklik", sicaklik))
                     if v is not None})
        with self.kilit:  # üretici erken bırakılırsa (istemci koptu) kilit de bırakılır
            yield from self.motor.parcalar(metin, hiz=hiz, tohum=tohum, sample_rate=ornekleme, **ayar)

    def seslendir(self, metin, **ayar):
        parcalar = list(self.parcalar(metin, **ayar))
        return np.concatenate(parcalar).astype(np.float32) if parcalar else np.zeros(0, dtype=np.float32)


_yuklenen, _yukleme_kilidi = {}, threading.Lock()


def yukle(kimlik):
    """Sesin motoru; ilk istekte yüklenir ve ısıtılır, sonra bellekte kalır."""
    ses = bul(kimlik)
    if ses is None:
        raise KeyError(kimlik)
    with _yukleme_kilidi:
        if ses.kimlik not in _yuklenen:
            _yuklenen[ses.kimlik] = Sentezleyici(ses)
        return _yuklenen[ses.kimlik]


def yuklenmis():
    return list(_yuklenen)

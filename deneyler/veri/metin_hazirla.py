"""Chatterbox ile seslendirilecek Türkçe cümle listesini hazırlar.

Kaynak: Mozilla Common Voice'un Türkçe cümle derlemi (CC0, kamu malı): metin/ klasöründeki
sentence-collector.txt (gündelik dil), setimes.txt (haber) ve wiki.tr.txt (ansiklopedik).

Her cümle normalizer-tr ile okunuş biçimine çevrilir (sayılar, tarihler, kısaltmalar yazıya döner),
çünkü EMA da metni modele aynı şekilde verir; Chatterbox'a da bu okunuş verilir ki ses ile metin
birebir eşleşsin.

Çalıştırmak için (Ema/deneyler klasöründe):
    .venv/bin/python veri/metin_hazirla.py --adet 60000
Çıktı: veri/cumleler.tsv  (id, kaynak, metin, okunus)
"""

import argparse
import random
import re
from pathlib import Path

from normalizer_tr import Normalizer

KLASOR = Path(__file__).parent
KAYNAKLAR = {"gundelik": ("sentence-collector.txt", 0.42), "haber": ("setimes.txt", 0.08), "wiki": ("wiki.tr.txt", 0.42)}
SAYI_ORANI = 0.08   # Common Voice cümlelerinde rakam yok; tarih, saat, tutar okumayı bu kalıplar öğretir
IZINLI = re.compile(r"^[A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû0-9 .,;:!?'\"()%&/\-]+$")

AYLAR = "Ocak Şubat Mart Nisan Mayıs Haziran Temmuz Ağustos Eylül Ekim Kasım Aralık".split()
GUNLER = "Pazartesi Salı Çarşamba Perşembe Cuma Cumartesi Pazar".split()
KALIN, SERT = set("aıou"), set("fstkçşhp")


def bulunma_eki(okunus):
    """'14:30' + 'da/de/ta/te': sayının okunuşunun son sesine göre ünlü ve ünsüz uyumu."""
    kelime = okunus.split()[-1].lower()
    unlu = next((c for c in reversed(kelime) if c in "aeıioöuü"), "e")
    return ("t" if kelime[-1] in SERT else "d") + ("a" if unlu in KALIN else "e")


def sayi_cumleleri(rng, normalizer, adet):
    """Tarih, saat, para, yüzde, ölçü, sıra sayısı ve telefon içeren kalıp cümleler."""
    def oku(s):
        return normalizer.normalize(s, ambiguity_policy="fallback").normalized_text

    def saat():
        s = f"{rng.randint(0, 23):02d}:{rng.choice([0, 5, 10, 15, 20, 30, 40, 45, 50])}".replace(":0", ":00")[:5]
        return s + "'" + bulunma_eki(oku(s))

    def tarih():
        return f"{rng.randint(1, 28)} {rng.choice(AYLAR)} {rng.randint(1950, 2035)}"

    def tutar():
        return f"{rng.randint(1, 999)}.{rng.randint(0, 999):03d}" if rng.random() < 0.5 else str(rng.randint(2, 999))

    kaliplar = [
        lambda: f"Toplantı {tarih()} tarihinde, saat {saat()} başlayacak.",
        lambda: f"Randevunuz {rng.randint(1, 28)} {rng.choice(AYLAR)} {rng.choice(GUNLER)} günü saat {saat()}.",
        lambda: f"Siparişinizin toplam tutarı {tutar()} TL.",
        lambda: f"Ürünün fiyatı {rng.randint(1, 999)},{rng.randint(10, 99)} lira oldu.",
        lambda: f"Bu yıl satışlarda %{rng.randint(1, 99)} oranında artış görüldü.",
        lambda: f"Şehrin nüfusu {rng.randint(1, 20)}.{rng.randint(0, 999):03d}.{rng.randint(0, 999):03d} kişiye ulaştı.",
        lambda: f"Hava sıcaklığı yarın {rng.randint(-15, 42)} derece olacak.",
        lambda: f"Evimiz {rng.randint(2, 30)}. katta, asansörün hemen yanında.",
        lambda: f"Okul {rng.randint(1850, 2024)} yılında kuruldu.",
        lambda: f"Bugün tam {rng.randint(2, 40)} km yürüdük.",
        lambda: f"Paketin ağırlığı {rng.randint(1, 99)} kg ve {rng.randint(1, 999)} gram.",
        lambda: f"Maç {rng.randint(0, 5)}-{rng.randint(0, 5)} sona erdi.",
        lambda: f"Telefon numaram 0{rng.randint(530, 559)} {rng.randint(100, 999)} {rng.randint(10, 99)} {rng.randint(10, 99)}.",
        lambda: f"Sınavdan {rng.randint(40, 100)} puan aldım.",
        lambda: f"Uçağımız saat {saat()} kalkıyor, lütfen geç kalmayın.",
        lambda: f"Kredi kartı borcunuz {tutar()} TL, son ödeme tarihi {rng.randint(1, 28)} {rng.choice(AYLAR)}.",
        lambda: f"{rng.randint(2, 12)} kişilik masa ayırtmak istiyorum.",
        lambda: f"Hesabınıza {tutar()} TL yatırıldı.",
        lambda: f"Kitabın {rng.randint(2, 40)}. bölümünü okuyorum.",
        lambda: f"Dolar bugün {rng.randint(30, 45)},{rng.randint(10, 99)} liradan işlem görüyor.",
    ]
    return [rng.choice(kaliplar)() for _ in range(adet)]


def temiz_cumleler(yol):
    for satir in yol.read_text(encoding="utf-8").splitlines():
        s = " ".join(satir.split()).replace("’", "'").replace("“", '"').replace("”", '"')
        if not (12 <= len(s) <= 180) or len(s.split()) < 2 or not IZINLI.match(s):
            continue
        if s[-1] not in ".!?":
            s += "."
        yield s


def main():
    parser = argparse.ArgumentParser(description="Seslendirilecek cümle listesini hazırla")
    parser.add_argument("--adet", type=int, default=60000)
    parser.add_argument("--tohum", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.tohum)

    havuzlar, gorulen = {}, set()
    for ad, (dosya, _) in KAYNAKLAR.items():
        cumleler = []
        for s in temiz_cumleler(KLASOR / "metin" / dosya):
            anahtar = s.lower()
            if anahtar not in gorulen:
                gorulen.add(anahtar)
                cumleler.append(s)
        rng.shuffle(cumleler)
        havuzlar[ad] = cumleler
        print(f"{ad}: {len(cumleler)} uygun cümle")
    normalizer = Normalizer()
    havuzlar["sayilar"] = sayi_cumleleri(rng, normalizer, int(args.adet * SAYI_ORANI) + 1)

    # Kaynakları oranlarına göre karıştır: listenin her başlangıç kısmı dengeli bir karışım olsun
    adlar = list(havuzlar)
    oranlar = [KAYNAKLAR[a][1] if a in KAYNAKLAR else SAYI_ORANI for a in adlar]
    secilen = []
    while len(secilen) < args.adet and any(havuzlar.values()):
        ad = rng.choices(adlar, weights=[o if havuzlar[a] else 0 for a, o in zip(adlar, oranlar)])[0]
        secilen.append((ad, havuzlar[ad].pop()))

    harf = 0
    with open(KLASOR / "cumleler.tsv", "w", encoding="utf-8") as f:
        f.write("id\tkaynak\tmetin\tokunus\n")
        for i, (ad, metin) in enumerate(secilen):
            try:
                okunus = normalizer.normalize(metin, ambiguity_policy="fallback").normalized_text
            except Exception:
                continue
            okunus = " ".join(okunus.split())
            harf += len(okunus)
            f.write(f"{i:06d}\t{ad}\t{metin}\t{okunus}\n")
    # EMA'nın parçalayıcısı saniyede ~18 harf varsayıyor
    print(f"\n{len(secilen)} cümle -> {KLASOR / 'cumleler.tsv'}")
    print(f"toplam {harf:,} harf, tahmini {harf / 18 / 3600:.0f} saat ses")


if __name__ == "__main__":
    main()

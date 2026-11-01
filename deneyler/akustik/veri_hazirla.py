"""Akustik model eğitimi için veriyi hazırlar: her cümle için metin, kelime süreleri ve latentler.

EMA'nın zaman çizelgesi kelime düzeyinde: her kelime (harfleri ve ardından gelen boşluk) birkaç kare
(25 Hz) alır, harfin kelime içindeki yeri süre tahmincisinin oranlarından gelir. Yani eğitim için harf
hizalaması gerekmez, kelimelerin seste nerede başladığını bilmek yeter.

1. Whisper denetiminden geçen her ses baştaki ve sondaki sessizlikten kırpılır, 16 kHz'e çevrilir.
2. Whisper kelime zamanlarını verir; Whisper'ın metni de EMA'nın metin işleyicisinden geçirilir
   ("54" -> "elli dört") ve cümleyle harf düzeyinde hizalanır; Whisper kelimeleri farklı bölse de
   ("sürgünü ile") her kelimenin başlangıcı bulunur. Benzerliği %90'ın altındaki cümle atlanır:
   yanlış hizalama modele yanlış konuşmayı öğretir.
3. Kelime i'nin kare sayısı = (i+1)'in başlangıcı - i'nin başlangıcı; ilk kelime 0'dan, son kelime sese
   kadar sürer.
4. VoxCPM2'nin kodlayıcısı sesi EMA'nın latentlerine çevirir.

Çalıştırmak için (Ema/deneyler klasöründe):
    .venv/bin/python akustik/veri_hazirla.py
Çıktı: akustik/veri/parca_XXXX.pt (her biri en çok 500 cümlelik liste) ve akustik/veri/ozet.tsv
"""

import argparse
import csv
import difflib
import sys
import time
from pathlib import Path

import mlx_whisper
import numpy as np
import torch
from ema_lightning.frontend import Frontend
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent.parent))
from latent_uyumu import voxcpm_vae, wav_oku  # noqa: E402

KOK = Path(__file__).parent.parent
WHISPER = "mlx-community/whisper-large-v3-turbo"
KARE_HZ = 25
EN_FAZLA_KARE = 250        # EMA'nın parçalayıcısı metni ~10 sn'lik parçalara böler
PARCA_BOYU = 500


def sessizligi_kirp(x, sr, esik_db=-45.0, bas_pay=0.10, son_pay=0.20):
    """20 ms'lik pencerelerde, en yüksek enerjinin esik_db altındaki baş ve son kısımları atar."""
    pencere = int(sr * 0.02)
    n = len(x) // pencere
    enerji = 10 * np.log10((x[:n * pencere].reshape(n, pencere) ** 2).mean(1) + 1e-10)
    sesli = np.where(enerji > enerji.max() + esik_db)[0]
    if not len(sesli):
        return x
    a = max(0, sesli[0] * pencere - int(bas_pay * sr))
    b = min(len(x), (sesli[-1] + 1) * pencere + int(son_pay * sr))
    return x[a:b]


def harfler(s):
    return "".join(c for c in s if c.isalpha())


def kelime_baslangiclari(sonuc, frontend, kelimeler):
    """Cümlenin her kelimesinin seste başladığı an (sn); eşleşme zayıfsa None.

    Whisper kelimeleri farklı bölebilir ("sürgünüyle" / "sürgünü ile") ya da rakam yazabilir ("54").
    Bu yüzden eşleştirme harf düzeyinde: Whisper'ın her harfine kendi kelimesi içinde doğrusal bir zaman
    verilir, iki harf dizisi hizalanır ve her kelimenin ilk harfinin zamanı okunur.
    """
    duyulan, zaman = [], []
    for segment in sonuc["segments"]:
        for w in segment.get("words", []):
            h = harfler(frontend(w["word"]))  # "54" -> "ellidört"
            for k in range(len(h)):
                zaman.append(w["start"] + (w["end"] - w["start"]) * k / len(h))
            duyulan.append(h)
    duyulan = "".join(duyulan)
    beklenen = "".join(harfler(k) for k in kelimeler)
    if not duyulan or not beklenen:
        return None
    eslesme = difflib.SequenceMatcher(None, beklenen, duyulan, autojunk=False)
    if eslesme.ratio() < 0.9:
        return None
    karsilik = np.full(len(beklenen), np.nan)
    for a, b, n in eslesme.get_matching_blocks():
        karsilik[a:a + n] = [zaman[b + j] for j in range(n)]
    gecerli = ~np.isnan(karsilik)
    karsilik = np.interp(np.arange(len(beklenen)), np.where(gecerli)[0], karsilik[gecerli])  # eşleşmeyen harfler
    ilk_harf = np.cumsum([0] + [len(harfler(k)) for k in kelimeler[:-1]])
    return [float(karsilik[min(i, len(beklenen) - 1)]) for i in ilk_harf]


def main():
    parser = argparse.ArgumentParser(description="Akustik model eğitim verisini hazırla")
    parser.add_argument("--veri", type=Path, required=True, help="üretilen seslerin klasörü (wavs/, denetim.tsv)")
    parser.add_argument("--cumleler", type=Path, default=KOK / "veri/cumleler.tsv")
    parser.add_argument("--cikti", type=Path, default=Path(__file__).parent / "veri")
    parser.add_argument("--adet", type=int, default=None, help="en çok kaç cümle (deneme için)")
    args = parser.parse_args()
    args.cikti.mkdir(parents=True, exist_ok=True)

    cihaz = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(hf_hub_download("canberkkkkkk/ema-lightning", "ema.pt"), map_location="cpu", weights_only=True)
    frontend = Frontend(ck["vocab"])
    vae = voxcpm_vae().to(cihaz)

    with open(args.cumleler, encoding="utf-8") as f:
        okunus = {s["id"]: s["okunus"] for s in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}
    with open(args.veri / "denetim.tsv", encoding="utf-8") as f:
        gecenler = [s["id"] for s in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE) if s["gecti"] == "1"]
    gecenler = sorted(gecenler)[:args.adet]

    ozet = open(args.cikti / "ozet.tsv", "w", encoding="utf-8")
    ozet.write("id\tdurum\tkare\tkelime\n")
    parca, parca_no, sayac = [], 0, {"tamam": 0}
    t0 = time.perf_counter()
    for i, kimlik in enumerate(gecenler, 1):
        x, sr = wav_oku(args.veri / "wavs" / f"{kimlik}.wav")
        # Chatterbox 24 kHz, VoxCPM2 48 kHz yazıyor; kodlayıcı 16 kHz istiyor
        x16 = sessizligi_kirp(resample_poly(x, 16000, sr).astype(np.float32) if sr != 16000 else x, 16000)
        metin = frontend(okunus[kimlik])
        kelimeler = metin.split(" ")
        T = int(np.ceil(len(x16) / 640))

        durum = "tamam"
        if T > EN_FAZLA_KARE or len(metin) > 250:
            durum = "cok_uzun"
        else:
            sonuc = mlx_whisper.transcribe(x16, path_or_hf_repo=WHISPER, language="tr", temperature=0.0,
                                           condition_on_previous_text=False, word_timestamps=True)
            baslangic = kelime_baslangiclari(sonuc, frontend, kelimeler)
            if baslangic is None:
                durum = "eslesme_zayif"
            else:
                # Kelime sınırları karelere: ilk kelime 0'dan başlar, sınırlar artan ve her kelime en az 1 kare
                sinir = [0] + [int(round(t * KARE_HZ)) for t in baslangic[1:]] + [T]
                for j in range(1, len(sinir)):
                    sinir[j] = max(sinir[j], sinir[j - 1] + 1)
                if sinir[-1] > T:  # sona sıkışan kelimeler sığmadı
                    durum = "hizalama_bozuk"
                else:
                    kelime_kare = np.diff(sinir)
                    if kelime_kare.max() > EN_FAZLA_KARE:
                        durum = "hizalama_bozuk"

        if durum == "tamam":
            with torch.no_grad():
                z = vae.encode(torch.from_numpy(x16)[None, None].to(cihaz), 16000)[0].T.cpu()[:T]
            parca.append({"id": kimlik, "metin": metin, "kelime_kare": torch.tensor(kelime_kare, dtype=torch.int16),
                          "z": z.to(torch.float16)})
            sayac["tamam"] += 1
        else:
            sayac[durum] = sayac.get(durum, 0) + 1
        ozet.write(f"{kimlik}\t{durum}\t{T}\t{len(kelimeler)}\n")

        if len(parca) == PARCA_BOYU or (i == len(gecenler) and parca):
            torch.save(parca, args.cikti / f"parca_{parca_no:04d}.pt")
            parca, parca_no = [], parca_no + 1
        if i % 100 == 0 or i == len(gecenler):
            print(f"{i}/{len(gecenler)}  {sayac}  ({time.perf_counter() - t0:.0f} sn)", flush=True)
    ozet.close()


if __name__ == "__main__":
    main()

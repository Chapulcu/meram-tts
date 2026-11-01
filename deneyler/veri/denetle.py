"""Chatterbox'ın ürettiği sesleri Whisper ile geri yazıya çevirip metinle karşılaştırır.

Büyük TTS modelleri arada kelime atlar, tekrarlar ya da uydurur; bu cümleler eğitim verisine girmemeli.
Her iki taraf da EMA'nın kendi metin işleyicisinden geçirilir (Whisper rakam yazar, okunuş yazıyla yazar),
sonra harf ve kelime hata oranları hesaplanır.

Çalıştırmak için (Ema/deneyler klasöründe):
    .venv/bin/python veri/denetle.py
Çıktı: <veri>/denetim.tsv  (id, cer, wer, harf_sn, gecti, whisper)
"""

import argparse
import csv
import re
import time
from pathlib import Path

import mlx_whisper
import torch
from ema_lightning.frontend import Frontend
from huggingface_hub import hf_hub_download

KLASOR = Path(__file__).parent
MODEL = "mlx-community/whisper-large-v3-turbo"
HARF_DISI = re.compile(r"[^a-zçğıöşüâîû ]")


def levenshtein(a, b):
    onceki = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        simdiki = [i]
        for j, y in enumerate(b, 1):
            simdiki.append(min(onceki[j] + 1, simdiki[j - 1] + 1, onceki[j - 1] + (x != y)))
        onceki = simdiki
    return onceki[-1]


def main():
    parser = argparse.ArgumentParser(description="Üretilen sesleri Whisper ile denetle")
    parser.add_argument("--veri", type=Path, required=True, help="üretilen seslerin klasörü")
    parser.add_argument("--cumleler", type=Path, default=KLASOR / "cumleler.tsv")
    parser.add_argument("--esik-cer", type=float, default=0.05, help="en fazla harf hata oranı")
    # Alt sınır düşük: kısa cümlelerde baştaki ve sondaki sessizlik hızı yapay olarak düşürüyor
    parser.add_argument("--harf-sn", type=float, nargs=2, default=(6.0, 22.0), help="kabul edilen konuşma hızı aralığı")
    args = parser.parse_args()

    vocab = torch.load(hf_hub_download("canberkkkkkk/ema-lightning", "ema.pt"), map_location="cpu", weights_only=True)["vocab"]
    frontend = Frontend(vocab)

    def sade(metin):
        return " ".join(HARF_DISI.sub(" ", frontend(metin)).split())

    with open(args.cumleler, encoding="utf-8") as f:
        okunus = {s["id"]: s["okunus"] for s in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}
    with open(args.veri / "uretim.tsv", encoding="utf-8") as f:
        uretilen = {s["id"]: float(s["sure_sn"]) for s in csv.DictReader(f, delimiter="\t")}

    cikti = args.veri / "denetim.tsv"
    bitmis = set()
    if cikti.exists():
        with open(cikti, encoding="utf-8") as f:
            bitmis = {s["id"] for s in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}
    yeni = not cikti.exists()

    t0 = time.perf_counter()
    with open(cikti, "a", encoding="utf-8") as log:
        if yeni:
            log.write("id\tcer\twer\tharf_sn\tgecti\twhisper\n")
        for kimlik, sure in uretilen.items():
            if kimlik in bitmis or not (args.veri / "wavs" / f"{kimlik}.wav").exists():
                continue
            sonuc = mlx_whisper.transcribe(str(args.veri / "wavs" / f"{kimlik}.wav"), path_or_hf_repo=MODEL,
                                           language="tr", temperature=0.0, condition_on_previous_text=False)
            duyulan = " ".join(sonuc["text"].split())
            ref, hyp = sade(okunus[kimlik]), sade(duyulan)
            # Harf hatası boşluksuz: "büyük babası" ile "büyükbabası" ses olarak aynı
            cer = levenshtein(ref.replace(" ", ""), hyp.replace(" ", "")) / max(len(ref.replace(" ", "")), 1)
            wer = levenshtein(ref.split(), hyp.split()) / max(len(ref.split()), 1)
            harf_sn = len(ref.replace(" ", "")) / max(sure, 1e-6)
            gecti = cer <= args.esik_cer and args.harf_sn[0] <= harf_sn <= args.harf_sn[1]
            log.write(f"{kimlik}\t{cer:.4f}\t{wer:.4f}\t{harf_sn:.1f}\t{int(gecti)}\t{duyulan}\n")
            log.flush()

    with open(cikti, encoding="utf-8") as f:
        hepsi = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    gecen = [s for s in hepsi if s["gecti"] == "1"]
    sure_gecen = sum(uretilen[s["id"]] for s in gecen)
    sure_hepsi = sum(uretilen[s["id"]] for s in hepsi)
    cerler = sorted(float(s["cer"]) for s in hepsi)
    print(f"denetlenen {len(hepsi)} cümle ({time.perf_counter() - t0:.0f} sn)")
    print(f"geçen: {len(gecen)} (%{100 * len(gecen) / max(len(hepsi), 1):.0f}), "
          f"{sure_gecen / 60:.1f} / {sure_hepsi / 60:.1f} dk ses")
    print(f"harf hata oranı: ortanca %{100 * cerler[len(cerler) // 2]:.1f}, ortalama %{100 * sum(cerler) / len(cerler):.1f}")
    print(f"ortalama kelime hata oranı: %{100 * sum(float(s['wer']) for s in hepsi) / len(hepsi):.1f}")


if __name__ == "__main__":
    main()

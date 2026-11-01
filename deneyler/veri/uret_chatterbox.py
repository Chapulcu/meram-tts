"""cumleler.tsv'deki okunuşları Chatterbox Multilingual V3 ile senin sesinle seslendirir.

Chatterbox (chatterbox-tts, Multilingual V3) kurulu ayrı bir Python ortamıyla çalışır:
    "$CHATTERBOX_PYTHON" veri/uret_chatterbox.py --ref sesim.wav --cikti veri/kayit_chatterbox --adet 50

Kaldığı yerden devam eder: var olan ses dosyalarını atlar. Her cümle kendi tohumuyla üretilir, yani
aynı cümle aynı ayarlarla her seferinde aynı sesi verir.
Çıktı: <cikti>/wavs/<id>.wav (24 kHz) ve <cikti>/uretim.tsv (id, süre, üretim süresi, cihaz)
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

KLASOR = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description="Cümleleri Chatterbox ile seslendir")
    parser.add_argument("--cumleler", type=Path, default=KLASOR / "cumleler.tsv")
    parser.add_argument("--ref", type=Path, required=True, help="sesinin referans kaydı (5-20 sn)")
    parser.add_argument("--cikti", type=Path, required=True)
    parser.add_argument("--baslangic", type=int, default=0, help="kaçıncı cümleden başlansın")
    parser.add_argument("--adet", type=int, default=50)
    parser.add_argument("--cihaz", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--cfg", type=float, default=0.5)
    args = parser.parse_args()

    wavs = args.cikti / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    with open(args.cumleler, encoding="utf-8") as f:
        satirlar = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))[args.baslangic:args.baslangic + args.adet]

    t0 = time.perf_counter()
    tts = ChatterboxMultilingualTTS.from_pretrained(device=args.cihaz, t3_model="v3")
    tts.prepare_conditionals(str(args.ref), exaggeration=args.exaggeration)
    print(f"model hazır ({args.cihaz}): {time.perf_counter() - t0:.0f} sn, referans: {args.ref.name}", flush=True)

    kayit = args.cikti / "uretim.tsv"
    yeni = not kayit.exists()
    toplam_ses = toplam_sure = 0.0
    with open(kayit, "a", encoding="utf-8") as log:
        if yeni:
            log.write("id\tsure_sn\turetim_sn\tcihaz\n")
        for i, satir in enumerate(satirlar, 1):
            yol = wavs / f"{satir['id']}.wav"
            if yol.exists():
                continue
            torch.manual_seed(int(satir["id"]))
            basla = time.perf_counter()
            try:
                wav = tts.generate(satir["okunus"], language_id="tr", exaggeration=args.exaggeration, cfg_weight=args.cfg)
            except Exception as hata:  # tek cümlenin hatası bütün üretimi durdurmasın
                print(f"[{satir['id']}] HATA: {type(hata).__name__}: {hata}", flush=True)
                continue
            gecen = time.perf_counter() - basla
            ses = wav.squeeze().cpu().numpy().astype(np.float32)
            # Önce geçici adla yaz: süreç yazarken çökerse yarım dosya "bitmiş" sanılıp atlanmasın
            gecici = wavs / f".{satir['id']}.yaziliyor"
            sf.write(gecici, ses, tts.sr, subtype="PCM_16", format="WAV")
            os.replace(gecici, yol)
            sure = len(ses) / tts.sr
            toplam_ses += sure
            toplam_sure += gecen
            log.write(f"{satir['id']}\t{sure:.2f}\t{gecen:.2f}\t{args.cihaz}\n")
            log.flush()
            if i % 10 == 0 or i == len(satirlar):
                print(f"{i}/{len(satirlar)}  {toplam_ses / 60:.1f} dk ses, {toplam_sure / 60:.1f} dk üretim "
                      f"(gerçek zamanın {toplam_ses / max(toplam_sure, 1e-9):.2f} katı)", flush=True)

    if toplam_sure:
        print(f"\nbitti: {toplam_ses:.0f} sn ses {toplam_sure:.0f} sn'de üretildi; "
              f"1 saat ses yaklaşık {toplam_sure / toplam_ses:.2f} saat sürer")


if __name__ == "__main__":
    main()

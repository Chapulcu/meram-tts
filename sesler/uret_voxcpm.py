"""cumleler.tsv'deki okunuşları VoxCPM2 ile seçilen sentetik sesle seslendirir (klonlama).

Referans: secilen/<ses>.wav ve söylediği metin (secilen/<ses>.txt'nin 2. satırı). Ses tarifi değil referans kayıt
verildiği için ses 10 saat boyunca aynı kalır. Şapkalı harfler düz harfe çevrilir (VoxCPM2 onları yutuyor).
Çıktı biçimi Chatterbox üretimiyle aynı, böylece denetim, veri hazırlama ve eğitim betikleri değişmeden çalışır.

    .venv/bin/python uret_voxcpm.py --ses kadin --adet 11000
Kaldığı yerden devam eder. Çıktı: ../deneyler/veri/<ses>/wavs/<id>.wav (48 kHz) ve uretim.tsv
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from voxcpm import VoxCPM

from aday_uret import sapkasiz

KLASOR = Path(__file__).parent
VERI = KLASOR.parent / "deneyler/veri"


def main():
    parser = argparse.ArgumentParser(description="Cümleleri VoxCPM2 ile seçilen sesle seslendir")
    parser.add_argument("--ses", required=True, help="secilen/ altındaki ses adı (örn. kadin)")
    parser.add_argument("--cumleler", type=Path, default=VERI / "cumleler.tsv")
    parser.add_argument("--baslangic", type=int, default=0)
    parser.add_argument("--adet", type=int, default=11000)
    args = parser.parse_args()

    referans = KLASOR / "secilen" / f"{args.ses}.wav"
    istem = (KLASOR / "secilen" / f"{args.ses}.txt").read_text(encoding="utf-8").splitlines()[1]
    cikti = VERI / args.ses
    wavs = cikti / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    with open(args.cumleler, encoding="utf-8") as f:
        satirlar = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))[args.baslangic:args.baslangic + args.adet]

    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=False, device="mps")
    sr = model.tts_model.sample_rate
    kayit = cikti / "uretim.tsv"
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
                ses = model.generate(text=sapkasiz(satir["okunus"]), prompt_wav_path=str(referans), prompt_text=istem,
                                     reference_wav_path=str(referans), cfg_value=2.0, inference_timesteps=10)
            except Exception as hata:  # tek cümlenin hatası bütün üretimi durdurmasın
                print(f"[{satir['id']}] HATA: {type(hata).__name__}: {hata}", flush=True)
                continue
            gecen = time.perf_counter() - basla
            ses = np.asarray(ses, dtype=np.float32)
            gecici = wavs / f".{satir['id']}.yaziliyor"   # yarım dosya "bitmiş" sanılmasın
            sf.write(gecici, ses, sr, subtype="PCM_16", format="WAV")
            os.replace(gecici, yol)
            sure = len(ses) / sr
            toplam_ses += sure
            toplam_sure += gecen
            log.write(f"{satir['id']}\t{sure:.2f}\t{gecen:.2f}\tmps\n")
            log.flush()
            if i % 50 == 0 or i == len(satirlar):
                print(f"{i}/{len(satirlar)}  {toplam_ses / 3600:.2f} sa ses, {toplam_sure / 3600:.2f} sa üretim "
                      f"(gerçek zamanın {toplam_ses / max(toplam_sure, 1e-9):.2f} katı)", flush=True)


if __name__ == "__main__":
    main()

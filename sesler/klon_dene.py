"""Seçilen sesleri sabitler ve klonlamayla uzun, zorlayıcı bir metinde aynı kaldıklarını dener.

Veri üretiminde ses tarifiyle değil, seçilen adayın kaydı referans verilerek üretilecek ("ultimate cloning":
referans ses + söylediği metin). Böylece 10 saat boyunca ses aynı kalır. Bu betik:
  1. seçilen adayları secilen/<ad>.wav olarak kopyalar (söylediği metinle birlikte),
  2. her biri için iki farklı tohumla uzun bir deneme metni üretir: secilen/<ad>_deneme<n>.wav

    .venv/bin/python klon_dene.py kadin=kadin_t1_d1 erkek=erkek_t1_d2 kiz_cocuk=kiz_cocuk_t1_d1
"""

import shutil
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from voxcpm import VoxCPM

from aday_uret import METIN, sapkasiz

KLASOR = Path(__file__).parent
DENEME = ("Dün akşam saat 19:45'te kapı çaldı. Kim olabilir ki bu saatte? Açtığımda karşımda yıllardır görmediğim "
          "eski komşumuz duruyordu! Elinde, 3 Mart 1998 tarihli, sararmış bir mektup vardı. Mektubu okurken "
          "gözlerim doldu, çünkü içinde annemin el yazısıyla yazılmış, hiç bilmediğim bir hikâye anlatılıyordu.")


def main():
    secimler = dict(a.split("=") for a in sys.argv[1:])
    (KLASOR / "secilen").mkdir(exist_ok=True)
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=False, device="mps")
    sr = model.tts_model.sample_rate
    for ad, aday in secimler.items():
        referans = KLASOR / "secilen" / f"{ad}.wav"
        shutil.copy(KLASOR / "adaylar" / f"{aday}.wav", referans)
        (KLASOR / "secilen" / f"{ad}.txt").write_text(f"{aday}\n{sapkasiz(METIN)}\n", encoding="utf-8")
        for n in (1, 2):
            torch.manual_seed(n)
            basla = time.perf_counter()
            wav = model.generate(text=sapkasiz(DENEME), prompt_wav_path=str(referans), prompt_text=sapkasiz(METIN),
                                 reference_wav_path=str(referans), cfg_value=2.0, inference_timesteps=10)
            sf.write(KLASOR / "secilen" / f"{ad}_deneme{n}.wav", wav, sr)
            print(f"{ad}_deneme{n}: {len(wav) / sr:.1f} sn ses, {time.perf_counter() - basla:.1f} sn'de", flush=True)


if __name__ == "__main__":
    main()

"""EMA Lightning ile Türkçe metni sese çevirir.

Kullanım:
    uv run main.py "Okunacak metin"            # cikti/ses.wav dosyasına yazar
    uv run main.py "Okunacak metin" --cal      # üretirken hoparlörden çalar
"""

import argparse
import time
from pathlib import Path

import torch
from ema_lightning import EMA

ORNEK = "Merhaba! Bu ses, bilgisayarınızda internetsiz olarak üretildi. Bugün 30 Ekim 2026, saat 14:30."


def main():
    parser = argparse.ArgumentParser(description="Türkçe metinden ses üret")
    parser.add_argument("metin", nargs="?", default=ORNEK)
    parser.add_argument("--cikti", default="cikti/ses.wav", help="WAV dosyasının yolu")
    parser.add_argument("--hiz", type=float, default=1.0, help="konuşma hızı, 0.25 ile 4 arası")
    parser.add_argument("--seed", type=int, default=0, help="aynı seed her seferinde aynı sesi verir")
    parser.add_argument("--cal", action="store_true", help="dosyaya yazmak yerine hoparlörden çal")
    args = parser.parse_args()

    # Paket kendiliğinden yalnızca NVIDIA GPU'yu seçiyor; Mac'te Apple GPU (MPS) CPU'dan ~5 kat hızlı
    tts = EMA(device="mps" if torch.backends.mps.is_available() else "auto")

    if args.cal:
        import sounddevice as sd

        with sd.OutputStream(samplerate=48000, channels=1, dtype="float32") as hoparlor:
            for parca in tts.stream(args.metin, speed=args.hiz, seed=args.seed):
                hoparlor.write(parca)
        return

    Path(args.cikti).parent.mkdir(parents=True, exist_ok=True)
    basla = time.perf_counter()
    ses = tts.say(args.metin, speed=args.hiz, seed=args.seed, path=args.cikti)
    sure = time.perf_counter() - basla
    print(f"{args.cikti}: {ses.duration:.1f} sn ses, {sure:.2f} sn'de üretildi "
          f"({ses.duration / sure:.0f}x gerçek zaman)")


if __name__ == "__main__":
    main()

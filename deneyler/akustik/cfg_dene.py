"""Örnekleme ayarlarını karşılaştırır: adım sayısı, yöntem ve yönlendirme (CFG) gücü.

Doğrulama cümlelerinde (egit.py ile aynı ayrım) her ayar için:
  harf hatası (Whisper), ortanca perde, perde oynaklığı (tonlamanın canlılığı), düzenlilik (harmonik netliği).
Aynı ölçüler hedef seste (Chatterbox) de hesaplanır: amaç hedefe yaklaşmak.
000021 cümlesinin her ayardaki sesi cikti/teshis/ klasörüne yazılır.

    .venv/bin/python akustik/cfg_dene.py akustik/agirliklar/ema_cfg.pt --ayarlar euler:16:1 euler:16:2 euler:16:3
"""

import argparse
import random
import sys
from pathlib import Path

import mlx_whisper
import numpy as np
import torch
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from egit import AKUSTIK, COZUCU, harfler, levenshtein, ornek_hazirla  # noqa: E402
from latent_uyumu import wav_oku, wav_yaz  # noqa: E402
from ornekle import Seslendirici, ses_ozellikleri  # noqa: E402

WHISPER = "mlx-community/whisper-large-v3-turbo"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agirlik", type=Path)
    parser.add_argument("--ayarlar", nargs="+", default=["yeniden:4:1", "euler:32:1"], help="yöntem:adım:yönlendirme[:sıcaklık:sway]")
    parser.add_argument("--dogrulama", type=int, default=10)
    parser.add_argument("--cozucu", type=Path, default=COZUCU)
    parser.add_argument("--veri", type=Path, default=AKUSTIK / "veri", help="egit.py'nin kullandığı hazır veri")
    parser.add_argument("--hedef", type=Path, required=True, help="hedef seslerin klasörü (örn. veri/kadin/wavs)")
    args = parser.parse_args()

    s = Seslendirici(args.agirlik, args.cozucu)
    frontend = s.tts._frontend
    kayitlar = [k for p in sorted(args.veri.glob("parca_*.pt")) for k in torch.load(p, weights_only=False)]
    ornekler = [o for o in (ornek_hazirla(k, s.model.stoi) for k in kayitlar) if o is not None]
    random.Random(1).shuffle(ornekler)
    dogrulama = ornekler[:args.dogrulama]

    def harf_hatasi(ses, metin, sr=48000):
        duyulan = mlx_whisper.transcribe(resample_poly(ses, 16000, sr).astype(np.float32), path_or_hf_repo=WHISPER,
                                         language="tr", temperature=0.0, condition_on_previous_text=False)["text"]
        a, b = harfler(metin), harfler(frontend(duyulan))
        return levenshtein(a, b) / max(len(a), 1)

    def satir(ad, olcumler):
        o = np.array(olcumler, dtype=float)
        print(f"{ad:28s} hata %{100 * np.mean(o[:, 0]):5.1f}   perde {np.nanmedian(o[:, 1]):4.0f} Hz   "
              f"oynaklık {np.nanmean(o[:, 2]):4.2f} yt   düzenlilik {np.nanmean(o[:, 3]):.3f}", flush=True)

    hedef = []
    for o in dogrulama:
        x, sr = wav_oku(args.hedef / f"{o['id']}.wav")
        hedef.append((harf_hatasi(x, o["metin"], sr), *ses_ozellikleri(x, sr)))
    satir("HEDEF (öğretmen)", hedef)

    model_adi = args.agirlik.stem.replace("ema_", "")
    for ayar in args.ayarlar:
        yontem, adim, w, *ek = ayar.split(":")
        sicaklik, sway = (float(ek[0]), float(ek[1])) if ek else (1.0, 0.0)
        olcumler = []
        for o in dogrulama:
            ses = s(o["metin"], adim=int(adim), yonlendirme=float(w), yontem=yontem, sicaklik=sicaklik, sway=sway)
            olcumler.append((harf_hatasi(ses, o["metin"]), *ses_ozellikleri(ses)))
            if o["id"] == "000021":
                wav_yaz(AKUSTIK / f"cikti/teshis/000021_7_{model_adi}_{yontem}{adim}_w{w}_s{sicaklik}_sw{sway}.wav", ses, 48000)
        satir(f"{yontem} {adim}, w={w}, s={sicaklik}, sw={sway}", olcumler)


if __name__ == "__main__":
    main()

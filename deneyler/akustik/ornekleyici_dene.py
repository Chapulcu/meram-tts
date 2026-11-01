"""Tanı aracı: ince ayarlı modelin anlaşılırlığı örnekleyiciye mi bağlı, eğitimin kendisine mi?

EMA'nın örnekleyicisi damıtılmış öğrenci için tasarlandı: dört anda (0, ¼, ½, ¾) latenti doğrudan tahmin edip
yeniden gürültüler. Düz akış eşlemeyle eğitilen bir model ise asıl olarak çok adımlı Euler çözümüyle örneklenir.
Bu betik aynı ağırlıkları her iki yolla seslendirir ve Whisper harf hatasını karşılaştırır.

    .venv/bin/python akustik/ornekleyici_dene.py akustik/agirliklar/ema_kadin.pt --adimlar 4 8 16 32
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from ema_lightning import EMA
from ema_lightning.decoder import load_decoder
from ema_lightning.frontend import Frontend
from ema_lightning.model import load_acoustic, rope

sys.path.insert(0, str(Path(__file__).parent))
from egit import AKUSTIK, COZUCU, harfler, levenshtein, ornek_hazirla, perde  # noqa: E402
import mlx_whisper  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402


@torch.no_grad()
def euler_latentleri(model, p, adim, cihaz, tohum=0):
    """Bir parçanın (Engine.plan'dan geçmiş) latentleri, `adim` adımlı Euler ile."""
    m = torch.ones(1, p.letters, dtype=torch.bool, device=cihaz)
    h, dur = p.h[None].to(cihaz), p.dur[None].to(cihaz)
    cw, wstart = p.cw[None].to(cihaz), p.wstart[None].to(cihaz)
    fw, fp = p.fw[None].to(cihaz), p.fp[None].to(cihaz)
    c = dur.clamp(min=1e-4) * m
    done = c.cumsum(-1)
    before = done - c
    word = cw.clamp(min=0)
    total = torch.zeros_like(c).scatter_add_(1, word, c).gather(1, word)
    cp = ((done - before.gather(1, wstart) - 0.5 * c) / total.clamp(min=1e-8)).clamp(0.0, 1.0) * m
    cond = model.aligner(h, cw, cp, fw, fp, m, cw.shape[1])
    cos, sin = rope(torch.arange(fw.shape[1], device=cihaz), model.dh)
    fmask = torch.ones(1, fw.shape[1], dtype=torch.bool, device=cihaz)
    x = torch.randn(1, fw.shape[1], model.latent_dim, generator=torch.Generator().manual_seed(tohum)).to(cihaz)
    for k in range(adim):
        t = torch.full((1,), k / adim, device=cihaz)
        x = x + model.backbone(x, cond, t, fmask, cos, sin) / adim
    return x[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agirlik", type=Path)
    parser.add_argument("--adimlar", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--adet", type=int, default=10)
    args = parser.parse_args()
    cihaz = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = load_acoustic(args.agirlik, cihaz)
    cozucu = load_decoder(COZUCU, cihaz)
    frontend = Frontend(model.vocab)
    tts = EMA._from_parts(model, cozucu, frontend, cihaz)
    kayitlar = [k for p in sorted((AKUSTIK / "veri").glob("parca_*.pt")) for k in torch.load(p, weights_only=False)]
    import random
    ornekler = [o for o in (ornek_hazirla(k, model.stoi) for k in kayitlar) if o is not None]
    random.Random(1).shuffle(ornekler)
    metinler = [o["metin"] for o in ornekler[:args.adet]]   # egit.py'nin doğrulama kümesi

    def olc(ses, metin):
        duyulan = mlx_whisper.transcribe(resample_poly(ses, 1, 3).astype(np.float32),
                                         path_or_hf_repo="mlx-community/whisper-large-v3-turbo", language="tr",
                                         temperature=0.0, condition_on_previous_text=False)["text"]
        a, b = harfler(metin), harfler(frontend(duyulan))
        return levenshtein(a, b) / max(len(a), 1), perde(ses)

    sonuc = {"EMA örnekleyici (4 adım, yeniden gürültüleme)": [olc(tts.say(m, seed=0).audio, m) for m in metinler]}
    for adim in args.adimlar:
        olcumler = []
        for m in metinler:
            pieces = tts._pieces(m, 1.0, 0)
            tts._engine.plan(pieces, 1.0)
            z = torch.cat([euler_latentleri(model, p, adim, cihaz) for p in pieces])
            ses = cozucu(z.T[None].contiguous()).reshape(-1).cpu().numpy()
            olcumler.append(olc(ses, m))
        sonuc[f"Euler, {adim} adım"] = olcumler
    for ad, o in sonuc.items():
        print(f"{ad:48s} harf hatası %{100 * np.mean([h for h, _ in o]):5.1f}   perde {np.nanmedian([p for _, p in o]):.0f} Hz")


if __name__ == "__main__":
    main()

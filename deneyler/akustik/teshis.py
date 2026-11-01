"""Tanı: robotik ses çözücüden mi geliyor, akustik modelden mi?

Aynı cümle beş yoldan sese çevrilir; dinleyince hangi parçanın bozduğu anlaşılır:
  1_hedef                  Chatterbox'ın orijinali (eğitim hedefi)
  2_hedef_kucuk_cozucu     hedefin latentleri  -> senin küçük çözücün          (yalnız çözücünün etkisi)
  3_hedef_buyuk_cozucu     hedefin latentleri  -> VoxCPM2'nin büyük çözücüsü    (latent biçiminin tavanı)
  4_model_buyuk_cozucu     akustik modelin latentleri -> büyük çözücü          (yalnız akustik modelin etkisi)
  5_model_kucuk_cozucu     akustik modelin latentleri -> küçük çözücü          (şu an duyduğun tam hat)

    .venv/bin/python akustik/teshis.py akustik/agirliklar/ema_kadin.pt 000021 000047 \
        --cozucu agirliklar/decoder_kadin.pt --hedef veri/kadin/wavs --veri akustik/veri_kadin
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
from ema_lightning import EMA
from ema_lightning.decoder import load_decoder
from ema_lightning.frontend import Frontend
from ema_lightning.model import load_acoustic

sys.path.insert(0, str(Path(__file__).parent.parent))
from latent_uyumu import voxcpm_vae, wav_yaz  # noqa: E402

AKUSTIK = Path(__file__).parent
KOK = AKUSTIK.parent


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agirlik", type=Path)
    parser.add_argument("idler", nargs="+")
    parser.add_argument("--cozucu", type=Path, required=True, help="o sese uyarlanmış küçük çözücü")
    parser.add_argument("--hedef", type=Path, required=True, help="hedef seslerin klasörü (wavs)")
    parser.add_argument("--veri", type=Path, default=AKUSTIK / "veri", help="egit.py'nin kullandığı hazır veri")
    args = parser.parse_args()
    cihaz = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cikti = AKUSTIK / "cikti/teshis"
    cikti.mkdir(parents=True, exist_ok=True)

    model = load_acoustic(args.agirlik, cihaz)
    kucuk = load_decoder(args.cozucu, cihaz)
    vae = voxcpm_vae().to(cihaz)
    tts = EMA._from_parts(model, kucuk, Frontend(model.vocab), cihaz)
    yakalanan = []
    asil_think = tts._engine.think

    def think(parcalar):
        asil_think(parcalar)
        yakalanan.extend(p.latents for p in parcalar)

    tts._engine.think = think

    kayitlar = {k["id"]: k for p in sorted(args.veri.glob("parca_*.pt")) for k in torch.load(p, weights_only=False)}
    for kimlik in args.idler:
        k = kayitlar[kimlik]
        shutil.copy(args.hedef / f"{kimlik}.wav", cikti / f"{kimlik}_1_hedef.wav")
        z_hedef = k["z"].float().T[None].to(cihaz)                          # [1, 64, T]
        yakalanan.clear()
        tts.say(k["metin"], seed=0)
        z_model = torch.cat(yakalanan).T[None].contiguous()                 # [1, 64, T']
        for ad, z in [("hedef", z_hedef), ("model", z_model)]:
            wav_yaz(cikti / f"{kimlik}_{'2' if ad == 'hedef' else '5'}_{ad}_kucuk_cozucu.wav",
                    kucuk(z).reshape(-1).cpu().numpy(), 48000)
            wav_yaz(cikti / f"{kimlik}_{'3' if ad == 'hedef' else '4'}_{ad}_buyuk_cozucu.wav",
                    vae.decode(z).reshape(-1).cpu().numpy(), 48000)
        print(f"{kimlik}: {k['metin']}")
    print(f"dosyalar: {cikti}")


if __name__ == "__main__":
    main()

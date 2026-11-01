"""EMA'nın küçük çözücüsünü (decoder.pt, 3M parametre) senin sesine uyarlar.

Öğretmen VoxCPM2'nin büyük AudioVAE çözücüsü: aynı latentlerden onun ürettiği 48 kHz sesi küçük çözücü
taklit etmeyi öğrenir. Kayıp çok çözünürlüklü STFT + log-mel; GAN yok, öğrenme hızı düşük, kısa eğitim.

Veri: eğitim için Chatterbox klonunun ilk kısmı; ölçüm için klonun ayrılan son kısmı (görülmemiş cümleler)
ve gerçek kayıt (hiç görülmemiş, asıl ses).

Çalıştırmak için (Ema/deneyler klasöründe):
    .venv/bin/python cozucu_uyarla.py --ad kadin --egitim a.wav b.m4a --dogrulama-sn 30 --test gercek.wav
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from ema_lightning.decoder import load_decoder
from huggingface_hub import hf_hub_download

from latent_uyumu import CIKTI, mel_mesafesi, ses_dosyasi_16k, voxcpm_vae, wav_oku, wav_yaz, yeniden_ornekle

AGIRLIKLAR = Path(__file__).parent / "agirliklar"
HOP = 1920          # bir latent karesi = 1920 örnek (48 kHz'de 40 ms)
KIRPMA = 48         # eğitimde bir örnek kaç kare (1.92 sn)
KENAR = 8           # kayıpta kenarlardan atılan kare; çözücü her yana 4 kare bakıyor


# ---------- veri ----------

@torch.no_grad()
def latent_ve_ogretmen(vae, ses_16k, cihaz, parca=500, baglam=25):
    """16 kHz ses -> VoxCPM2 latentleri [64, T] ve öğretmenin 48 kHz sesi [T*1920].

    VoxCPM2'nin çözücüsü nedensel; uzun sesi parça parça, her parçaya solundan bağlam vererek çözüyoruz.
    """
    # Kodlayıcı da nedensel ve parça parça çalışıyor: MPS tek evrişimde ~16,7 milyon örneği aşamıyor (~17 dk ses)
    adim, sol = 60 * 16000, 2 * 16000                                            # ikisi de 640'ın katı (1 kare)
    x = torch.from_numpy(ses_16k)
    zs = []
    for s in range(0, len(x), adim):
        a = max(0, s - sol)
        z = vae.encode(x[a:s + adim][None, None].to(cihaz), 16000)
        zs.append(z[..., (s - a) // 640:])
    z = torch.cat(zs, -1)                                                        # [1, 64, T]
    T, sesler = z.shape[-1], []
    for s in range(0, T, parca):
        a = max(0, s - baglam)
        y = vae.decode(z[..., a:s + parca]).reshape(-1)
        sesler.append(y[(s - a) * HOP:])
    return z[0].cpu(), torch.cat(sesler)[:T * HOP].cpu()


# ---------- kayıplar ----------

def mel_filtreleri(sr, n_fft, n_mels):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    kutular = np.floor((n_fft + 1) * mel2hz(np.linspace(0, hz2mel(sr / 2), n_mels + 2)) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        a, b, c = kutular[i:i + 3]
        fb[i, a:b] = (np.arange(a, b) - a) / max(b - a, 1)
        fb[i, b:c] = (c - np.arange(b, c)) / max(c - b, 1)
    return torch.from_numpy(fb)


class Kayip:
    COZUNURLUKLER = ((512, 128), (1024, 256), (2048, 512))

    def __init__(self, cihaz):
        self.pencereler = {n: torch.hann_window(n, device=cihaz) for n, _ in self.COZUNURLUKLER}
        self.mel = mel_filtreleri(48000, 2048, 128).to(cihaz)

    def genlik(self, x, n_fft, hop):
        return torch.stft(x, n_fft, hop, window=self.pencereler[n_fft], return_complex=True).abs().clamp_min(1e-5)

    def __call__(self, ogrenci, ogretmen):
        stft = 0.0
        for n_fft, hop in self.COZUNURLUKLER:
            X, Y = self.genlik(ogrenci, n_fft, hop), self.genlik(ogretmen, n_fft, hop)
            stft = stft + torch.norm(Y - X) / torch.norm(Y) + (X.log() - Y.log()).abs().mean()
        X, Y = self.genlik(ogrenci, 2048, 512), self.genlik(ogretmen, 2048, 512)
        mel = (torch.log(self.mel @ X + 1e-5) - torch.log(self.mel @ Y + 1e-5)).abs().mean()
        return stft / len(self.COZUNURLUKLER) + mel


# ---------- ölçüm ----------

@torch.no_grad()
def coz(cozucu, z, cihaz):
    cozucu.eval()
    y = cozucu(z[None].to(cihaz)).reshape(-1).cpu().numpy()
    cozucu.train()
    return y


def olc(cozucu, ornek, cihaz):
    """Çözülen sesin aslına (16 kHz) ve öğretmene mesafesi; latent_uyumu.py ile aynı ölçü."""
    y = coz(cozucu, ornek["z"], cihaz)
    y16 = yeniden_ornekle(y, 48000, 16000)
    return mel_mesafesi(ornek["asil"], y16), mel_mesafesi(ornek["ogretmen16"], y16), y


def main():
    parser = argparse.ArgumentParser(description="EMA'nın küçük çözücüsünü bir sese uyarla")
    parser.add_argument("--egitim", nargs="+", required=True, type=Path, help="eğitim kayıtları")
    parser.add_argument("--dogrulama-sn", type=float, default=30.0, help="son eğitim kaydının sonundan ayrılacak saniye")
    parser.add_argument("--test", required=True, type=Path, help="hiç görülmeyecek test kaydı")
    parser.add_argument("--adim", type=int, default=2000)
    parser.add_argument("--yigin", type=int, default=8)
    parser.add_argument("--ogrenme-hizi", type=float, default=1e-4)
    parser.add_argument("--ad", required=True, help="çıktı: agirliklar/decoder_<ad>.pt")
    args = parser.parse_args()

    cihaz = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)
    cikti = CIKTI / "uyarlama"
    cikti.mkdir(parents=True, exist_ok=True)
    AGIRLIKLAR.mkdir(exist_ok=True)

    # --- veri: latentler ve öğretmen sesi, bir kez hesaplanır
    t0 = time.perf_counter()
    vae = voxcpm_vae().to(cihaz)
    egitim_z, egitim_y = [], []
    for i, yol in enumerate(args.egitim):
        ses = ses_dosyasi_16k(yol, en_uzun=None)
        if i == len(args.egitim) - 1 and args.dogrulama_sn > 0:
            ayrilan = int(args.dogrulama_sn * 16000) // 640 * 640
            ses, dogrulama_ses = ses[:-ayrilan], ses[-ayrilan:]
        z, y = latent_ve_ogretmen(vae, ses, cihaz)
        egitim_z.append(z)
        egitim_y.append(y)
        print(f"eğitim: {yol.name}, {len(ses) / 16000:.1f} sn -> {z.shape[-1]} kare")

    ornekler = {}
    for ad, ses in [("dogrulama", dogrulama_ses), ("test", ses_dosyasi_16k(args.test, en_uzun=None))]:
        z, y = latent_ve_ogretmen(vae, ses, cihaz)
        ornekler[ad] = {"z": z, "asil": ses, "ogretmen16": yeniden_ornekle(y.numpy(), 48000, 16000)}
        wav_yaz(cikti / f"{args.ad}_{ad}_asil.wav", ses, 16000)
        wav_yaz(cikti / f"{args.ad}_{ad}_ogretmen.wav", y.numpy(), 48000)
        print(f"{ad}: {len(ses) / 16000:.1f} sn")
    del vae
    print(f"veri hazır: {time.perf_counter() - t0:.0f} sn, cihaz: {cihaz}\n")

    # --- öğrenci: EMA'nın çözücüsü
    yol = hf_hub_download("canberkkkkkk/ema-lightning", "decoder.pt")
    cfg = torch.load(yol, map_location="cpu", weights_only=True)["cfg"]
    cozucu = load_decoder(yol, cihaz).train().requires_grad_(True)

    once = {ad: olc(cozucu, o, cihaz) for ad, o in ornekler.items()}
    for ad, (asil, ogr, y) in once.items():
        wav_yaz(cikti / f"{args.ad}_{ad}_once.wav", y, 48000)
        print(f"ÖNCE  {ad:10s} asla mesafe {asil:.3f}, öğretmene mesafe {ogr:.3f}")

    kayip_fn = Kayip(cihaz)
    opt = torch.optim.AdamW(cozucu.parameters(), lr=args.ogrenme_hizi, betas=(0.8, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.adim, eta_min=args.ogrenme_hizi * 0.1)
    agirliklar = [z.shape[-1] for z in egitim_z]
    en_iyi, en_iyi_adim = float("inf"), 0
    t0 = time.perf_counter()
    for adim in range(1, args.adim + 1):
        zs, ys = [], []
        for k in np.random.choice(len(egitim_z), args.yigin, p=np.array(agirliklar) / sum(agirliklar)):
            s = np.random.randint(0, egitim_z[k].shape[-1] - KIRPMA)
            zs.append(egitim_z[k][:, s:s + KIRPMA])
            ys.append(egitim_y[k][s * HOP:(s + KIRPMA) * HOP])
        z, hedef = torch.stack(zs).to(cihaz), torch.stack(ys).to(cihaz)
        cikan = cozucu(z)
        kesit = slice(KENAR * HOP, (KIRPMA - KENAR) * HOP)
        kayip = kayip_fn(cikan[:, kesit], hedef[:, kesit])
        opt.zero_grad(set_to_none=True)
        kayip.backward()
        torch.nn.utils.clip_grad_norm_(cozucu.parameters(), 1.0)
        opt.step()
        sched.step()

        if adim == 20:
            print(f"hız: adım başına {(time.perf_counter() - t0) / 20:.2f} sn")
        if adim % 250 == 0 or adim == args.adim:
            asil, ogr, _ = olc(cozucu, ornekler["dogrulama"], cihaz)
            print(f"adım {adim:5d}  kayıp {kayip.item():.3f}  doğrulama: asla {asil:.3f}, öğretmene {ogr:.3f}"
                  f"  ({time.perf_counter() - t0:.0f} sn)", flush=True)
            if asil < en_iyi:
                en_iyi, en_iyi_adim = asil, adim
                sd = {k: v.detach().cpu() for k, v in cozucu.state_dict().items()}
                torch.save({"G": sd, "cfg": cfg, "step": adim, "kaynak": "canberkkkkkk/ema-lightning decoder.pt",
                            "ogretmen": "openbmb/VoxCPM2 audiovae.pth"}, AGIRLIKLAR / f"decoder_{args.ad}.pt")

    # --- en iyi ağırlıklarla son ölçüm
    cozucu = load_decoder(AGIRLIKLAR / f"decoder_{args.ad}.pt", cihaz)
    print(f"\nEn iyi ağırlıklar: adım {en_iyi_adim} -> {AGIRLIKLAR / f'decoder_{args.ad}.pt'}")
    for ad, o in ornekler.items():
        asil, ogr, y = olc(cozucu, o, cihaz)
        wav_yaz(cikti / f"{args.ad}_{ad}_sonra.wav", y, 48000)
        print(f"SONRA {ad:10s} asla mesafe {asil:.3f} (önce {once[ad][0]:.3f}), "
              f"öğretmene {ogr:.3f} (önce {once[ad][1]:.3f})")

    # EMA'nın kendi sesinde ne kaybedildi?
    ema_16k = yeniden_ornekle(*wav_oku(CIKTI / "1_ema_orijinal.wav"), 16000)
    vae = voxcpm_vae().to(cihaz)
    z = latent_ve_ogretmen(vae, ema_16k, cihaz)[0]
    eski = load_decoder(hf_hub_download("canberkkkkkk/ema-lightning", "decoder.pt"), cihaz)
    for ad, c in [("eski çözücü", eski), ("yeni çözücü", cozucu)]:
        print(f"EMA'nın kendi sesi, {ad}: asla mesafe "
              f"{mel_mesafesi(ema_16k, yeniden_ornekle(coz(c, z, cihaz), 48000, 16000)):.3f}")
    print(f"\nDinlemek için: {cikti}")


if __name__ == "__main__":
    main()

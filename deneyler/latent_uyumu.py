"""Deney: EMA Lightning, VoxCPM2'nin AudioVAE latent uzayında mı çalışıyor?

Eğer öyleyse kendi kayıtların VoxCPM2'nin kodlayıcısıyla EMA'nın anladığı latentlere çevrilebilir
ve EMA'nın küçük çözücüsü (decoder.pt) yeniden kullanılabilir.

Üç test:
  A. EMA'nın ürettiği latentleri VoxCPM2'nin çözücüsüyle sese çevir; EMA'nın kendi sesine benziyor mu?
  B. EMA'nın sesini VoxCPM2'nin kodlayıcısıyla geri latente çevir; EMA'nın latentleriyle örtüşüyor mu?
  C. Başka bir sesi VoxCPM2 ile kodla, EMA'nın çözücüsüyle geri çöz; ses korunuyor mu?
     Varsayılan ses macOS'un Yelda'sı; --kayit ile kendi kaydın kullanılır.

Çalıştırmak için (Ema/deneyler klasöründe):
    .venv/bin/python latent_uyumu.py
    .venv/bin/python latent_uyumu.py --kayit ~/Desktop/benim_sesim.m4a

Kayıt; m4a, wav, aiff, mp3 gibi macOS'un açabildiği her biçimde olabilir. İlk 60 saniyesi kullanılır.
"""

import argparse
import importlib.util
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch
from ema_lightning import EMA
from huggingface_hub import hf_hub_download

CIKTI = Path(__file__).parent / "cikti"
METIN = "Sabahın erken saatlerinde liman henüz uyanmamıştı. Balıkçılar ağlarını sessizce topluyordu."
YELDA_METNI = "Merhaba, ben Yelda. Bu cümle, farklı bir sesin kodlanıp yeniden çözülmesini denemek için okunuyor."
EN_UZUN = 60  # saniye


# ---------- ses dosyası yardımcıları ----------

def wav_oku(yol):
    with wave.open(str(yol)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        return x, w.getframerate()


def wav_yaz(yol, x, sr):
    with wave.open(str(yol), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def yeniden_ornekle(x, sr, hedef):
    """macOS'un afconvert aracıyla örnekleme hızını değiştirir."""
    with tempfile.TemporaryDirectory() as d:
        giris, cikis = Path(d) / "a.wav", Path(d) / "b.wav"
        wav_yaz(giris, x, sr)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{hedef}", str(giris), str(cikis)], check=True)
        return wav_oku(cikis)[0]


def ses_dosyasi_16k(yol, en_uzun=EN_UZUN):
    """Herhangi bir ses dosyasını tek kanal, 16 kHz'e çevirir; sesi tepe 0.9 olacak şekilde ölçekler."""
    with tempfile.TemporaryDirectory() as d:
        cikis = Path(d) / "k.wav"
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(yol), str(cikis)], check=True)
        x = wav_oku(cikis)[0][:None if en_uzun is None else en_uzun * 16000]
    return x * (0.9 / max(np.abs(x).max(), 1e-6))


# ---------- ölçüler ----------

def mel_filtreleri(sr, n_fft, n_mels=80):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    noktalar = mel2hz(np.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2))
    kutular = np.floor((n_fft + 1) * noktalar / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        a, b, c = kutular[i], kutular[i + 1], kutular[i + 2]
        fb[i, a:b] = (np.arange(a, b) - a) / max(b - a, 1)
        fb[i, b:c] = (c - np.arange(b, c)) / max(c - b, 1)
    return torch.from_numpy(fb)


def log_mel(x, sr=16000):
    n_fft, hop = 1024, 256
    spec = torch.stft(torch.from_numpy(x), n_fft, hop, window=torch.hann_window(n_fft), return_complex=True).abs()
    return torch.log(mel_filtreleri(sr, n_fft) @ spec + 1e-5)


def mel_mesafesi(a, b, sr=16000):
    """İki sesin log-mel farkı (küçük = benzer). Kaymaya karşı ±40 ms içinde en iyi hizalamayı alır."""
    n = min(len(a), len(b))
    A, B = log_mel(a[:n], sr), log_mel(b[:n], sr)
    en_iyi = float("inf")
    for kayma in range(-3, 4):
        if kayma >= 0:
            fark = (A[:, kayma:] - B[:, :B.shape[1] - kayma]).abs().mean()
        else:
            fark = (A[:, :kayma] - B[:, -kayma:]).abs().mean()
        en_iyi = min(en_iyi, float(fark))
    return en_iyi


def latent_karsilastir(z1, z2):
    """[T, 64] iki latent dizisi: en iyi kaymada boyut başına korelasyon ve göreli hata."""
    sonuc = None
    for kayma in range(-3, 4):
        a = z1[max(0, kayma):len(z1) + min(0, kayma)]
        b = z2[max(0, -kayma):len(z2) - max(0, kayma)]
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        korelasyon = np.mean([np.corrcoef(a[:, i], b[:, i])[0, 1] for i in range(a.shape[1])])
        hata = float(((a - b) ** 2).mean() / (a ** 2).mean())
        if sonuc is None or korelasyon > sonuc[0]:
            sonuc = (korelasyon, hata, kayma)
    return sonuc


# ---------- modeller ----------

def voxcpm_vae():
    """VoxCPM2'nin AudioVAE'sini yalnızca kendi dosyasından yükler (paketin ağır bağımlılıkları gerekmez)."""
    kok = Path(importlib.util.find_spec("voxcpm").submodule_search_locations[0])
    spec = importlib.util.spec_from_file_location("audio_vae_v2", kok / "modules/audiovae/audio_vae_v2.py")
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    vae = modul.AudioVAE(modul.AudioVAEConfig())
    ck = torch.load(hf_hub_download("openbmb/VoxCPM2", "audiovae.pth"), map_location="cpu", weights_only=True)
    eksik, fazla = vae.load_state_dict(ck.get("state_dict", ck), strict=False)
    print(f"VoxCPM2 AudioVAE yüklendi (eksik anahtar: {len(eksik)}, fazla anahtar: {len(fazla)})")
    return vae.eval().requires_grad_(False)


@torch.no_grad()
def main(kayit=None, kayit_adi="kayit"):
    CIKTI.mkdir(exist_ok=True)
    tts = EMA(device="cpu")
    vae = voxcpm_vae()
    ema_cozucu = tts._engine.decoder

    # EMA'nın ürettiği latentleri yakala
    yakalanan = []
    asil_think = tts._engine.think

    def think(parcalar):
        asil_think(parcalar)
        yakalanan.extend(p.latents.cpu().numpy() for p in parcalar)

    tts._engine.think = think
    ses = tts.say(METIN, seed=0)
    z_ema = np.concatenate(yakalanan)                      # [T, 64], 25 Hz
    ema_48k = ses.audio
    ema_16k = yeniden_ornekle(ema_48k, 48000, 16000)
    wav_yaz(CIKTI / "1_ema_orijinal.wav", ema_48k, 48000)
    print(f"\nEMA latentleri: {z_ema.shape}, ortalama {z_ema.mean():.3f}, std {z_ema.std():.3f}")

    # A. EMA latentleri -> VoxCPM2 çözücüsü
    z = torch.from_numpy(z_ema).T[None]
    vox_48k = vae.decode(z).squeeze().numpy()
    wav_yaz(CIKTI / "2_ema_latent_voxcpm_cozucu.wav", vox_48k, 48000)
    vox_16k = yeniden_ornekle(vox_48k, 48000, 16000)
    karisik = z_ema[np.random.default_rng(0).permutation(len(z_ema))]   # aynı değerler, sırası bozuk: "ilgisiz ses" çıtası
    karisik_16k = yeniden_ornekle(vae.decode(torch.from_numpy(karisik).T[None]).squeeze().numpy(), 48000, 16000)

    # B. EMA sesi -> VoxCPM2 kodlayıcısı -> latent
    z_vox = vae.encode(torch.from_numpy(ema_16k)[None, None], 16000)[0].T.numpy()
    print(f"VoxCPM2 kodlayıcısının latentleri: {z_vox.shape}, ortalama {z_vox.mean():.3f}, std {z_vox.std():.3f}")
    korelasyon, hata, kayma = latent_karsilastir(z_ema, z_vox)
    rastgele_kor = latent_karsilastir(z_ema, z_vox[np.random.default_rng(1).permutation(len(z_vox))])[0]
    geri_48k = ema_cozucu(torch.from_numpy(z_vox).T[None].contiguous()).squeeze().numpy()
    wav_yaz(CIKTI / "3_ema_sesi_voxcpm_kodla_ema_coz.wav", geri_48k, 48000)

    print("\n=== A. EMA latentleri VoxCPM2 çözücüsüyle ===")
    print(f"EMA'nın kendi sesine log-mel mesafesi: {mel_mesafesi(ema_16k, vox_16k):.3f}")
    print(f"Sırası bozulmuş latentlerle (ilgisiz ses çıtası): {mel_mesafesi(ema_16k, karisik_16k):.3f}")

    print("\n=== B. EMA sesi VoxCPM2 kodlayıcısıyla geri latente ===")
    print(f"Boyut başına korelasyon: {korelasyon:.3f} (kayma {kayma} kare), göreli hata: {hata:.3f}")
    print(f"Sırası bozulmuş eşleşmede korelasyon (çıta): {rastgele_kor:.3f}")
    print(f"Geri çözülen sesin EMA sesine mesafesi: {mel_mesafesi(ema_16k, yeniden_ornekle(geri_48k, 48000, 16000)):.3f}")

    # C. Başka bir ses: verilen kayıt ya da macOS Yelda
    if kayit:
        ad, baslik = kayit_adi, f"Kaydın ({Path(kayit).name})"
        diger_16k = ses_dosyasi_16k(kayit)
    else:
        ad, baslik = "yelda", "Yelda'nın sesi"
        with tempfile.TemporaryDirectory() as d:
            aiff = Path(d) / "y.aiff"
            subprocess.run(["say", "-v", "Yelda", "-o", str(aiff), YELDA_METNI], check=True)
            diger_16k = ses_dosyasi_16k(aiff)
    wav_yaz(CIKTI / f"4_{ad}_orijinal.wav", diger_16k, 16000)
    z_d = vae.encode(torch.from_numpy(diger_16k)[None, None], 16000)
    diger_vox = vae.decode(z_d).squeeze().numpy()
    diger_ema = ema_cozucu(z_d.contiguous()).squeeze().numpy()
    wav_yaz(CIKTI / f"5_{ad}_voxcpm_kodla_voxcpm_coz.wav", diger_vox, 48000)
    wav_yaz(CIKTI / f"6_{ad}_voxcpm_kodla_ema_coz.wav", diger_ema, 48000)
    z_d_karisik = z_d[..., torch.randperm(z_d.shape[-1], generator=torch.Generator().manual_seed(0))]
    print(f"\n=== C. {baslik}, {len(diger_16k) / 16000:.1f} sn: VoxCPM2 kodlayıcısı -> çözücüler ===")
    print(f"VoxCPM2'nin kendi çözücüsü (tavan): {mel_mesafesi(diger_16k, yeniden_ornekle(diger_vox, 48000, 16000)):.3f}")
    print(f"EMA'nın küçük çözücüsü:             {mel_mesafesi(diger_16k, yeniden_ornekle(diger_ema, 48000, 16000)):.3f}")
    print(f"Sırası bozuk latent (çıta):         "
          f"{mel_mesafesi(diger_16k, yeniden_ornekle(ema_cozucu(z_d_karisik.contiguous()).squeeze().numpy(), 48000, 16000)):.3f}")
    print(f"\nDinlemek için: {CIKTI}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMA ile VoxCPM2 latent uzayı uyumu deneyi")
    parser.add_argument("--kayit", help="test C için kendi ses kaydın (m4a, wav, aiff, mp3…)")
    parser.add_argument("--ad", default="kayit", help="çıktı dosyalarının adı, birden çok kaydı karşılaştırırken (örn. kayit_chatterbox)")
    args = parser.parse_args()
    if args.kayit and not Path(args.kayit).expanduser().is_file():  # modeller yüklenmeden önce, hemen söyle
        parser.error(f"kayıt dosyası bulunamadı: {args.kayit}\nÖnce kaydı al ve bu adla kaydet ya da komuttaki yolu düzelt.")
    main(args.kayit, args.ad)

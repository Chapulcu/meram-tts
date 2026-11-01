"""İnce ayarlı akustik model için örnekleyici: istenen adım sayısı ve yönlendirme (CFG) ile seslendirme.

EMA paketinin örnekleyicisi damıtılmış öğrenciye özel: dört anda latenti tahmin edip yeniden gürültüler,
yönlendirme yok. Düz akış eşlemeyle ince ayarlanan model çok adımlı Euler'le daha iyi sonuç veriyor;
koşul bırakmayla eğitildiyse yönlendirme de kullanılabilir:
    v = v_metinsiz + w · (v_metinli - v_metinsiz)      (w = 1: yönlendirme yok)

Süreler ve zaman çizelgesi paketin kendi yolundan (Engine.plan) gelir; yalnızca latent üretimi değişir.

    from ornekle import Seslendirici
    s = Seslendirici("akustik/agirliklar/ema_kadin.pt", "agirliklar/decoder_kadin.pt")
    ses = s("Merhaba, nasılsınız?", adim=16, yonlendirme=2.0)    # 48 kHz numpy dizisi
"""

import numpy as np
import torch
from ema_lightning import EMA
from ema_lightning.audio import Resampler
from ema_lightning.decoder import load_decoder
from ema_lightning.frontend import Frontend
from ema_lightning.model import load_acoustic, rope

# Streamlit uygulamasının da kullandığı en iyi ayar. İlk ince ayarlı sesle (8,5 saat veri) 50 doğrulama cümlesinde
# (cfg_dene.py) hedefle aynı: harf hatası %0,6, perde oynaklığı 3,51 yt, düzenlilik 0,684. 16 adım çok yakın,
# iki kat hızlı; 32 adımda bile M4 Max'te 8,7 sn ses ~115 ms (76x gerçek zaman).
EN_IYI = {"adim": 32, "yonlendirme": 2.0, "sicaklik": 0.7, "sway": -1.0}


def anlar(adim, sway):
    """Euler adımlarının zaman noktaları. sway < 0 adımları başa (kaba yapının kurulduğu yere) yığar
    (F5-TTS'teki "sway sampling"); sway = 0 eşit aralık."""
    u = torch.linspace(0, 1, adim + 1)
    return (u + sway * (torch.cos(torch.pi / 2 * u) - 1 + u)).tolist()


@torch.no_grad()
def latent(model, p, cihaz, adim=16, yonlendirme=1.0, yontem="euler", tohum=0, sicaklik=1.0, sway=0.0):
    """Engine.plan'dan geçmiş bir parçanın latentleri [T, 64]. sicaklik başlangıç gürültüsünü ölçekler."""
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
    T = fw.shape[1]
    cos, sin = rope(torch.arange(T, device=cihaz), model.dh)
    fmask = torch.ones(1, T, dtype=torch.bool, device=cihaz)
    g = torch.Generator().manual_seed(tohum)

    def hiz(x, t):
        tt = torch.full((1,), t, device=cihaz)
        if yonlendirme == 1.0:
            return model.backbone(x, cond, tt, fmask, cos, sin)
        v = model.backbone(torch.cat([x, x]), torch.cat([cond, torch.zeros_like(cond)]), tt.repeat(2),
                           fmask.repeat(2, 1), cos, sin)
        metinli, metinsiz = v.chunk(2)
        return metinsiz + yonlendirme * (metinli - metinsiz)

    x = sicaklik * torch.randn(1, T, model.latent_dim, generator=g).to(cihaz)
    if yontem == "euler":
        t = anlar(adim, sway)
        for a, b in zip(t[:-1], t[1:]):
            x = x + hiz(x, a) * (b - a)
        return x[0]
    # "yeniden": EMA'nın kendi örnekleyicisi (latenti tahmin et, sonraki anın gürültüsüyle karıştır)
    noktalar = [k / adim for k in range(adim)]
    for k, t in enumerate(noktalar):
        x1 = x + (1 - t) * hiz(x, t)
        if k + 1 < len(noktalar):
            x = (1 - noktalar[k + 1]) * torch.randn(x.shape, generator=g).to(cihaz) + noktalar[k + 1] * x1
    return x1[0]


class Seslendirici:
    def __init__(self, akustik, cozucu, cihaz=None):
        self.cihaz = torch.device(cihaz or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model = load_acoustic(akustik, self.cihaz)
        self.cozucu = load_decoder(cozucu, self.cihaz)
        self.tts = EMA._from_parts(self.model, self.cozucu, Frontend(self.model.vocab), self.cihaz)

    @torch.no_grad()
    def parcalar(self, metin, adim=16, yonlendirme=1.0, yontem="euler", hiz=1.0, tohum=0, sicaklik=1.0, sway=0.0,
                 sample_rate=48000):
        """Metni parça parça seslendirir: her parça (en çok ~10 sn'lik cümle grubu) hazır olunca verilir.
        Akış için: ilk ses, metnin tamamı bitmeden çalmaya başlayabilir."""
        parcalar = self.tts._pieces(metin, hiz, tohum)
        if not parcalar:
            return
        self.tts._engine.plan(parcalar, hiz)
        r = Resampler(sample_rate) if sample_rate != 48000 else None  # paketin kendi süzgeci, say() ile aynı yol
        for i, p in enumerate(parcalar):
            z = latent(self.model, p, self.cihaz, adim, yonlendirme, yontem, tohum + i, sicaklik, sway)
            ses = self.cozucu(z.T[None].contiguous()).reshape(-1).float().cpu()
            if p.pause:
                ses = torch.cat([ses, torch.zeros(int(p.pause * 48000))])
            yield (r.push(ses) if r else ses).numpy()
        if r and (kuyruk := r.flush()) is not None and kuyruk.numel():
            yield kuyruk.numpy()

    def __call__(self, metin, **ayar):
        sesler = list(self.parcalar(metin, **ayar))
        return np.concatenate(sesler) if sesler else np.zeros(0, dtype=np.float32)


def ses_ozellikleri(x, sr=48000):
    """(ortanca perde Hz, perde oynaklığı yarım ton, düzenlilik 0-1). Basit öz ilinti, 40 ms kareler.

    Perde oynaklığı tonlamanın canlılığını ölçer: düşükse ses tekdüze ("robotik") duyulur.
    Düzenlilik sesli karelerde periyodikliğin gücü: harmonikler bulanıklaştıkça düşer.
    """
    from scipy.signal import resample_poly  # yalnızca ölçüm için; uygulama scipy olmadan çalışsın

    x = resample_poly(x, 16000, sr)
    pencere, f0, duzen = 640, [], []
    for s in range(0, len(x) - pencere, pencere // 2):
        p = x[s:s + pencere] - x[s:s + pencere].mean()
        if (p ** 2).mean() < 1e-4:
            continue
        r = np.correlate(p, p, "full")[pencere - 1:]
        a, b = 16000 // 400, 16000 // 60
        k = a + int(np.argmax(r[a:b]))
        if r[k] > 0.4 * r[0]:
            f0.append(16000 / k)
            duzen.append(r[k] / r[0])
    if len(f0) < 3:
        return float("nan"), float("nan"), float("nan")
    yarim_ton = 12 * np.log2(np.array(f0) / np.median(f0))
    return float(np.median(f0)), float(np.std(yarim_ton)), float(np.mean(duzen))

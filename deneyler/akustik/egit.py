"""EMA'nın akustik modelini (ema.pt, 5,6M) senin sesine ince ayarlar.

Model sıfırdan değil EMA'nın ağırlıklarından başlar: Türkçe okumayı zaten biliyor, ses rengini öğrenmesi yeter.
- Metin kodlayıcısı donuk kalır (harflerin anlamı değişmiyor).
- Süre tahmincisi: kelime başına toplam kare sayısı, hizalamadan gelen gerçek değere yaklaştırılır
  (log uzayında karesel hata). Harfin kelime içindeki yeri modelin kendi oranlarından gelir; EMA da öyle.
- Hizalayıcı ve DiT: akış eşleme (flow matching). x_t = (1-t)·gürültü + t·latent, hedef hız = latent - gürültü.
  Bu, modelin kendi örnekleyicisiyle uyumlu: x1 = x_t + (1-t)·v. t'nin yarısı modelin kullandığı dört
  andan (0, ¼, ½, ¾), yarısı düzgün dağılımdan seçilir.
- Örneklerin %15'inde koşul sıfırlanır (koşul bırakma); böylece seslendirirken yönlendirme (CFG) kullanılabilir.
  Yönlendirmeli ve çok adımlı seslendirme için ornekle.py.

Bilinen sınır: yayımlanan model DMD2 ile 4 adıma damıtılmış bir öğrenci; burada düz akış eşlemeyle
eğitiliyor. Ses biraz yumuşayabilir; gerekirse sonraki adım yeniden damıtma.

Çalıştırmak için (Ema/deneyler klasöründe):
    .venv/bin/python akustik/egit.py --adim 3000
Çıktı: akustik/agirliklar/ema_<ad>.pt (EMA'nın kendi biçiminde) ve akustik/cikti/<ad>/ altında örnek sesler
"""

import argparse
import copy
import math
import random
import sys
import time
from itertools import pairwise
from pathlib import Path

import mlx_whisper
import numpy as np
import torch
from ema_lightning import EMA
from ema_lightning.decoder import load_decoder
from ema_lightning.frontend import Frontend
from ema_lightning.model import load_acoustic, rope
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent.parent))
from latent_uyumu import wav_yaz  # noqa: E402

AKUSTIK = Path(__file__).parent
KOK = AKUSTIK.parent
COZUCU = KOK / "agirliklar/decoder_kadin.pt"   # örnek varsayılan; --cozucu ile verilir
WHISPER = "mlx-community/whisper-large-v3-turbo"


# ---------- veri ----------

def kelime_yapisi(metin):
    """Her harfin kelimesi ve o kelimenin ilk harfi; EMA'nın Engine.piece'iyle aynı kural."""
    basla = [i for i, ch in enumerate(metin) if ch != " " and (i == 0 or metin[i - 1] == " ")] or [0]
    sinir = [0] + basla[1:] + [len(metin)]
    cw, wstart = [], []
    for w, (a, b) in enumerate(pairwise(sinir)):
        cw += [w] * (b - a)
        wstart += [a] * (b - a)
    return cw, wstart


def ornek_hazirla(kayit, stoi):
    metin, n = kayit["metin"], kayit["kelime_kare"].long()
    cw, wstart = kelime_yapisi(metin)
    if cw[-1] + 1 != len(n):
        return None
    T = int(n.sum())
    fw = torch.repeat_interleave(torch.arange(len(n)), n)
    fp = (torch.arange(T) - (n.cumsum(0) - n)[fw]).float() / n[fw].float()
    return {"id": kayit["id"], "metin": metin, "ids": torch.tensor([stoi.get(ch, 1) for ch in metin]),
            "cw": torch.tensor(cw), "wstart": torch.tensor(wstart), "fw": fw, "fp": fp, "n": n,
            "z": kayit["z"][:T].float()}


def doldur(satirlar, deger):
    out = torch.full((len(satirlar), max(len(s) for s in satirlar)) + satirlar[0].shape[1:], deger, dtype=satirlar[0].dtype)
    for i, s in enumerate(satirlar):
        out[i, :len(s)] = s
    return out


def toplu(ornekler, cihaz):
    ids = doldur([o["ids"] for o in ornekler], 0)
    fw = doldur([o["fw"] for o in ornekler], -1)
    n = doldur([o["n"] for o in ornekler], 0)
    return {"ids": ids.to(cihaz), "mask": (ids != 0).to(cihaz), "cw": doldur([o["cw"] for o in ornekler], -1).to(cihaz),
            "wstart": doldur([o["wstart"] for o in ornekler], 0).to(cihaz), "fw": fw.to(cihaz),
            "fp": doldur([o["fp"] for o in ornekler], 0.0).to(cihaz), "fmask": (fw >= 0).to(cihaz),
            "n": n.to(cihaz), "wmask": (n > 0).to(cihaz), "z": doldur([o["z"] for o in ornekler], 0.0).to(cihaz)}


# ---------- model ----------

def kosul(model, b):
    """Metin -> süreler, kare başına koşul. sound_stage'in örnekleme döngüsünden önceki kısmıyla aynı."""
    with torch.no_grad():
        h = model.text(b["ids"], b["mask"])
    dur = model.chardur(h, b["mask"])
    c = dur.detach().clamp(min=1e-4) * b["mask"]
    done = c.cumsum(-1)
    before = done - c
    word = b["cw"].clamp(min=0)
    total = torch.zeros_like(c).scatter_add_(1, word, c).gather(1, word)
    cp = ((done - before.gather(1, b["wstart"]) - 0.5 * c) / total.clamp(min=1e-8)).clamp(0.0, 1.0) * b["mask"]
    cond = model.aligner(h, b["cw"], cp, b["fw"], b["fp"], b["mask"], b["cw"].shape[1])
    return dur, cond


def kayiplar(model, b, t, gurultu, birakma=0.0):
    dur, cond = kosul(model, b)
    if birakma > 0:
        # Yönlendirme (CFG) için: bazı örneklerde koşul tamamen sıfırlanır, model metinsiz tahmini de öğrenir.
        # Seslendirirken v = v_metinsiz + w·(v_metinli - v_metinsiz) ile metne daha sıkı bağlanır.
        tut = (torch.rand(cond.shape[0], 1, 1, device=cond.device) >= birakma).float()
        cond = cond * tut
    # süre: kelime başına tahmin edilen toplam kare, gerçeğe log uzayında
    tahmin = torch.zeros(b["n"].shape, device=dur.device).scatter_add_(1, b["cw"].clamp(min=0), dur * b["mask"])
    w = b["wmask"].float()
    kayip_sure = (((torch.log1p(tahmin) - torch.log1p(b["n"].float())) ** 2) * w).sum() / w.sum()
    # akış eşleme
    x1, x0 = b["z"], gurultu
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    cos, sin = rope(torch.arange(x1.shape[1], device=x1.device), model.dh)
    v = model.backbone(xt, cond, t, b["fmask"], cos, sin)
    f = b["fmask"].float()
    kayip_akis = ((((v - (x1 - x0)) ** 2).mean(-1)) * f).sum() / f.sum()
    return kayip_akis, kayip_sure


def t_sec(B, times, cihaz):
    duz = torch.rand(B, device=cihaz)
    dort = torch.tensor(times, device=cihaz)[torch.randint(len(times), (B,), device=cihaz)]
    return torch.where(torch.rand(B, device=cihaz) < 0.5, dort, duz)


@torch.no_grad()
def dogrulama_kaybi(model, dogrulama, cihaz, yigin=16):
    """Sabit gürültü ve sabit t ile, adımlar arasında karşılaştırılabilir doğrulama kaybı."""
    model.eval()
    g = torch.Generator().manual_seed(0)
    akis = sure = 0.0
    parcalar = [dogrulama[i:i + yigin] for i in range(0, len(dogrulama), yigin)]
    for p in parcalar:
        b = toplu(p, cihaz)
        sonuclar = []
        for t in model.times:
            gurultu = torch.randn(b["z"].shape, generator=g).to(cihaz)
            sonuclar.append(kayiplar(model, b, torch.full((len(p),), t, device=cihaz), gurultu))
        akis += sum(a for a, _ in sonuclar).item() / len(sonuclar)
        sure += sonuclar[0][1].item()
    model.train()
    return akis / len(parcalar), sure / len(parcalar)


# ---------- ölçüm ----------

def harfler(s):
    return "".join(c for c in s if c.isalpha())


def levenshtein(a, b):
    onceki = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        simdiki = [i]
        for j, y in enumerate(b, 1):
            simdiki.append(min(onceki[j] + 1, simdiki[j - 1] + 1, onceki[j - 1] + (x != y)))
        onceki = simdiki
    return onceki[-1]


def perde(x, sr=48000):
    """Seslendirmenin ortanca temel frekansı (Hz); basit öz ilinti, yalnızca sesli kareler."""
    x = resample_poly(x, 1, 3)
    sr //= 3
    pencere, f0 = int(sr * 0.04), []
    for s in range(0, len(x) - pencere, pencere // 2):
        p = x[s:s + pencere] - x[s:s + pencere].mean()
        if (p ** 2).mean() < 1e-4:
            continue
        r = np.correlate(p, p, "full")[pencere - 1:]
        a, b = sr // 400, sr // 60          # 60–400 Hz arası
        k = a + int(np.argmax(r[a:b]))
        if r[k] > 0.4 * r[0]:
            f0.append(sr / k)
    return float(np.median(f0)) if f0 else float("nan")


def seslendir_ve_olc(model, cozucu, frontend, ornekler, klasor, cihaz):
    """Modelin kendi çıkarım yoluyla (süreler dahil tahmin) seslendirir; Whisper harf hatası ve perde."""
    klasor.mkdir(parents=True, exist_ok=True)
    tts = EMA._from_parts(model.eval(), cozucu, frontend, cihaz)
    hatalar, perdeler = [], []
    for o in ornekler:
        ses = tts.say(o["metin"], seed=0)
        wav_yaz(klasor / f"{o['id']}.wav", ses.audio, 48000)
        duyulan = mlx_whisper.transcribe(resample_poly(ses.audio, 1, 3).astype(np.float32), path_or_hf_repo=WHISPER, language="tr",
                                         temperature=0.0, condition_on_previous_text=False)["text"]
        a, b = harfler(o["metin"]), harfler(frontend(duyulan))
        hatalar.append(levenshtein(a, b) / max(len(a), 1))
        perdeler.append(perde(ses.audio))
    model.train()
    return float(np.mean(hatalar)), float(np.nanmedian(perdeler))


# ---------- ana akış ----------

def main():
    parser = argparse.ArgumentParser(description="EMA'nın akustik modelini bir sese ince ayarla")
    parser.add_argument("--veri", type=Path, default=AKUSTIK / "veri")
    parser.add_argument("--adim", type=int, default=3000)
    parser.add_argument("--yigin", type=int, default=16)
    parser.add_argument("--ogrenme-hizi", type=float, default=1e-4)
    parser.add_argument("--dogrulama", type=int, default=20, help="ayrılacak cümle sayısı")
    parser.add_argument("--olcum-araligi", type=int, default=500)
    parser.add_argument("--kosul-birakma", type=float, default=0.15, help="koşulun sıfırlandığı örnek oranı (CFG için)")
    parser.add_argument("--ad", required=True, help="çıktı: akustik/agirliklar/ema_<ad>.pt")
    parser.add_argument("--cozucu", type=Path, default=COZUCU, help="bu sese uyarlanmış çözücü (ölçüm sesleri için)")
    args = parser.parse_args()

    cihaz = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)
    random.seed(0)
    yol = hf_hub_download("canberkkkkkk/ema-lightning", "ema.pt")
    ck = torch.load(yol, map_location="cpu", weights_only=True)
    model = load_acoustic(yol, cihaz).train()
    model.requires_grad_(True)
    model.text.requires_grad_(False)
    frontend = Frontend(ck["vocab"])
    cozucu = load_decoder(args.cozucu, cihaz)

    kayitlar = [k for p in sorted(args.veri.glob("parca_*.pt")) for k in torch.load(p, weights_only=False)]
    ornekler = [o for o in (ornek_hazirla(k, model.stoi) for k in kayitlar) if o is not None]
    random.Random(1).shuffle(ornekler)
    dogrulama, egitim = ornekler[:args.dogrulama], ornekler[args.dogrulama:]
    print(f"eğitim: {len(egitim)} cümle, {sum(len(o['z']) for o in egitim) / 25 / 3600:.2f} saat; "
          f"doğrulama: {len(dogrulama)} cümle; cihaz: {cihaz}")

    cikti = AKUSTIK / "cikti" / args.ad
    hata, f0 = seslendir_ve_olc(copy.deepcopy(model), cozucu, frontend, dogrulama, cikti / "once", cihaz)
    dk, ds = dogrulama_kaybi(model, dogrulama, cihaz)
    print(f"ÖNCE   harf hatası %{100 * hata:.1f}, perde {f0:.0f} Hz | doğrulama akış {dk:.3f}, süre {ds:.3f}", flush=True)

    egitilen = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(egitilen, lr=args.ogrenme_hizi, betas=(0.9, 0.99), weight_decay=0.01)
    isinma = min(200, args.adim // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / isinma) *
                                              (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, args.adim) / args.adim))))
    ortalama = copy.deepcopy(model).requires_grad_(False)   # ağırlıkların üstel ortalaması: kaydedilen budur

    AGIRLIK = AKUSTIK / "agirliklar"
    AGIRLIK.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    for adim in range(1, args.adim + 1):
        b = toplu(random.sample(egitim, min(args.yigin, len(egitim))), cihaz)
        t = t_sec(len(b["ids"]), model.times, cihaz)
        kayip_akis, kayip_sure = kayiplar(model, b, t, torch.randn_like(b["z"]), args.kosul_birakma)
        kayip = kayip_akis + kayip_sure
        opt.zero_grad(set_to_none=True)
        kayip.backward()
        torch.nn.utils.clip_grad_norm_(egitilen, 1.0)
        opt.step()
        sched.step()
        bozunma = min(0.999, (1 + adim) / (10 + adim))
        with torch.no_grad():
            for a, m in zip(ortalama.parameters(), model.parameters()):
                a.lerp_(m, 1 - bozunma)

        if adim == 20:
            print(f"hız: adım başına {(time.perf_counter() - t0) / 20:.2f} sn", flush=True)
        if adim % 100 == 0:
            print(f"adım {adim:5d}  akış {kayip_akis.item():.3f}  süre {kayip_sure.item():.3f}  "
                  f"({time.perf_counter() - t0:.0f} sn)", flush=True)
        if adim % args.olcum_araligi == 0 or adim == args.adim:
            dk, ds = dogrulama_kaybi(ortalama, dogrulama, cihaz)
            print(f"  doğrulama (ortalama ağırlıklar): akış {dk:.3f}, süre {ds:.3f}", flush=True)
            torch.save({"ema": {k: v.detach().cpu() for k, v in ortalama.state_dict().items()},
                        "cfg": ck["cfg"], "vocab": ck["vocab"], "step": adim, "kosul_birakma": args.kosul_birakma,
                        "kaynak": "canberkkkkkk/ema-lightning ema.pt", "veri": str(args.veri)},
                       AGIRLIK / f"ema_{args.ad}.pt")

    hata, f0 = seslendir_ve_olc(ortalama, cozucu, frontend, dogrulama, cikti / "sonra", cihaz)
    print(f"SONRA  harf hatası %{100 * hata:.1f}, perde {f0:.0f} Hz")
    print(f"ağırlıklar: {AGIRLIK / f'ema_{args.ad}.pt'}; sesler: {cikti}")


if __name__ == "__main__":
    main()

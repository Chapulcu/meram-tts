"""VoxCPM2'nin ses tasarımıyla (Voice Design) gerçek kimseye ait olmayan aday sesler üretir.

Metnin başına parantez içinde sesin tarifi yazılır; tarif İngilizce (VoxCPM2 en iyi onu anlıyor), metin Türkçe.
Tasarım her denemede biraz farklı çıktığı için her tariften birkaç deneme, her deneme kendi tohumuyla:
beğenilen aday aynı tohumla yeniden üretilebilir ve sonraki veri üretiminde referans kayıt olur.

    .venv/bin/python aday_uret.py                 # bütün sesler
    .venv/bin/python aday_uret.py --ses kadin     # yalnızca biri
Çıktı: adaylar/<ses>_t<tarif>_d<deneme>.wav ve adaylar/adaylar.tsv
"""

import argparse
import time
from pathlib import Path

import soundfile as sf
import torch
from voxcpm import VoxCPM

KLASOR = Path(__file__).parent / "adaylar"
METIN = ("Merhaba! Bugün size kısa bir hikâye okuyacağım. Sabahın erken saatlerinde liman henüz uyanmamıştı; "
         "martılar teknelerin etrafında dönüyor, balıkçılar ağlarını sessizce topluyordu.")
TARIFLER = {
    "kadin": [
        "A woman in her thirties, warm, soft and clear voice, calm storyteller",
        "A young adult woman, bright, friendly and articulate voice",
        "A middle-aged woman, gentle, soothing, slightly low-pitched voice, audiobook narrator",
        "A woman in her twenties, smooth, elegant and confident voice, news presenter",
    ],
    "erkek": [
        "A man in his thirties, deep and warm baritone, calm and trustworthy voice",
        "A middle-aged man, rich and resonant voice, audiobook narrator",
        "A young adult man, friendly, clear and energetic voice",
        "A man in his forties, soft-spoken, gentle and reassuring voice",
    ],
    "kiz_cocuk": [
        "An 8-year-old girl, cheerful, bright and sweet voice",
        "A 10-year-old girl, clear, curious and lively voice",
        "A young girl around 7 years old, soft, innocent and happy voice",
        "A 9-year-old girl, playful and expressive voice",
    ],
    "erkek_cocuk": [
        "An 8-year-old boy, curious, energetic and cheerful voice",
        "A 10-year-old boy, clear, friendly and lively voice",
        "A young boy around 7 years old, playful and excited voice",
        "A 9-year-old boy, bright and confident voice",
    ],
    # 2. tur: ilk turdaki erkek çocukların çoğu ergen gibi kalın çıktı; perdeyi ve yaşı vurgulayan tarifler
    "erkek_cocuk2": [
        "A 6-year-old little boy, high-pitched, bright childish voice, voice has not broken yet",
        "A 7-year-old boy, light and high voice, cheerful and curious, clearly a young child",
        "A small boy around 6, thin high-pitched voice, playful and sweet",
        "An 8-year-old boy with a high, clear child's voice, energetic, kindergarten-like",
        "A young child, boy, about 7 years old, soft high voice, innocent and excited",
        "A 6-year-old boy, giggly, lively, very young high-pitched voice, cartoon-like but natural",
    ],
}


def sapkasiz(metin):
    """VoxCPM2'nin ayrıştırıcısı şapkalı harfleri nadir, ayrı parçalar olarak görüyor ve kelimeyi yutuyor
    ("hikâye" -> "hika"). Okunuş değişmediği için düz harfe çeviriyoruz."""
    return metin.translate(str.maketrans("âîûÂÎÛ", "aiuAIU"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ses", choices=list(TARIFLER), nargs="*", default=list(TARIFLER))
    parser.add_argument("--deneme", type=int, default=2)
    args = parser.parse_args()
    KLASOR.mkdir(exist_ok=True)

    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=False, device="mps")
    sr = model.tts_model.sample_rate
    kayit = KLASOR / "adaylar.tsv"
    yeni = not kayit.exists()
    with open(kayit, "a", encoding="utf-8") as log:
        if yeni:
            log.write("dosya\tses\ttarif_no\tdeneme\ttohum\ttarif\n")
        for ses in args.ses:
            for t, tarif in enumerate(TARIFLER[ses], 1):
                for d in range(1, args.deneme + 1):
                    tohum = 1000 * t + d
                    torch.manual_seed(tohum)
                    basla = time.perf_counter()
                    wav = model.generate(text=f"({tarif}){sapkasiz(METIN)}", cfg_value=2.0, inference_timesteps=10)
                    ad = f"{ses}_t{t}_d{d}.wav"
                    sf.write(KLASOR / ad, wav, sr)
                    log.write(f"{ad}\t{ses}\t{t}\t{d}\t{tohum}\t{tarif}\n")
                    log.flush()
                    print(f"{ad}: {len(wav) / sr:.1f} sn, {time.perf_counter() - basla:.1f} sn'de", flush=True)


if __name__ == "__main__":
    main()

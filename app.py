"""EMA Lightning ile Türkçe metni ve belgeleri sese çeviren Streamlit uygulaması.

Çalıştırmak için:
    uv run streamlit run app.py
"""

import sys

if sys.version_info < (3, 11):
    # Düz "streamlit run" macOS'un eski sistem Python'unu (3.9) kullanabiliyor; orada ses modeli kurulu değil
    import streamlit as st

    st.error("Uygulama yanlış Python ortamıyla başlatıldı (Python "
             f"{sys.version_info.major}.{sys.version_info.minor}). Bu pencereyi kapatıp Terminal'de şunu çalıştır:\n\n"
             "`uv run streamlit run app.py` (proje klasöründe)")
    st.stop()

import hashlib  # noqa: E402
import io
import os
import random
import secrets
import shutil
import subprocess
import tempfile
import time
import wave
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st

from belge import BICIMLER, BelgeHatasi, oku
from ses_dosyasi import FFMPEG, WAV_SINIR, SesYazici, pcm

# Sesler (ve ince ayarlı seslerin dosyaları) API ile ortak kayıtta: ses_kaydi.py. Dosyaları olmayan ses listede
# görünmez; sesler/ses_hatti.sh bir sesi bitirince uygulama yeniden başlatıldığında kendiliğinden belirir.
import ses_kaydi  # noqa: E402
from ses_kaydi import EN_IYI  # noqa: E402

KOK = Path(__file__).parent
# Bulut: Hugging Face Space (SPACE_ID'yi Hugging Face kendisi tanımlar) ya da EMA_BULUT=1 ile yerel Docker denemesi.
# Bulutta kullanıcı sunucunun klasörlerine erişemez: çıktılar geçici klasöre yazılır ve hep indirme düğmesiyle verilir.
BULUT = bool(os.getenv("SPACE_ID")) or os.getenv("EMA_BULUT") == "1"
# Yerelde EMA_CIKTI_KLASORU ile değiştirilebilir (macOS servisi ~/Music/EMA Seslendirme kullanır; bkz. servis/servisler.sh)
BELGE_CIKTI = (Path(tempfile.gettempdir()) / "ema_belgeler" if BULUT else
               Path(os.getenv("EMA_CIKTI_KLASORU") or KOK / "cikti/belgeler"))
KARAKTER_SN = 14.4          # ölçülen konuşma hızı (1× hızda saniyede karakter); süre tahmini için
PARAGRAF_ARASI, BOLUM_ARASI = 0.35, 1.2   # saniye sessizlik
ONIZLEME_SINIR = 300e6      # bundan büyük dosya tarayıcıda çalınmaz
INDIRME_SINIR = 1.5e9 if BULUT else 300e6   # bulutta tek yol indirme; yerelde büyük dosya klasörden alınır
ZIP_SINIR = 1e9             # toplamı bundan büyük dosyalar paketlenmez (diskte iki kopya olmasın)
# Paylaşılan bulut sunucusunda tek belgede en çok bu kadar karakter (~3 saat ses); yerelde sınır yok
EN_UZUN_BELGE = int(os.getenv("EMA_EN_UZUN_BELGE", "150000")) if BULUT else None

ORNEKLER = {
    "Karşılama": "Merhaba, size nasıl yardımcı olabilirim?",
    "Tarih ve tutar": "15 Ekim 2026 Çarşamba günü saat 14:30'da, 1.250.000 TL tutarındaki ödemeniz hesabınıza yatırılacak.",
    "Hikâye": "Sabahın erken saatlerinde liman henüz uyanmamıştı. Balıkçılar ağlarını sessizce topluyor, "
              "martılar ise teknelerin etrafında dönerek şanslarını deniyordu.",
}
ORNEKLEME_HIZLARI = [48000, 24000, 16000, 8000]


@st.cache_resource(show_spinner="Ses modeli yükleniyor…")
def model(kimlik):
    # İnce ayarlı sesler yönlendirmeli (CFG) çok adımlı örnekleyiciyle çalışır; paketin 4 adımlısı onlarda robotik.
    # Mac'te Apple GPU (MPS) kullanılır; paket kendiliğinden yalnızca NVIDIA GPU'yu seçiyor.
    return ses_kaydi.yukle(kimlik)


@st.cache_data(show_spinner="Belge okunuyor…", max_entries=4)
def belge_oku(ad, veri):
    return oku(ad, veri)


def seslendir(metin, tohum):
    """Bir metnin sesi, seçili örnekleme hızında float32 dizi."""
    return tts.seslendir(metin, hiz=hiz, tohum=tohum, ornekleme=ornekleme, adim=adim, yonlendirme=yonlendirme,
                         sicaklik=sicaklik)


def wav_bayt(audio, sample_rate):
    tampon = io.BytesIO()
    with wave.open(tampon, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm(audio))
    return tampon.getvalue()


def sure_yazisi(saniye):
    dk, sn = divmod(int(round(saniye)), 60)
    sa, dk = divmod(dk, 60)
    return f"{sa} sa {dk} dk" if sa else (f"{dk} dk {sn} sn" if dk else f"{sn} sn")


def ornek_sec():
    if secim := st.session_state.ornek:
        st.session_state.metin = ORNEKLER[secim]


st.set_page_config(page_title="Türkçe Seslendirme · EMA Lightning", page_icon="🔊")

st.title("Türkçe Seslendirme")
st.caption("EMA Lightning ile metni ve belgeleri sese çevirir. "
           + ("Metin ve belgeler bu Space'in sunucusunda işlenir, başka bir servise gönderilmez."
              if BULUT else "Her şey bu bilgisayarda çalışır, metin hiçbir yere gönderilmez."))

mevcut = {s.kimlik: s for s in ses_kaydi.mevcut_sesler()}
secilen = mevcut[st.segmented_control("Ses", list(mevcut), default=next(iter(mevcut)), key="ses_kimlik",
                                      required=True, format_func=lambda k: mevcut[k].ad)]
secilen_ses = secilen.ad
tts = model(secilen.kimlik)
ozel_ses = secilen.ozel
if secilen.aciklama:
    st.caption(secilen.aciklama)

with st.expander("Ayarlar"):
    hiz = st.slider("Konuşma hızı", 0.5, 2.0, 1.0, 0.05, format="%.2f×")
    sol, sag = st.columns(2)
    ornekleme = sol.selectbox("Örnekleme hızı", ORNEKLEME_HIZLARI, format_func=lambda r: f"{r // 1000} kHz",
                              help="Telefon santrali için 8 kHz, diğer her şey için 48 kHz.")
    seed = sag.number_input("Seed", min_value=0, value=None, step=1, placeholder="Rastgele",
                            help="Aynı seed, aynı metin için her seferinde aynı sesi verir.")
    adim, yonlendirme, sicaklik = EN_IYI["adim"], EN_IYI["yonlendirme"], EN_IYI["sicaklik"]
    if ozel_ses:
        st.markdown("**Örnekleyici** · yalnızca ince ayarlı sesler için")
        a, b, c = st.columns(3)
        adim = a.select_slider("Adım", [4, 8, 16, 32], value=EN_IYI["adim"],
                               help="Fazlası biraz daha temiz ama daha yavaş. 32 ölçümde en iyisi; 16 çok yakın ve iki kat hızlı.")
        yonlendirme = b.slider("Yönlendirme", 1.0, 3.0, EN_IYI["yonlendirme"], 0.25,
                               help="Metne bağlılık. 1: kapalı (anlaşılırlık düşer), 2: önerilen.")
        sicaklik = c.slider("Sıcaklık", 0.5, 1.0, EN_IYI["sicaklik"], 0.05,
                            help="Başlangıç gürültüsü. Düşükse daha düzgün ama daha tekdüze.")

sekme_metin, sekme_belge = st.tabs(["Metin", "Belgeden seslendir"])

# ---------------------------------------------------------------- Metin
with sekme_metin:
    st.pills("Örnek metinler", list(ORNEKLER), key="ornek", on_change=ornek_sec, label_visibility="collapsed")
    metin = st.text_area("Metin", key="metin", height=200, max_chars=10_000,
                         placeholder="Seslendirilecek Türkçe metni buraya yazın…")

    if st.button("Sese dönüştür", type="primary", icon=":material/graphic_eq:", width="stretch"):
        if not metin.strip():
            st.warning("Önce seslendirilecek bir metin yazın.")
        else:
            tohum = random.SystemRandom().randrange(2**31) if seed is None else int(seed)
            with st.spinner("Ses üretiliyor…"):
                basla = time.perf_counter()
                audio = seslendir(metin, tohum)
                sure = time.perf_counter() - basla
            st.session_state.sonuc = {
                "wav": wav_bayt(audio, ornekleme),
                "sure": len(audio) / ornekleme,
                "uretim": sure,
                "seed": tohum,
                "ornekleme": ornekleme,
                "ses": secilen_ses,
                "dosya": f"ses-{datetime.now():%Y%m%d-%H%M%S}.wav",
                "yeni": True,
            }

    if sonuc := st.session_state.get("sonuc"):
        st.audio(sonuc["wav"], format="audio/wav", autoplay=sonuc["yeni"])
        sonuc["yeni"] = False  # indirme gibi sonraki yenilemelerde ses kendiliğinden yeniden çalmasın
        bilgi, indir = st.columns([3, 1], vertical_alignment="center")
        bilgi.caption(f"{sonuc['ses']} · {sonuc['sure']:.1f} sn ses · {sonuc['uretim']:.2f} sn'de üretildi "
                      f"({sonuc['sure'] / sonuc['uretim']:.0f}× gerçek zaman) · "
                      f"{sonuc['ornekleme'] // 1000} kHz · seed {sonuc['seed']}")
        indir.download_button("WAV indir", sonuc["wav"], file_name=sonuc["dosya"], mime="audio/wav",
                              icon=":material/download:", on_click="ignore", width="stretch")

# ---------------------------------------------------------------- Belge
with sekme_belge:
    dosya = st.file_uploader(
        "Belge", type=BICIMLER, label_visibility="collapsed",
        help="Metinli PDF, EPUB, Word (docx, doc), RTF, ODT, HTML, TXT ve Markdown. "
             "Taranmış PDF'ler ve kopya korumalı e-kitaplar okunamaz.")
    bolumler, hata = None, None
    if dosya is not None:
        veri = dosya.getvalue()
        belge_kimligi = hashlib.sha1(veri).hexdigest()[:12]
        try:
            bolumler = belge_oku(dosya.name, veri)
        except BelgeHatasi as h:
            hata = str(h)
    if hata:
        st.error(hata, icon=":material/error:")

    if bolumler:
        duzeltmeler = st.session_state.setdefault("duzeltmeler", {})

        def bolum_metni(i):
            return duzeltmeler.get((belge_kimligi, i), bolumler[i].metin)

        n = len(bolumler)
        toplam = sum(len(b.metin) for b in bolumler)
        st.caption(f"{dosya.name} · {n} {'sayfa' if dosya.name.lower().endswith('.pdf') else 'bölüm'} · "
                   f"{toplam:,} karakter · tahmini {sure_yazisi(toplam / KARAKTER_SN / hiz)} ses".replace(",", "."))

        if n > 1:
            ilk, son = st.select_slider(
                "Seslendirilecek aralık", options=list(range(n)), value=(0, n - 1),
                format_func=lambda i: f"{i + 1}. {bolumler[i].baslik[:40]}")
        else:
            ilk, son = 0, 0
        secilen = list(range(ilk, son + 1))
        secili_karakter = sum(len(bolum_metni(i)) for i in secilen)
        tahmini_ses = secili_karakter / KARAKTER_SN / hiz

        with st.expander(f"Gözden geçir ve düzelt · {len(secilen)} bölüm, "
                         f"tahmini {sure_yazisi(tahmini_ses)} ses"):
            st.caption("PDF'ten çıkarılan metin kusursuz olmayabilir: tablolar, dipnotlar ve sütunlar karışabilir. "
                       "Seslendirmeden önce burada düzeltebilirsin; paragrafları boş satırla ayır.")
            gozden = st.selectbox("Bölüm", secilen, format_func=lambda i: f"{i + 1}. {bolumler[i].baslik[:60]}",
                                  key=f"gozden_{belge_kimligi}")
            anahtar = f"duzelt_{belge_kimligi}_{gozden}"

            def kaydet(i=gozden, k=anahtar):
                duzeltmeler[(belge_kimligi, i)] = st.session_state[k]

            st.text_area("Metin", bolum_metni(gozden), height=260, key=anahtar, on_change=kaydet,
                         label_visibility="collapsed")

        sol, sag = st.columns(2)
        tek_dosya = sol.radio("Çıktı", ["Tek dosya", "Bölüm başına ayrı dosya (ZIP)"], horizontal=False,
                              disabled=len(secilen) == 1) == "Tek dosya" or len(secilen) == 1
        bicimler = ["MP3", "WAV"] if FFMPEG else ["WAV"]
        bicim = sag.radio("Biçim", bicimler, help="MP3 çok daha küçük: 1 saatlik ses MP3'te ~30 MB, WAV'da ~330 MB."
                          + ("" if FFMPEG else " MP3 için ffmpeg gerekli: brew install ffmpeg"))
        if bicim == "WAV" and tahmini_ses * ornekleme * 2 / (1 if tek_dosya else len(secilen)) > WAV_SINIR:
            st.info(f"Bu belge yaklaşık {tahmini_ses * ornekleme * 2 / 1e9:.1f} GB WAV eder. WAV dosyası en çok 4 GB "
                    "olabildiği için ses birkaç parçaya bölünecek. Uzun belgeler için MP3 öneririm.",
                    icon=":material/info:")

        cok_uzun = EN_UZUN_BELGE is not None and secili_karakter > EN_UZUN_BELGE
        if cok_uzun:
            st.warning(f"Bu sunucuda tek seferde en çok {EN_UZUN_BELGE:,} karakter seslendirilebiliyor; seçili aralık "
                       f"{secili_karakter:,} karakter. Aralığı daraltıp parça parça seslendirebilirsin."
                       .replace(",", "."), icon=":material/warning:")

        if st.button("Belgeyi seslendir", type="primary", icon=":material/menu_book:", width="stretch",
                     disabled=cok_uzun):
            ad = Path(dosya.name).stem
            if BULUT:  # paylaşılan sunucu: bir saatten eski çıktıları temizle, her işe benzersiz klasör
                for eski in BELGE_CIKTI.glob("*") if BELGE_CIKTI.exists() else []:
                    if time.time() - eski.stat().st_mtime > 3600:
                        shutil.rmtree(eski, ignore_errors=True)
            klasor = BELGE_CIKTI / f"{ad}-{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
            klasor.mkdir(parents=True, exist_ok=True)
            uzanti = ".mp3" if bicim == "MP3" else ".wav"
            taban_tohum = random.SystemRandom().randrange(2**31) if seed is None else int(seed)
            paragraf_listesi = {i: [p for p in bolum_metni(i).split("\n\n") if any(c.isalpha() for c in p)]
                                for i in secilen}
            toplam_harf = sum(len(p) for ps in paragraf_listesi.values() for p in ps) or 1
            ilerleme = st.progress(0.0, text="Başlıyor…")
            basla, biten_harf, ses_suresi, dosyalar, yazici = time.perf_counter(), 0, 0.0, [], None
            try:
                for sira, i in enumerate(secilen, 1):
                    if yazici is None:
                        dosya_adi = (f"{ad}{uzanti}" if tek_dosya else
                                     f"{sira:03d} - {bolumler[i].baslik[:50]}".replace("/", "-") + uzanti)
                        yazici = SesYazici(klasor / dosya_adi, bicim, ornekleme)
                    else:  # tek dosyada bölümler arası
                        yazici.yaz(np.zeros(int(BOLUM_ARASI * ornekleme)))
                        ses_suresi += BOLUM_ARASI
                    for j, paragraf in enumerate(paragraf_listesi[i]):
                        audio = seslendir(paragraf, (taban_tohum + 7919 * i + j) % 2**31)
                        yazici.yaz(audio)
                        yazici.yaz(np.zeros(int(PARAGRAF_ARASI * ornekleme)))
                        ses_suresi += len(audio) / ornekleme + PARAGRAF_ARASI
                        biten_harf += len(paragraf)
                        oran = biten_harf / toplam_harf
                        gecen = time.perf_counter() - basla
                        ilerleme.progress(min(oran, 1.0), text=f"{sira}/{len(secilen)}. bölüm · %{100 * oran:.0f} · "
                                          f"{sure_yazisi(ses_suresi)} ses hazır · kalan ~{sure_yazisi(gecen / oran - gecen)}")
                    if not tek_dosya:
                        dosyalar += yazici.kapat()
                        yazici = None
            finally:  # hata ya da durdurmada da açık dosyayı düzgün kapat: o ana kadarki ses kullanılabilir kalsın
                if yazici is not None:
                    dosyalar += yazici.kapat()

            paket = None
            # Bulutta birden çok dosyayı almanın tek yolu paket; yerelde dev dosyalar klasörden alınır
            if len(dosyalar) > 1 and (BULUT or sum(d.stat().st_size for d in dosyalar) < ZIP_SINIR):
                paket = klasor / f"{ad}.zip"
                with zipfile.ZipFile(paket, "w", zipfile.ZIP_STORED) as z:  # ses zaten sıkıştırılmış
                    for d in dosyalar:
                        z.write(d, d.name)
            ilerleme.empty()
            st.session_state.belge_sonuc = {"dosyalar": dosyalar, "paket": paket, "klasor": klasor,
                                            "sure": ses_suresi, "uretim": time.perf_counter() - basla,
                                            "ses": secilen_ses}

    if bs := st.session_state.get("belge_sonuc"):
        st.success(f"{bs['ses']} · {sure_yazisi(bs['sure'])} ses {sure_yazisi(bs['uretim'])}'de üretildi "
                   f"({bs['sure'] / max(bs['uretim'], 1e-9):.0f}× gerçek zaman) · {len(bs['dosyalar'])} dosya",
                   icon=":material/check_circle:")
        ilk_dosya = bs["dosyalar"][0]
        if ilk_dosya.stat().st_size < ONIZLEME_SINIR:   # dev dosyayı tarayıcıya yükleme
            st.audio(str(ilk_dosya))
        indirilecek = bs["paket"] or (ilk_dosya if len(bs["dosyalar"]) == 1 else None)
        finder = not BULUT and sys.platform == "darwin"
        sol, sag = st.columns([3, 2]) if finder else (st.container(), None)
        if indirilecek is not None and indirilecek.stat().st_size < INDIRME_SINIR:
            mime = {".zip": "application/zip", ".mp3": "audio/mpeg", ".wav": "audio/wav"}[indirilecek.suffix]
            sol.download_button(f"{indirilecek.suffix[1:].upper()} indir ({indirilecek.stat().st_size / 1e6:.1f} MB)",
                                lambda p=indirilecek: p.read_bytes(), file_name=indirilecek.name, mime=mime,
                                icon=":material/download:", on_click="ignore", width="stretch")
        else:
            sol.caption("Dosyalar tarayıcıdan indirmek için çok büyük; klasörden alabilirsin." if finder else
                        "Ses tarayıcıdan indirmek için çok büyük; aralığı daraltıp parça parça seslendir.")
        if finder:
            sag.button("Klasörü Finder'da aç", icon=":material/folder_open:", width="stretch",
                       on_click=lambda k=bs["klasor"]: subprocess.run(["open", str(k)], check=False))
            st.caption(f"Klasör: {bs['klasor']}")

st.divider()
st.caption("Bu sesler yapay zekâ ile üretilir. Yayımlarken ya da bir görüşmede kullanırken dinleyicilere bunu belirtin.")

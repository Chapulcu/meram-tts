"""Meram API'si (Türkçe seslendirme): başka uygulamalar metni buradan sese çevirir.

Çalıştırmak için (Ema klasöründe):
    uv run python api.py                         # yalnızca bu bilgisayardan: http://127.0.0.1:8000
    EMA_API_ANAHTARI=gizli EMA_API_HOST=0.0.0.0 uv run python api.py     # ağdaki diğer cihazlardan da
Belgeler ve deneme ekranı: http://127.0.0.1:8000/docs

Uçlar:
    GET  /sesler                  kullanılabilir sesler
    POST /seslendir               metin -> ses dosyası (wav, mp3, opus, aac, flac, pcm)
    GET  /seslendir?metin=…       aynısı, tarayıcıdan ya da <audio src="…"> ile denemek için
    POST /seslendir/akis          ses parça parça gelir; uzun metinde ilk ses hemen çalmaya başlar
    POST /v1/audio/speech         OpenAI TTS API'siyle uyumlu; OpenAI istemcisi olan uygulamalar adresi değiştirerek kullanır
    GET  /saglik                  sunucu ayakta mı, hangi sesler yüklü

Ayarlar (ortam değişkenleri):
    EMA_API_HOST       dinlenecek adres (varsayılan 127.0.0.1, yalnızca bu bilgisayar)
    EMA_API_PORT       port (varsayılan 8000)
    EMA_API_ANAHTARI   tanımlıysa her istek "Authorization: Bearer <anahtar>" ya da "X-API-Key: <anahtar>" ister.
                       127.0.0.1 dışında bir adreste anahtarsız başlatılmaz.
    EMA_API_CORS       tarayıcıdan çağıracak web uygulamalarının kökenleri, virgülle (örn. http://localhost:3000)
"""

import hmac
import io
import os
import subprocess
import sys
import tempfile
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

import ses_kaydi
from ses_dosyasi import FFMPEG, pcm

HOST = os.getenv("EMA_API_HOST", "127.0.0.1")
PORT = int(os.getenv("EMA_API_PORT", "8000"))
ANAHTAR = os.getenv("EMA_API_ANAHTARI", "").strip()
EN_UZUN = 100_000           # tek istekte en çok karakter (~2 saat ses)
EN_UZUN_AKIS = 2_000_000    # akışta

Bicim = Literal["wav", "mp3", "opus", "aac", "flac", "pcm"]
MIME = {"wav": "audio/wav", "mp3": "audio/mpeg", "opus": "audio/ogg", "aac": "audio/aac", "flac": "audio/flac",
        "pcm": "audio/pcm"}
FFMPEG_CIKTI = {"mp3": ["-b:a", "64k", "-f", "mp3"], "opus": ["-c:a", "libopus", "-b:a", "32k", "-f", "ogg"],
                "aac": ["-c:a", "aac", "-b:a", "64k", "-f", "adts"], "flac": ["-f", "flac"]}


# ---------------------------------------------------------------- yardımcılar

def kodla(audio, sr, bicim):
    if bicim == "pcm":
        return pcm(audio)
    if bicim == "wav":
        tampon = io.BytesIO()
        with wave.open(tampon, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm(audio))
        return tampon.getvalue()
    if not FFMPEG:
        raise HTTPException(501, f"{bicim} için ffmpeg gerekli (brew install ffmpeg); wav ya da pcm kullanın")
    # FLAC süre bilgisini dosyanın başına sonradan yazar; bu yüzden akışa değil geçici dosyaya kodlanır
    with tempfile.NamedTemporaryFile(suffix=f".{bicim}") as gecici:
        hedef = gecici.name if bicim == "flac" else "-"
        sonuc = subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i",
                                "-", *FFMPEG_CIKTI[bicim], hedef], input=pcm(audio), capture_output=True)
        if sonuc.returncode != 0:
            raise HTTPException(500, f"{bicim} kodlanamadı: {sonuc.stderr.decode(errors='ignore').strip()}")
        return Path(gecici.name).read_bytes() if bicim == "flac" else sonuc.stdout


def akis_wav_basligi(sr):
    """Uzunluğu bilinmeyen akış için WAV başlığı: boyut alanları en büyük değerde (oynatıcılar bunu kabul eder)."""
    return (b"RIFF" + (0xFFFFFFFF).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little")
            + (1).to_bytes(2, "little") + (1).to_bytes(2, "little") + sr.to_bytes(4, "little")
            + (sr * 2).to_bytes(4, "little") + (2).to_bytes(2, "little") + (16).to_bytes(2, "little")
            + b"data" + (0xFFFFFFFF - 36).to_bytes(4, "little"))


def ses_bul(kimlik):
    if kimlik is None:
        return ses_kaydi.varsayilan_ses()
    ses = ses_kaydi.bul(kimlik)
    if ses is None:
        raise HTTPException(404, {"hata": f"'{kimlik}' adında bir ses yok",
                                  "sesler": [s.kimlik for s in ses_kaydi.mevcut_sesler()]})
    return ses


def anahtar_denetle(authorization: str | None = Header(None), x_api_key: str | None = Header(None)):
    if not ANAHTAR:
        return
    verilen = x_api_key or (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(verilen.encode(), ANAHTAR.encode()):
        raise HTTPException(401, "Geçersiz ya da eksik API anahtarı", headers={"WWW-Authenticate": "Bearer"})


def seslendir(ses, metin, hiz, tohum, ornekleme, adim=None, yonlendirme=None, sicaklik=None):
    tohum = ses_kaydi.rastgele_tohum() if tohum is None else tohum
    basla = time.perf_counter()
    audio = ses_kaydi.yukle(ses.kimlik).seslendir(metin, hiz=hiz, tohum=tohum, ornekleme=ornekleme, adim=adim,
                                                  yonlendirme=yonlendirme, sicaklik=sicaklik)
    return audio, tohum, time.perf_counter() - basla


def ses_yaniti(audio, ornekleme, bicim, ses, tohum, uretim):
    sure = len(audio) / ornekleme
    return Response(kodla(audio, ornekleme, bicim), media_type=MIME[bicim], headers={
        "X-Ses": ses.kimlik, "X-Tohum": str(tohum), "X-Ses-Suresi": f"{sure:.3f}",
        "X-Uretim-Suresi": f"{uretim:.3f}", "X-Ornekleme": str(ornekleme),
        "Content-Disposition": f'inline; filename="ses.{bicim}"'})


# ---------------------------------------------------------------- uygulama

@asynccontextmanager
async def omur(_):
    varsayilan = ses_kaydi.varsayilan_ses()
    print(f"Varsayılan ses yükleniyor: {varsayilan.ad} ({ses_kaydi.CIHAZ})", flush=True)
    ses_kaydi.yukle(varsayilan.kimlik)  # ısıtılmış olarak hazır olsun; diğer sesler ilk istekte yüklenir
    print(f"Hazır: http://{HOST}:{PORT}  ·  belgeler: http://{HOST}:{PORT}/docs  ·  "
          f"anahtar {'açık' if ANAHTAR else 'kapalı'}", flush=True)
    yield


app = FastAPI(title="Meram API · Türkçe Seslendirme", version="1.0", lifespan=omur,
              description="EMA Lightning tabanlı, tamamen bu bilgisayarda çalışan Türkçe metinden sese API'si. "
                          "Varsayılan ses listedeki ilk mevcut sestir. Üretilen sesler yapay zekâ "
                          "ile üretilmiştir; yayımlarken ya da bir görüşmede kullanırken dinleyicilere belirtin.")
if cors := [k.strip() for k in os.getenv("EMA_API_CORS", "").split(",") if k.strip()]:
    app.add_middleware(CORSMiddleware, allow_origins=cors, allow_methods=["*"], allow_headers=["*"],
                       expose_headers=["X-Ses", "X-Tohum", "X-Ses-Suresi", "X-Uretim-Suresi", "X-Ornekleme"])
korumali = [Depends(anahtar_denetle)]


class SeslendirIstek(BaseModel):
    metin: str = Field(..., min_length=1, max_length=EN_UZUN, description="Seslendirilecek Türkçe metin; sayılar, "
                       "tarihler ve kısaltmalar kendiliğinden okunur")
    ses: str | None = Field(None, description="Ses kimliği (GET /sesler); boşsa varsayılan ses", examples=["kadin"])
    hiz: float = Field(1.0, ge=0.25, le=4.0, description="Konuşma hızı")
    tohum: int | None = Field(None, ge=0, description="Aynı tohum aynı metin için aynı sesi verir; boşsa rastgele")
    ornekleme: Literal[48000, 24000, 16000, 8000] = Field(48000, description="Örnekleme hızı (Hz); telefon için 8000")
    bicim: Bicim = Field("wav", description="Ses dosyası biçimi; pcm: başlıksız 16 bit tek kanal")
    adim: Literal[4, 8, 16, 32] | None = Field(None, description="Yalnızca ince ayarlı sesler: örnekleme adımı (varsayılan 32)")
    yonlendirme: float | None = Field(None, ge=1.0, le=4.0, description="Yalnızca ince ayarlı sesler: metne bağlılık (varsayılan 2)")
    sicaklik: float | None = Field(None, ge=0.3, le=1.2, description="Yalnızca ince ayarlı sesler: başlangıç gürültüsü (varsayılan 0,7)")


@app.get("/", include_in_schema=False)
def kok():
    return {"ad": app.title, "belgeler": "/docs", "sesler": "/sesler", "saglik": "/saglik"}


@app.get("/saglik", summary="Sunucu durumu")
def saglik():
    return {"durum": "hazir", "cihaz": ses_kaydi.CIHAZ, "varsayilan_ses": ses_kaydi.varsayilan_ses().kimlik,
            "yuklu_sesler": ses_kaydi.yuklenmis(), "anahtar": bool(ANAHTAR), "ffmpeg": bool(FFMPEG)}


@app.get("/sesler", summary="Kullanılabilir sesler", dependencies=korumali)
def sesler():
    varsayilan = ses_kaydi.varsayilan_ses().kimlik
    return [{"kimlik": s.kimlik, "ad": s.ad, "aciklama": s.aciklama, "ince_ayarli": s.ozel,
             "varsayilan": s.kimlik == varsayilan, "yuklu": s.kimlik in ses_kaydi.yuklenmis()}
            for s in ses_kaydi.mevcut_sesler()]


@app.post("/seslendir", summary="Metni sese çevir", dependencies=korumali,
          response_class=Response, responses={200: {"content": {m: {} for m in MIME.values()}}})
def seslendir_post(istek: SeslendirIstek):
    ses = ses_bul(istek.ses)
    audio, tohum, uretim = seslendir(ses, istek.metin, istek.hiz, istek.tohum, istek.ornekleme, istek.adim,
                                     istek.yonlendirme, istek.sicaklik)
    return ses_yaniti(audio, istek.ornekleme, istek.bicim, ses, tohum, uretim)


@app.get("/seslendir", summary="Metni sese çevir (GET; hızlı deneme ve <audio> etiketi için)",
         dependencies=korumali, response_class=Response)
def seslendir_get(metin: str = Query(..., min_length=1, max_length=5000), ses: str | None = None,
                  hiz: float = Query(1.0, ge=0.25, le=4.0), tohum: int | None = Query(None, ge=0),
                  bicim: Bicim = "mp3", ornekleme: Literal[48000, 24000, 16000, 8000] = 48000):
    s = ses_bul(ses)
    audio, t, uretim = seslendir(s, metin, hiz, tohum, ornekleme)
    return ses_yaniti(audio, ornekleme, bicim, s, t, uretim)


class AkisIstek(SeslendirIstek):
    metin: str = Field(..., min_length=1, max_length=EN_UZUN_AKIS)
    bicim: Literal["wav", "pcm"] = Field("wav", description="wav: başlık bir kez en başta, uzunluk bilinmiyor; "
                                         "pcm: başlıksız 16 bit tek kanal")


@app.post("/seslendir/akis", summary="Metni parça parça sese çevir (akış)", dependencies=korumali,
          response_class=StreamingResponse)
def seslendir_akis(istek: AkisIstek):
    """Ses, her parça (en çok ~10 sn'lik cümle grubu) üretildikçe gönderilir. Sesli asistanlar ve uzun metinler için:
    ilk ses tamamı bitmeden çalmaya başlar. İstemci bağlantıyı keserse üretim durur."""
    ses = ses_bul(istek.ses)
    tohum = ses_kaydi.rastgele_tohum() if istek.tohum is None else istek.tohum
    motor = ses_kaydi.yukle(ses.kimlik)

    def uret():
        if istek.bicim == "wav":
            yield akis_wav_basligi(istek.ornekleme)
        for parca in motor.parcalar(istek.metin, hiz=istek.hiz, tohum=tohum, ornekleme=istek.ornekleme,
                                    adim=istek.adim, yonlendirme=istek.yonlendirme, sicaklik=istek.sicaklik):
            yield pcm(parca)

    return StreamingResponse(uret(), media_type=MIME[istek.bicim], headers={
        "X-Ses": ses.kimlik, "X-Tohum": str(tohum), "X-Ornekleme": str(istek.ornekleme)})


# ---------------------------------------------------------------- OpenAI uyumlu uç

class OpenAIIstek(BaseModel):
    model: str = Field("ema", description="Yok sayılır; uyumluluk için")
    input: str = Field(..., min_length=1, max_length=EN_UZUN)
    voice: str | None = Field(None, description="Ses kimliği ya da adı (kadin, elif, erkek, ema, …). Tanınmayan adlar (alloy gibi) "
                              "varsayılan sese gider")
    response_format: Bicim = "mp3"
    speed: float = Field(1.0, ge=0.25, le=4.0)
    instructions: str | None = Field(None, description="Yok sayılır")


@app.post("/v1/audio/speech", summary="OpenAI uyumlu seslendirme", dependencies=korumali,
          response_class=Response, tags=["OpenAI uyumlu"])
def openai_speech(istek: OpenAIIstek):
    """OpenAI'nin `audio/speech` ucuyla aynı istek ve yanıt. OpenAI istemcisi kullanan bir uygulamada temel adresi
    `http://127.0.0.1:8000/v1` yapmak yeter. Ses 24 kHz üretilir (OpenAI'yle aynı; pcm de 24 kHz, 16 bit)."""
    ses = (ses_kaydi.bul(istek.voice) if istek.voice else None) or ses_kaydi.varsayilan_ses()
    audio, tohum, uretim = seslendir(ses, istek.input, istek.speed, None, 24000)
    return ses_yaniti(audio, 24000, istek.response_format, ses, tohum, uretim)


@app.get("/v1/models", include_in_schema=False, dependencies=korumali)
def openai_modeller():
    return {"object": "list", "data": [{"id": s.kimlik, "object": "model", "owned_by": "yerel"}
                                       for s in ses_kaydi.mevcut_sesler()]}


if __name__ == "__main__":
    import uvicorn

    if HOST not in ("127.0.0.1", "localhost", "::1") and not ANAHTAR:
        sys.exit(f"{HOST} adresi ağa açık; önce bir anahtar tanımlayın: EMA_API_ANAHTARI=... uv run python api.py")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")

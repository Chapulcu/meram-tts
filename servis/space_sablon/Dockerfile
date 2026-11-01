# Hugging Face Docker Space: Türkçe seslendirme (Streamlit + EMA Lightning, yalnızca işlemci).
# Yerelde denemek için:  docker build -t ema-seslendirme . && docker run --rm -p 8501:8501 ema-seslendirme
FROM python:3.12-slim

# ffmpeg: MP3 çıktısı · antiword: eski Word (.doc) belgeleri
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg antiword \
 && rm -rf /var/lib/apt/lists/*

# Hugging Face, Space'leri 1000 numaralı kullanıcıyla çalıştırır; dosyaların sahibi o olmalı
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    EMA_BULUT=1 \
    EMA_SESLER={{SESLER_LISTESI}} \
    EMA_AKUSTIK_KLASORU=/home/user/app/agirliklar \
    EMA_COZUCU_KLASORU=/home/user/app/agirliklar
WORKDIR /home/user/app

# torch'un yalnızca işlemci sürümü: CUDA'lı sürümden yaklaşık 2 GB küçük; ücretsiz donanımda GPU yok
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user "$(grep '^torch==' requirements.txt)" --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

# EMA'nın kendi ağırlıkları (orijinal ses ve metin işleyici) derlemede iner; Space hızlı açılsın
RUN python -c "from huggingface_hub import hf_hub_download as d; [d('canberkkkkkk/ema-lightning', f) for f in ('ema.pt', 'decoder.pt', 'config.json')]"

COPY --chown=user . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]

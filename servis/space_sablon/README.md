---
title: Meram
emoji: 🔊
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 8501
pinned: false
license: apache-2.0
short_description: Türkçe metin ve belgelerden hızlı seslendirme
---

# Meram

**Türkçe seslendirme** · kaynak kod: https://github.com/Chapulcu/meram-tts

Türkçe metni ve belgeleri sese çeviren bir uygulama. Altında [EMA Lightning](https://huggingface.co/canberkkkkkk/ema-lightning)
var: 8,6 milyon parametrelik, işlemcide bile gerçek zamandan hızlı çalışan bir Türkçe konuşma modeli. Sayılar, tarihler,
saatler, tutarlar ve kısaltmalar kendiliğinden okunur.

## Sesler

{{SESLER}}

Bu sesler [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)'nin yazılı tariften ses tasarlama özelliğiyle üretildi;
hiçbiri gerçek bir kişinin sesi değildir. Her ses için EMA Lightning'in akustik modeli ve çözücüsü, o sesle üretilmiş
yaklaşık 10 saatlik veriyle yeniden eğitildi. Sesler yönlendirmeli (CFG), çok adımlı bir örnekleyiciyle çalışır;
Ayarlar'da adım, yönlendirme ve sıcaklık değiştirilebilir.

## Belgeden seslendirme

| Biçim | Nasıl okunur |
|---|---|
| PDF | Yalnızca metin katmanı olan PDF'ler. Tireyle bölünmüş kelimeler birleşir, tekrar eden üst/alt bilgiler ve sayfa numaraları atılır. Taranmış PDF'ler okunmaz. |
| EPUB | Kitabın kendi okuma sırasıyla, bölüm bölüm. Kopya korumalı (DRM) kitaplar okunmaz. |
| DOCX, ODT | Başlık stilleri bölüm sınırı olur. |
| DOC, RTF | Eski Word belgeleri antiword ile, RTF striprtf ile; Türkçe kod sayfası tanınır. |
| HTML, TXT, MD | Başlıklar bölüm sınırı olur; eski Türkçe kodlamalı (Windows-1254) metinler tanınır. |

Çıkarılan metin seslendirmeden önce gözden geçirilip düzeltilebilir. Sonuç tek bir MP3/WAV ya da bölüm başına ayrı
dosyalar (ZIP) olarak indirilir.

**Sınırlar:** Ücretsiz donanımda (2 işlemci çekirdeği) GPU yoktur; ses yine de gerçek zamandan hızlı üretilir. Paylaşılan
sunucuyu kilitlememek için tek seferde en çok 150.000 karakter (yaklaşık 3 saat ses) seslendirilir; daha uzun belgeler
aralık seçilerek parça parça seslendirilir. Bu sınır `EMA_EN_UZUN_BELGE` ortam değişkeniyle değiştirilebilir.

## Sorumlu kullanım

Bu uygulamadaki sesler yapay zekâ ile üretilmiştir. Yayımlarken ya da bir görüşmede kullanırken dinleyicilere bunu
belirtin ("Bu ses yapay zekâ ile üretilmiştir."). Sesleri gerçek bir kişinin sesiymiş gibi sunmayın, dolandırıcılıkta ya
da başkasının yerine geçmek için kullanmayın.

## Yerelde çalıştırma

```bash
docker build --platform linux/amd64 -t meram-tts .
docker run --rm --platform linux/amd64 -p 8501:8501 meram-tts
```

Sonra tarayıcıda http://localhost:8501 adresini açın. `--platform linux/amd64` Hugging Face'in donanımıyla aynı
ortamı kurar; metin normalleştirici `normalizer-tr`'nin ARM Linux için hazır paketi olmadığından Apple Silicon Mac'lerde
bu bayrak gerekir (Docker onu taklit ederek çalıştırır, biraz yavaştır).

## Teşekkür ve lisanslar

- [EMA Lightning](https://huggingface.co/canberkkkkkk/ema-lightning), Canberk Aslan, Apache 2.0
- [normalizer-tr](https://github.com/erdemtuna/normalizer-tr), Erdem Tuna, Apache 2.0 (Türkçe metin normalleştirme)
- [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2), OpenBMB, Apache 2.0 (ince ayarlı seslerin çözücüsüne öğretmen olarak)
- Eğitim cümleleri: [Mozilla Common Voice](https://github.com/common-voice/common-voice) Türkçe cümle derlemi (CC0)

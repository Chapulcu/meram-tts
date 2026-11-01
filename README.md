# Meram

**Türkçe seslendirme** · github.com/Chapulcu/meram-tts

[EMA Lightning](https://huggingface.co/canberkkkkkk/ema-lightning) tabanlı, tamamen yerelde çalışan Türkçe metinden
sese sistemi: yeni seslerle ince ayarlanmış modeller, metin ve belgeden seslendirme yapan bir Streamlit uygulaması,
OpenAI uyumlu bir API ve kendi sesini eğitmek için uçtan uca bir eğitim hattı.

EMA Lightning 8,6 milyon parametrelik, çok hızlı bir Türkçe konuşma modeli; ama tek bir sesi var. Bu proje onun
akustik modelini ve çözücüsünü yeni seslerle yeniden eğitiyor. Sayılar, tarihler, saatler, tutarlar ve kısaltmalar
kendiliğinden okunur.

## Sesler

| Ad | Ses | Harf hatası* | Perde oynaklığı* | Düzenlilik* |
|---|---|---|---|---|
| **Elif** | Kadın · otuzlu yaşlarda, sıcak ve sakin bir anlatıcı | %0,8 (hedef %0,2) | 3,46 yt (hedef 3,61) | 0,759 (hedef 0,755) |
| **Kerem** | Erkek · otuzlu yaşlarda, tok ve güven veren bir bariton | %1,6 (hedef %0,2) | 5,12 yt (hedef 4,74) | 0,584 (hedef 0,601) |
| **Defne** | Kız çocuk · sekiz yaşında, neşeli ve tatlı | %0,3 (hedef %0,2) | 3,51 yt (hedef 3,47) | 0,776 (hedef 0,771) |
| EMA (orijinal) | EMA Lightning'in kendi sesi | | | |

\* Eğitimde görülmemiş 50 cümlede; harf hatası Whisper large-v3-turbo ile. "Hedef", sesin eğitim verisini üreten
öğretmen modelin (VoxCPM2) kendi değeri. Perde oynaklığı tonlamanın canlılığını, düzenlilik harmoniklerin netliğini
ölçer; hedefe yakın olması sesin robotik olmadığını gösterir.

Sesler [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)'nin yazılı tariften ses tasarlama özelliğiyle üretildi;
hiçbiri gerçek bir kişinin sesi değildir.

**Hız:** Apple M4 Max'te (GPU) 8,7 saniyelik ses ~115 ms'de üretilir (gerçek zamanın ~76 katı); yalnızca 2 işlemci
çekirdeğiyle ~15 katı.

## Hızlı başlangıç

[uv](https://docs.astral.sh/uv/) ve Python 3.12 gerekir. MP3 çıktısı için [ffmpeg](https://ffmpeg.org/).

```bash
git clone https://github.com/Chapulcu/meram-tts.git && cd meram-tts
uv sync
uv run streamlit run app.py          # uygulama: http://localhost:8501
uv run python api.py                 # API: http://127.0.0.1:8000  (belgeler: /docs)
```

İlk çalıştırmada EMA Lightning'in kendi ağırlıkları Hugging Face'ten iner (~35 MB); ince ayarlı seslerin ağırlıkları
bu depoda.

## Uygulama

- **Metin:** yaz, sesi seç, dinle ya da WAV olarak indir.
- **Belgeden seslendir:** PDF (metin katmanı olanlar), EPUB, DOCX, DOC, ODT, RTF, HTML, TXT ve Markdown.
  Bölüm aralığı seçilir, çıkarılan metin seslendirmeden önce düzeltilebilir; sonuç tek bir MP3/WAV ya da bölüm başına
  ayrı dosyalar (ZIP). Kitap uzunluğunda belgeler (12 saati aşan ses dahil) parça parça diske yazılır.
- **Ayarlar:** konuşma hızı, örnekleme hızı (48/24/16/8 kHz), tohum; ince ayarlı seslerde örnekleme adımı,
  yönlendirme (CFG) ve sıcaklık.

## API

```bash
# Ses dosyası (wav, mp3, opus, aac, flac, pcm)
curl -X POST http://127.0.0.1:8000/seslendir -H "Content-Type: application/json" \
  -d '{"metin": "Toplantı 15 Ekim Çarşamba saat 14:30 olarak planlandı.", "ses": "elif", "bicim": "mp3"}' -o ses.mp3

# Akış: uzun metinde ilk ses ~200 ms'de gelir
curl -X POST http://127.0.0.1:8000/seslendir/akis -H "Content-Type: application/json" \
  -d '{"metin": "Uzun bir metin…", "ses": "kerem", "bicim": "wav"}' -o akis.wav
```

| Uç | |
|---|---|
| `GET /sesler` | kullanılabilir sesler |
| `POST /seslendir`, `GET /seslendir?metin=…` | metin -> ses dosyası |
| `POST /seslendir/akis` | parça parça ses (sesli asistanlar, uzun metinler) |
| `POST /v1/audio/speech` | **OpenAI uyumlu**; OpenAI istemcisi olan uygulamalarda temel adresi `http://127.0.0.1:8000/v1` yapmak yeter |

Ses alanı kimlik (`kadin`, `erkek`, `kiz-cocuk`, `ema`) ya da ad (`elif`, `kerem`, `defne`) kabul eder.
Varsayılan olarak yalnızca bu bilgisayardan erişilir; ağa açmak için `EMA_API_ANAHTARI` tanımlanmalı (anahtarsız
ağa açık başlamaz). Ayrıntılar `api.py` başında.

## macOS servisi

Uygulama ve API, oturum açılınca kendiliğinden başlayan ve çökünce yeniden kalkan launchd servisleri olarak kurulabilir:

```bash
bash servis/servisler.sh kur        # ya da: guncelle, durum, kayit, durdur, baslat, kaldir
```

macOS'un gizlilik koruması servislerin `~/Documents` altını okumasına izin vermediği için servisler kodun ve
ağırlıkların bir kopyasıyla `~/Library/Application Support/ema-seslendirme` altından çalışır; belge çıktıları
`~/Music/EMA Seslendirme` klasörüne yazılır.

## Hugging Face Space

`bash servis/space_hazirla.sh` bu depodan kendi başına çalışan bir Docker Space klasörü (`huggingface_space/`) üretir:
Python 3.12, işlemci sürümü torch, ffmpeg, antiword; yalnızca seçilen sesler. Hugging Face'in x86_64 ortamında
test edildi. (Not: Hugging Face, Docker Space'lerini ücretsiz donanımda çalıştırmak için PRO abonelik istiyor.)

## Kendi sesini eklemek

Eğitim hattı iki yoldan veri üretebilir: **VoxCPM2 ile tasarlanmış bir ses** (yazılı tariften) ya da **bir kayıttan
klonlanan ses** (Chatterbox). Ses modeli ve çözücü o veriyle ince ayarlanır.

| Adım | Ne yapar | Betik |
|---|---|---|
| 1. Cümleler | Common Voice Türkçe derleminden (CC0) 60 bin cümle + tarih/saat/tutar kalıpları; okunuşa çevrilir | `deneyler/veri/metin_hazirla.py` |
| 2. Veri üretimi | ~11 bin cümle (~10 saat) yeni sesle seslendirilir | `sesler/ses_hatti.sh` (VoxCPM2) ya da `deneyler/veri/uret_parti.sh` (Chatterbox) |
| 3. Denetim | Whisper ile geri yazılır; harf hatası %5'i aşan cümleler atılır (%91–95 geçiyor) | `deneyler/veri/denetle.py` |
| 4. Çözücü | EMA'nın küçük çözücüsü, VoxCPM2'nin büyük çözücüsü öğretmen olarak, yeni sese uyarlanır | `deneyler/cozucu_uyarla.py` |
| 5. Hizalama | Whisper kelime zamanları harf düzeyinde hizalanır; sesler VoxCPM2 kodlayıcısıyla latentlere çevrilir | `deneyler/akustik/veri_hazirla.py` |
| 6. Akustik model | EMA'nın ağırlıklarından başlanır; süre + akış eşleme (flow matching), %15 koşul bırakma | `deneyler/akustik/egit.py` |
| 7. Ölçüm | Harf hatası, perde, oynaklık, düzenlilik; örnekleme ayarları karşılaştırılır | `deneyler/akustik/cfg_dene.py` |

`sesler/ses_hatti.sh <ad>` tasarlanmış bir ses için 2–7. adımları sırayla çalıştırır (Mac'te 10 saatlik veriyle
yaklaşık bir gün). Eğitilen sesi uygulamada göstermek için `ses_kaydi.py`'deki listeye bir satır ya da kişisel sesler
için `kisisel_sesler.ornek.py`'yi `kisisel_sesler.py` olarak kopyalayıp doldurmak yeter. Eğitim ortamları:
`deneyler/requirements.txt` ve `sesler/requirements.txt`.

**Neden işe yarıyor:** EMA Lightning, VoxCPM2'nin AudioVAE'siyle aynı latent uzayında çalışıyor (64 boyut, 25 Hz;
`deneyler/latent_uyumu.py` ile doğrulandı: latent korelasyonu 0,97). Bu sayede yayımlanmamış kodlayıcının yerine
VoxCPM2'ninki kullanılabiliyor ve büyük çözücü küçük olana öğretmen oluyor. Yayımlanan model 4 adıma damıtılmış
olduğu için ince ayar sonrası yönlendirmeli (CFG), çok adımlı Euler örnekleyicisi (`deneyler/akustik/ornekle.py`)
kullanılıyor; robotiklik ve titrek perde ancak bununla ve ~10 saatlik veriyle gideriliyor.

## Dizin yapısı

```
app.py, api.py              Streamlit uygulaması, API
ses_kaydi.py                seslerin ortak listesi (uygulama + API)
belge.py, ses_dosyasi.py    belge okuyucular, uzun ses yazımı (MP3 akışı, 4 GB'lık WAV sınırına karşı parçalama)
servis/                     macOS servisleri, Hugging Face Space üretimi ve şablonları
deneyler/                   eğitim hattı; agirliklar/ ve akustik/agirliklar/ ince ayarlı sesler
sesler/                     VoxCPM2 ile ses tasarımı ve tam ses hattı
```

## Sorumlu kullanım

Üretilen sesler yapay zekâ ile üretilmiştir. Yayımlarken ya da bir görüşmede kullanırken dinleyicilere bunu belirtin.
Sesleri gerçek bir kişinin sesiymiş gibi sunmayın; dolandırıcılıkta ya da başkasının yerine geçmek için kullanmayın.
Birinin sesini klonlayacaksanız onun açık izni olmalı.

## Lisans ve teşekkür

Apache License 2.0 (bkz. `LICENSE` ve `NOTICE`). İnce ayarlı ağırlıklar EMA Lightning'in ağırlıklarından
türetilmiştir.

- [EMA Lightning](https://huggingface.co/canberkkkkkk/ema-lightning), Canberk Aslan, Apache 2.0
- [normalizer-tr](https://github.com/erdemtuna/normalizer-tr), Erdem Tuna, Apache 2.0
- [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2), OpenBMB, Apache 2.0
- [Mozilla Common Voice](https://github.com/common-voice/common-voice) Türkçe cümle derlemi, CC0

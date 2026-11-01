# Aday sesler: dinleme listesi (2. üretim)

32 aday VoxCPM2'nin ses tasarımıyla üretildi; hiçbiri gerçek bir kişiye ait değil. Hepsi aynı metni okuyor.

**İlk üretimdeki hata düzeltildi.** VoxCPM2'nin ayrıştırıcısı şapkalı harfleri ("â", "î", "û") nadir, ayrı parçalar
olarak görüyor ve kelimeyi yutuyordu: "hikâye" -> "hika". Artık bu harfler okunmadan önce düz harfe çevriliyor.
Whisper'ın kelime zamanlarına göre "hikaye" artık bütün adaylarda normal uzunlukta (0,34–0,50 sn).

**Perde rehberi.** Yetişkin kadın ~165–255 Hz, yetişkin erkek ~85–155 Hz, 7–10 yaş çocuk ~220–300 Hz.
⚠ işaretliler tarifine uymayan perdede çıktı. Perde ölçüsü kaba bir tahmindir; karar kulağın.

| Ses | Tarif | Deneme 1 | Deneme 2 |
|---|---|---|---|
| **Kadın** | 1 · otuzlu yaşlar, sıcak, yumuşak, sakin anlatıcı | `kadin_t1_d1` 246 Hz | `kadin_t1_d2` 205 Hz |
| | 2 · genç, parlak, samimi, net | `kadin_t2_d1` 262 Hz | `kadin_t2_d2` 276 Hz (ince) |
| | 3 · orta yaş, yatıştırıcı, hafif kalın, sesli kitap | `kadin_t3_d1` 205 Hz | `kadin_t3_d2` 205 Hz |
| | 4 · yirmili yaşlar, akıcı, zarif, haber sunucusu | `kadin_t4_d1` 182 Hz | `kadin_t4_d2` 225 Hz |
| **Erkek** | 1 · otuzlu yaşlar, tok bariton, güven veren | `erkek_t1_d1` 125 Hz | `erkek_t1_d2` 124 Hz |
| | 2 · orta yaş, dolgun, sesli kitap | `erkek_t2_d1` ⚠ 216 Hz | `erkek_t2_d2` ⚠ 186 Hz |
| | 3 · genç, samimi, enerjik | `erkek_t3_d1` 131 Hz | `erkek_t3_d2` 125 Hz |
| | 4 · kırklı yaşlar, yumuşak, güven verici | `erkek_t4_d1` 122 Hz | `erkek_t4_d2` 122 Hz |
| **Kız çocuk** | 1 · 8 yaş, neşeli, tatlı | `kiz_cocuk_t1_d1` 308 Hz | `kiz_cocuk_t1_d2` 276 Hz |
| | 2 · 10 yaş, net, meraklı | `kiz_cocuk_t2_d1` 291 Hz | `kiz_cocuk_t2_d2` 308 Hz |
| | 3 · ~7 yaş, yumuşak, masum | `kiz_cocuk_t3_d1` 254 Hz | `kiz_cocuk_t3_d2` 286 Hz |
| | 4 · 9 yaş, oyuncu, anlatımlı | `kiz_cocuk_t4_d1` 276 Hz | `kiz_cocuk_t4_d2` 333 Hz |
| **Erkek çocuk** | 1 · 8 yaş, meraklı, enerjik | `erkek_cocuk_t1_d1` 242 Hz | `erkek_cocuk_t1_d2` ⚠ 174 Hz |
| | 2 · 10 yaş, net, samimi | `erkek_cocuk_t2_d1` 276 Hz | `erkek_cocuk_t2_d2` ⚠ 195 Hz |
| | 3 · ~7 yaş, oyuncu, heyecanlı | `erkek_cocuk_t3_d1` ⚠ 144 Hz | `erkek_cocuk_t3_d2` ⚠ 150 Hz |
| | 4 · 9 yaş, parlak, kendinden emin | `erkek_cocuk_t4_d1` ⚠ 150 Hz | `erkek_cocuk_t4_d2` ⚠ 162 Hz |

Dosyalar `adaylar/` klasöründe (`.wav`). Ölçümler `adaylar/olcumler.tsv`, tarifler ve tohumlar `adaylar/adaylar.tsv` içinde.

## Seçilen sesler

`kadin_t1_d1`, `erkek_t1_d2`, `kiz_cocuk_t1_d1` -> `secilen/` klasöründe referans olarak sabitlendi.
Klonlamayla uzun bir deneme metni (soru, ünlem, saat, tarih, uzun cümle) ikişer kez okutuldu:
`secilen/<ses>_deneme1.wav` ve `_deneme2.wav`. Perde referansla aynı kaldı (±10 Hz).

## Erkek çocuk: 2. tur (perdeyi ve yaşı vurgulayan tarifler)

Hepsi çocuk aralığında (219–320 Hz). Dosyalar `adaylar/erkek_cocuk2_t<tarif>_d<deneme>.wav`.

| Tarif | d1 | d2 | d3 |
|---|---|---|---|
| 1 · 6 yaş, ince ve tiz, sesi henüz kalınlaşmamış | 276 Hz | 235 Hz | 267 Hz |
| 2 · 7 yaş, hafif ve ince, neşeli, meraklı | 267 Hz | 267 Hz | 276 Hz |
| 3 · ~6 yaş, ince tiz, oyuncu ve tatlı | 219 Hz | 250 Hz | 254 Hz |
| 4 · 8 yaş, net ve ince, enerjik | 302 Hz | 291 Hz | 232 Hz |
| 5 · ~7 yaş, yumuşak ince, masum ve heyecanlı | 291 Hz | 308 Hz | 320 Hz |
| 6 · 6 yaş, kıkırdayan, canlı, çizgi film tadında ama doğal | 262 Hz | 246 Hz | 250 Hz |

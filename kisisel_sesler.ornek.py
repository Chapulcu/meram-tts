"""Bu bilgisayara özel (kişisel) sesler için örnek dosya.

Kullanmak için bu dosyayı kisisel_sesler.py adıyla kopyala ve satırları doldur. kisisel_sesler.py .gitignore'da
olduğu için depoya girmez; Hugging Face Space'e de kopyalanmaz (servis/space_hazirla.sh). Kendi sesini
eğitmek için README'deki "Kendi sesini eklemek" bölümüne bak.

Her satır: (kimlik, görünen ad, açıklama, dosya adı, akustik model adı). Dosyalar:
deneyler/akustik/agirliklar/ema_<akustik model adı ya da dosya adı>.pt ve deneyler/agirliklar/decoder_<dosya adı>.pt;
dosya adı ve akustik model adı boşsa (None) kimlik kullanılır.
"""

ONDE = [   # listenin başında: ilki varsayılan ses olur
    # ("benim-sesim", "Benim sesim", "Kişisel ses; yaklaşık 8 saatlik veriyle ince ayarlandı", "benim_sesim", None),
]
SONDA = [  # listenin sonunda
]

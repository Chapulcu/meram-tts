"""Belgelerden seslendirilecek metni çıkarır: bölüm başlığı ve paragraflar.

Desteklenen biçimler:
  .txt .md          karakter kodlaması kendiliğinden tanınır (UTF-8, eski Türkçe dosyalar için Windows-1254)
  .epub             kitabın kendi okuma sırasıyla (spine), bölüm bölüm; kopya korumalı (DRM) dosyalar okunmaz
  .docx .odt        Word'ün ve LibreOffice'in başlık stilleri bölüm sınırı olur
  .rtf              striprtf ile; Türkçe kod sayfası (\\ansicpg1254) tanınır
  .doc              eski Word: Linux'ta antiword, macOS'te yerleşik textutil ile
  .html .htm        başlık etiketleri (h1–h3) bölüm sınırı olur
  .pdf              yalnızca metin katmanı olan PDF'ler; her sayfa bir bölüm, satır sonu tireleri birleştirilir,
                    tekrar eden üst ve alt bilgiler ile sayfa numaraları atılır

    from belge import oku
    bolumler = oku("kitap.epub", Path("kitap.epub").read_bytes())
    bolumler[0].baslik, bolumler[0].paragraflar
"""

import posixpath
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path

BICIMLER = ["txt", "md", "epub", "pdf", "docx", "doc", "rtf", "odt", "html", "htm"]
EN_KISA_BOLUM = 40          # bundan kısa bölümler (kapak, boş sayfa) atlanır
TXT_KISIM = 12_000          # başlıksız uzun düz metin bu kadar karakterlik kısımlara bölünür


class BelgeHatasi(Exception):
    """Kullanıcıya olduğu gibi gösterilecek, anlaşılır bir okuma hatası."""


@dataclass
class Bolum:
    baslik: str
    paragraflar: list = field(default_factory=list)

    @property
    def metin(self):
        return "\n\n".join(self.paragraflar)


def oku(ad, veri):
    uzanti = Path(ad).suffix.lower().lstrip(".")
    okuyucu = {"txt": _txt, "md": _markdown, "epub": _epub, "pdf": _pdf, "docx": _docx, "odt": _odt, "rtf": _rtf,
               "doc": _doc, "html": _html_belge, "htm": _html_belge}.get(uzanti)
    if okuyucu is None:
        raise BelgeHatasi(f".{uzanti} biçimi desteklenmiyor. Desteklenenler: {', '.join(BICIMLER)}")
    bolumler = [b for b in okuyucu(veri, uzanti) if len(b.metin) >= EN_KISA_BOLUM or uzanti == "pdf"]
    bolumler = [b for b in bolumler if b.paragraflar]
    if not bolumler:
        raise BelgeHatasi("Belgede okunabilir metin bulunamadı.")
    return bolumler


# ---------- düz metin ----------

def coz(veri):
    """Baytları metne: önce UTF-8, olmazsa eski Türkçe Windows kodlaması."""
    for kodlama in ("utf-8-sig", "cp1254"):
        try:
            return veri.decode(kodlama)
        except UnicodeDecodeError:
            continue
    return veri.decode("latin-1")


# "Bölüm 3", "BÖLÜM ÜÇ: Yolculuk", "BİRİNCİ BÖLÜM" gibi kısa satırlar
BASLIK_SATIRI = re.compile(r"^\s*((BÖLÜM|Bölüm|KISIM|Kısım)\s+\S+.{0,60}|\S+\s+(BÖLÜM|Bölüm|KISIM|Kısım))\s*$")


def _paragraflara(metin):
    return [" ".join(p.split()) for p in re.split(r"\n\s*\n", metin) if p.strip()]


def _bolumle(paragraflar, baslik_mi, varsayilan="Metin"):
    """Paragrafları başlık satırlarından bölümlere ayırır; başlık yoksa uzun metni kısımlara böler."""
    bolumler, simdiki = [], Bolum(varsayilan)
    for p in paragraflar:
        if baslik_mi(p):
            if simdiki.paragraflar:
                bolumler.append(simdiki)
            simdiki = Bolum(p.strip("# ").strip(), [p.strip("# ").strip()])  # başlık da okunur
        else:
            simdiki.paragraflar.append(p)
    if simdiki.paragraflar:
        bolumler.append(simdiki)
    if len(bolumler) == 1 and len(bolumler[0].metin) > TXT_KISIM:
        bolumler = _kisimlara(bolumler[0].paragraflar)
    return bolumler


def _kisimlara(paragraflar):
    kisimlar, simdiki, boy = [], [], 0
    for p in paragraflar:
        if simdiki and boy + len(p) > TXT_KISIM:
            kisimlar.append(simdiki)
            simdiki, boy = [], 0
        simdiki.append(p)
        boy += len(p)
    if simdiki:
        kisimlar.append(simdiki)
    return [Bolum(f"Kısım {i}", k) for i, k in enumerate(kisimlar, 1)]


def _txt(veri, _):
    return _bolumle(_paragraflara(coz(veri)), lambda p: len(p) < 80 and bool(BASLIK_SATIRI.match(p)))


def _markdown(veri, _):
    metin = coz(veri)
    metin = re.sub(r"```.*?```", "", metin, flags=re.S)                    # kod blokları okunmaz
    metin = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", metin)                      # görseller
    metin = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", metin)                  # bağlantılar: yalnızca yazısı
    metin = re.sub(r"^\s*(#{1,6})\s*(.+)$", lambda m: f"\n\n{m[1]} {m[2]}\n\n", metin, flags=re.M)
    metin = re.sub(r"^\s*([-*+]|\d+\.)\s+", "", metin, flags=re.M)          # liste işaretleri
    metin = re.sub(r"(\*\*|__|\*|_|`)", "", metin)
    return _bolumle(_paragraflara(metin), lambda p: bool(re.match(r"^#{1,3} ", p)))


# ---------- HTML ve EPUB ----------

BLOK = {"p", "div", "li", "br", "tr", "blockquote", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6", "dd", "dt"}
ATLA = {"script", "style", "head", "nav", "rt", "sup"}   # sup: dipnot numaraları


class _HtmlMetin(HTMLParser):
    """HTML'den (etiket, metin) blokları: başlıklar ayrı, diğerleri paragraf."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.bloklar, self.parca, self.etiket, self.atla = [], [], "p", 0

    def _bitir(self):
        metin = " ".join("".join(self.parca).split())
        if metin:
            self.bloklar.append((self.etiket, metin))
        self.parca = []

    def handle_starttag(self, tag, attrs):
        if tag in ATLA:
            self.atla += 1
        elif tag in BLOK:
            self._bitir()
            self.etiket = tag if tag in ("h1", "h2", "h3") else "p"

    def handle_endtag(self, tag):
        if tag in ATLA:
            self.atla = max(0, self.atla - 1)
        elif tag in BLOK:
            self._bitir()
            self.etiket = "p"

    def handle_data(self, data):
        if not self.atla:
            self.parca.append(data)

    def close(self):
        super().close()
        self._bitir()
        return self.bloklar


def html_bloklari(html):
    p = _HtmlMetin()
    p.feed(html)
    return p.close()


def _html_belge(veri, _):
    bolumler, simdiki = [], Bolum("Metin")
    for etiket, metin in html_bloklari(coz(veri)):
        if etiket in ("h1", "h2"):
            if simdiki.paragraflar:
                bolumler.append(simdiki)
            simdiki = Bolum(metin, [metin])
        else:
            simdiki.paragraflar.append(metin)
    if simdiki.paragraflar:
        bolumler.append(simdiki)
    return bolumler


def _yerel(etiket):
    return etiket.rsplit("}", 1)[-1]


def _epub(veri, _):
    try:
        z = zipfile.ZipFile(BytesIO(veri))
    except zipfile.BadZipFile as hata:
        raise BelgeHatasi("Bu dosya geçerli bir EPUB değil.") from hata
    adlar = set(z.namelist())
    if "META-INF/encryption.xml" in adlar:
        sifreli = ET.fromstring(z.read("META-INF/encryption.xml"))
        # Yalnızca yazı tipi gizleme (font obfuscation) varsa kitap okunabilir; içerik şifreliyse DRM'dir
        if any(_yerel(e.tag) == "CipherReference" and not re.search(r"\.(ttf|otf|woff2?)$", e.get("URI", ""), re.I)
               for e in sifreli.iter()):
            raise BelgeHatasi("Bu e-kitap kopya korumalı (DRM). Korumalı kitaplar okunamaz; "
                              "korumasız bir EPUB kullanın.")
    try:
        kap = ET.fromstring(z.read("META-INF/container.xml"))
        opf_yolu = next(e.get("full-path") for e in kap.iter() if _yerel(e.tag) == "rootfile")
        opf = ET.fromstring(z.read(opf_yolu))
    except (KeyError, StopIteration, ET.ParseError) as hata:
        raise BelgeHatasi("EPUB'ın içindekiler dosyası okunamadı.") from hata
    kok = posixpath.dirname(opf_yolu)
    manifest = {e.get("id"): e for e in opf.iter() if _yerel(e.tag) == "item"}
    sira = [e.get("idref") for e in opf.iter() if _yerel(e.tag) == "itemref"]

    bolumler = []
    for idref in sira:
        oge = manifest.get(idref)
        if oge is None or "nav" in (oge.get("properties") or ""):
            continue
        yol = posixpath.normpath(posixpath.join(kok, oge.get("href", "")))
        if yol not in adlar:
            continue
        bloklar = html_bloklari(coz(z.read(yol)))
        paragraflar = [m for _, m in bloklar]
        baslik = next((m for e, m in bloklar if e in ("h1", "h2", "h3")), None)
        if paragraflar:
            bolumler.append(Bolum(baslik or f"Bölüm {len(bolumler) + 1}", paragraflar))
    return bolumler


# ---------- Word ve textutil ----------

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx(veri, _):
    try:
        z = zipfile.ZipFile(BytesIO(veri))
        belge = ET.fromstring(z.read("word/document.xml"))
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as hata:
        raise BelgeHatasi("Bu dosya geçerli bir Word (.docx) belgesi değil.") from hata
    # Stil kimlikleri dile göre değişir ("Heading1", Türkçe Word'de "Balk1"); iç adları ise hep İngilizce
    stil_adi = {}
    if "word/styles.xml" in z.namelist():
        for s in ET.fromstring(z.read("word/styles.xml")).iter(W + "style"):
            ad = s.find(W + "name")
            stil_adi[s.get(W + "styleId")] = (ad.get(W + "val") if ad is not None else "").lower()

    bolumler, simdiki = [], Bolum("Metin")
    for p in belge.iter(W + "p"):
        metin = " ".join("".join(t.text or "" for t in p.iter(W + "t")).split())
        if not metin:
            continue
        stil = p.find(f"{W}pPr/{W}pStyle")
        ad = stil_adi.get(stil.get(W + "val"), "") if stil is not None else ""
        if ad in ("title", "heading 1", "heading 2"):
            if simdiki.paragraflar:
                bolumler.append(simdiki)
            simdiki = Bolum(metin, [metin])
        else:
            simdiki.paragraflar.append(metin)
    if simdiki.paragraflar:
        bolumler.append(simdiki)
    if len(bolumler) == 1 and len(bolumler[0].metin) > TXT_KISIM:
        bolumler = _kisimlara(bolumler[0].paragraflar)
    return bolumler


def _satirlardan(metin):
    """Her satırı paragraf sayan çeviricilerin (antiword, textutil, striprtf) çıktısı -> bölümler."""
    return _bolumle([" ".join(s.split()) for s in metin.splitlines() if s.strip()],
                    lambda p: len(p) < 80 and bool(BASLIK_SATIRI.match(p)))


T = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"


def _odt_metni(oge):
    """ODT paragrafının metni: <text:s/> boşluk, <text:tab/> ve <text:line-break/> ayraç sayılır."""
    parcalar = [oge.text or ""]
    for alt in oge:
        etiket = _yerel(alt.tag)
        if etiket == "s":
            parcalar.append(" " * int(alt.get(T + "c", "1")))
        elif etiket in ("tab", "line-break"):
            parcalar.append(" ")
        elif etiket != "note":                      # dipnotlar okunmaz
            parcalar.append(_odt_metni(alt))
        parcalar.append(alt.tail or "")
    return "".join(parcalar)


def _odt(veri, _):
    """OpenDocument (LibreOffice): içerik zip içindeki content.xml'de; 1. ve 2. düzey başlıklar bölüm sınırı."""
    try:
        icerik = ET.fromstring(zipfile.ZipFile(BytesIO(veri)).read("content.xml"))
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as hata:
        raise BelgeHatasi("Bu dosya geçerli bir OpenDocument (.odt) belgesi değil.") from hata
    bolumler, simdiki = [], Bolum("Metin")
    for oge in icerik.iter():
        etiket = _yerel(oge.tag)
        if etiket not in ("p", "h") or not oge.tag.startswith(T):
            continue
        metin = " ".join(_odt_metni(oge).split())
        if not metin:
            continue
        if etiket == "h" and oge.get(T + "outline-level", "1") in ("1", "2"):
            if simdiki.paragraflar:
                bolumler.append(simdiki)
            simdiki = Bolum(metin, [metin])
        else:
            simdiki.paragraflar.append(metin)
    if simdiki.paragraflar:
        bolumler.append(simdiki)
    if len(bolumler) == 1 and len(bolumler[0].metin) > TXT_KISIM:
        bolumler = _kisimlara(bolumler[0].paragraflar)
    return bolumler


def _rtf(veri, _):
    from striprtf.striprtf import rtf_to_text

    ham = veri.decode("latin-1")    # RTF 7 bitlik; Türkçe harfler \'fe ya da \u351 kaçışlarıyla gelir
    if not ham.lstrip().startswith("{\\rtf"):
        raise BelgeHatasi("Bu dosya geçerli bir RTF belgesi değil.")
    kod = re.search(r"\\ansicpg(\d+)", ham)
    try:
        metin = rtf_to_text(ham, encoding=f"cp{kod[1]}" if kod else "cp1254", errors="replace")
    except LookupError:
        metin = rtf_to_text(ham, encoding="cp1254", errors="replace")
    return _satirlardan(metin)


def _doc(veri, uzanti):
    """Eski Word (.doc): Linux'ta antiword, macOS'te yerleşik textutil."""
    with tempfile.TemporaryDirectory() as d:
        giris = Path(d) / f"belge.{uzanti}"
        giris.write_bytes(veri)
        if shutil.which("antiword"):
            komut = ["antiword", "-w", "0", "-m", "UTF-8.txt", str(giris)]
        elif shutil.which("textutil"):
            komut = ["textutil", "-convert", "txt", "-encoding", "UTF-8", "-stdout", str(giris)]
        else:
            raise BelgeHatasi(".doc okumak için antiword gerekli (Linux: apt install antiword). "
                              "Belgeyi Word'de .docx olarak kaydedip yükleyebilirsin.")
        sonuc = subprocess.run(komut, capture_output=True, timeout=120)
    if sonuc.returncode != 0:
        raise BelgeHatasi(f".doc dosyası okunamadı ({komut[0]}: {sonuc.stderr.decode(errors='ignore').strip()}). "
                          "Belgeyi Word'de .docx olarak kaydedip yükleyebilirsin.")
    return _satirlardan(coz(sonuc.stdout))


# ---------- PDF ----------

def _pdf(veri, _):
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        okuyucu = PdfReader(BytesIO(veri))
        if okuyucu.is_encrypted and not okuyucu.decrypt(""):
            raise BelgeHatasi("PDF parolalı; parolasız bir kopya kullanın.")
        sayfalar = [(s.extract_text() or "") for s in okuyucu.pages]
    except PdfReadError as hata:
        raise BelgeHatasi("PDF okunamadı; dosya bozuk olabilir.") from hata
    if sum(len(s.strip()) for s in sayfalar) < 20 * max(len(sayfalar), 1) * 0.1:
        raise BelgeHatasi("Bu PDF'te metin katmanı yok gibi görünüyor (taranmış ya da görsel olarak kaydedilmiş). "
                          "Bu tür PDF'ler için yazı tanıma (OCR) gerekir; henüz desteklenmiyor.")

    satirlar = [[s.strip() for s in sayfa.splitlines()] for sayfa in sayfalar]
    # Üst ve alt bilgi: sayfaların çoğunda ilk ya da son satırlarda aynen tekrar eden satırlar
    kenar = Counter()
    for ss in satirlar:
        dolu = [s for s in ss if s]
        kenar.update(set(dolu[:2] + dolu[-2:]))
    tekrar = {s for s, n in kenar.items() if len(satirlar) >= 4 and n >= 0.5 * len(satirlar)}
    sayfa_no = re.compile(r"^(\d{1,4}|-\s*\d{1,4}\s*-|sayfa\s+\d+(\s*/\s*\d+)?)$", re.I)

    bolumler = []
    for i, ss in enumerate(satirlar, 1):
        temiz = [s for s in ss if s and s not in tekrar and not sayfa_no.match(s)]
        bolumler.append(Bolum(f"Sayfa {i}", _pdf_paragraflari(temiz)))
    return bolumler


def _pdf_paragraflari(satirlar):
    """Görsel satırları paragraflara: tireyle bölünen kelimeler birleşir, kısa ve noktayla biten satır paragrafı bitirir."""
    if not satirlar:
        return []
    ortalama = sum(len(s) for s in satirlar) / len(satirlar)
    paragraflar, simdiki = [], ""
    for s in satirlar:
        if simdiki.endswith("-") and s[:1].islower():
            simdiki = simdiki[:-1] + s                       # "ör-" + "nek" -> "örnek"
        else:
            simdiki = f"{simdiki} {s}".strip()
        if s[-1:] in ".!?:\"”" and len(s) < 0.8 * ortalama:
            paragraflar.append(simdiki)
            simdiki = ""
    if simdiki:
        paragraflar.append(simdiki)
    return paragraflar

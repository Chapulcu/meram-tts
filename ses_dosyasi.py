"""Uzun sesleri parça parça diske yazar; belleğe hiç toplamaz.

MP3: ham ses doğrudan ffmpeg'e akar, arada WAV yok. WAV: biçimin 4 GB sınırına yaklaşınca
"ad (2).wav" gibi yeni bir parçaya geçer (48 kHz'te bir parça ~11 saat). Kitap uzunluğunda sesler
ikisinde de sorunsuz yazılır.

    yazici = SesYazici(Path("kitap.mp3"), "MP3", 48000)
    yazici.yaz(ses)          # float32, -1..1
    dosyalar = yazici.kapat()
"""

import shutil
import subprocess
import wave

import numpy as np

FFMPEG = shutil.which("ffmpeg")
WAV_SINIR = 3_900_000_000


def pcm(audio):
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()


class SesYazici:
    def __init__(self, yol, bicim, sr, ffmpeg=FFMPEG, wav_sinir=WAV_SINIR):
        if bicim == "MP3" and not ffmpeg:
            raise RuntimeError("MP3 için ffmpeg gerekli: brew install ffmpeg")
        self.yol, self.bicim, self.sr, self.ffmpeg, self.wav_sinir = yol, bicim, sr, ffmpeg, wav_sinir
        self.dosyalar, self.surec, self.wav, self.yazilan = [], None, None, 0
        self._ac()

    def _ac(self):
        n = len(self.dosyalar) + 1
        yol = self.yol if n == 1 else self.yol.with_name(f"{self.yol.stem} ({n}){self.yol.suffix}")
        self.dosyalar.append(yol)
        if self.bicim == "MP3":
            self.surec = subprocess.Popen([self.ffmpeg, "-y", "-loglevel", "error", "-f", "s16le", "-ar", str(self.sr),
                                           "-ac", "1", "-i", "-", "-b:a", "64k", str(yol)], stdin=subprocess.PIPE)
        else:
            self.wav = wave.open(str(yol), "wb")
            self.wav.setnchannels(1)
            self.wav.setsampwidth(2)
            self.wav.setframerate(self.sr)
            self.yazilan = 0

    def yaz(self, audio):
        veri = pcm(audio)
        if self.surec:
            self.surec.stdin.write(veri)
            return
        if self.yazilan and self.yazilan + len(veri) > self.wav_sinir:
            self.wav.close()
            self._ac()
        self.wav.writeframes(veri)
        self.yazilan += len(veri)

    def kapat(self):
        if self.surec:
            self.surec.stdin.close()
            if self.surec.wait() != 0:
                raise RuntimeError("ffmpeg sesi MP3'e çeviremedi")
        else:
            self.wav.close()
        return self.dosyalar

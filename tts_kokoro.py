"""TTS Kokoro a parte, pensato per essere agganciato allo streaming dell'LLM.

Integrazione in app_fast (3 punti):

    from tts_kokoro import KokoroTTS
    tts = KokoroTTS()              # 1) una volta all'avvio (si carica in background)

    tts.stop()                     # 2) prima di una nuova domanda (zittisce la risposta precedente)
    ...
    tts.feed(delta)                # 3) dentro il ciclo dello stream, per ogni pezzetto di testo
    ...
    tts.flush()                    #    a fine stream, per leggere l'ultimo pezzo rimasto

Come funziona:
- feed() accumula i pezzetti in un buffer e appena c'è una frase (o una
  proposizione) completa la manda a Kokoro, senza aspettare la fine della risposta.
- Il PRIMO pezzo di ogni risposta viene tagliato prima (anche alla virgola) per
  iniziare a parlare il prima possibile; i successivi sono frasi più naturali.
- Un thread sintetizza la frase dopo mentre un altro riproduce quella corrente.
"""

import queue
import re
import threading

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 24000

_FINE_FRASE = ".!?…\n"
_PAUSA = ",;:"

_RE_EMOJI = re.compile("[\U00010000-\U0010ffff\u2600-\u27bf]")
_RE_MARKDOWN = re.compile(r"[*_#`~>|]")
_RE_SPAZI = re.compile(r"\s+")


def pulisci_testo(testo):
    testo = _RE_EMOJI.sub("", testo)
    testo = _RE_MARKDOWN.sub("", testo)
    return _RE_SPAZI.sub(" ", testo).strip()


class KokoroTTS:
    def __init__(
        self,
        lang_code="i",
        voice="im_nicola",   # "im_nicola" per la voce maschile
        speed=1.0,
        device=None,       # None = automatico, "cpu" per lasciare la GPU all'LLM
        primo_pezzo=8,     # caratteri minimi per il primo pezzo (partenza veloce)
        min_frase=20,      # caratteri minimi per le frasi successive
        min_pausa=40,      # caratteri minimi per tagliare a virgola/due punti dopo il primo pezzo
    ):
        self.lang_code = lang_code
        self.voice = voice
        self.speed = speed
        self.device = device
        self.primo_pezzo = primo_pezzo
        self.min_frase = min_frase
        self.min_pausa = min_pausa

        self.enabled = True
        self._buf = ""
        self._primo = True
        self._generation = 0
        self._text_q = queue.Queue()
        self._audio_q = queue.Queue(maxsize=3)
        self._closed = threading.Event()

        threading.Thread(target=self._synth_loop, daemon=True).start()
        threading.Thread(target=self._play_loop, daemon=True).start()

    # ---------- API pubblica ----------

    def feed(self, delta):
        """Da chiamare per ogni pezzetto di testo che arriva dallo stream."""
        if not self.enabled or not delta:
            return
        self._buf += delta
        while True:
            taglio = self._trova_taglio()
            if taglio is None:
                return
            pezzo, self._buf = self._buf[:taglio], self._buf[taglio:]
            self._emetti(pezzo)

    def flush(self):
        """Da chiamare a fine stream: legge ciò che è rimasto nel buffer."""
        if self.enabled and self._buf.strip():
            self._emetti(self._buf)
        self._buf = ""
        self._primo = True

    def say(self, testo):
        """Legge subito un testo intero (senza passare dal buffer)."""
        if not self.enabled:
            return
        testo = pulisci_testo(testo)
        if any(c.isalnum() for c in testo):
            self._text_q.put((self._generation, testo))

    def stop(self):
        """Interrompe l'audio in corso e svuota buffer e code."""
        self._generation += 1
        self._buf = ""
        self._primo = True
        for coda in (self._text_q, self._audio_q):
            try:
                while True:
                    coda.get_nowait()
            except queue.Empty:
                pass
        sd.stop()

    def close(self):
        self._closed.set()
        self.stop()

    # ---------- buffer ----------

    def _trova_taglio(self):
        buf = self._buf
        min_fine = self.primo_pezzo if self._primo else self.min_frase
        min_pausa = self.min_frase if self._primo else self.min_pausa

        for i, ch in enumerate(buf):
            if ch not in _FINE_FRASE and ch not in _PAUSA:
                continue
            # Taglio solo se il carattere dopo è uno spazio (così "3.5" o "3,5" non si spezzano);
            # l'a capo invece taglia sempre.
            dopo_ok = ch == "\n" or (i + 1 < len(buf) and buf[i + 1].isspace())
            if not dopo_ok:
                continue
            soglia = min_fine if ch in _FINE_FRASE else min_pausa
            if i + 1 >= soglia:
                return i + 1
        return None

    def _emetti(self, pezzo):
        self._primo = False
        self.say(pezzo)

    # ---------- thread interni ----------

    def _synth_loop(self):
        try:
            from kokoro import KPipeline

            if self.device:
                pipeline = KPipeline(lang_code=self.lang_code, device=self.device)
            else:
                pipeline = KPipeline(lang_code=self.lang_code)
        except Exception as exc:
            print(f"[TTS]: impossibile caricare Kokoro ({exc}). Voce disattivata.")
            self.enabled = False
            return

        print("[TTS]: Kokoro pronto.")
        while not self._closed.is_set():
            try:
                gen, testo = self._text_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if gen != self._generation:
                continue
            try:
                for _, _, audio in pipeline(testo, voice=self.voice, speed=self.speed):
                    if gen != self._generation:
                        break
                    if hasattr(audio, "detach"):
                        audio = audio.detach().cpu().numpy()
                    self._audio_q.put((gen, np.asarray(audio, dtype=np.float32)))
            except Exception as exc:
                print(f"[TTS]: errore di sintesi: {exc}")

    def _play_loop(self):
        while not self._closed.is_set():
            try:
                gen, audio = self._audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if gen != self._generation:
                continue
            try:
                sd.play(audio, SAMPLE_RATE)
                sd.wait()
            except Exception as exc:
                print(f"[TTS]: errore di riproduzione: {exc}")

import base64
import json
import threading
import time
from functools import lru_cache

import cv2
import requests

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

from tts_kokoro import KokoroTTS

# 127.0.0.1 invece di localhost: su Windows evita il tentativo iniziale su IPv6
SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
IMAGE_SIZE = (336, 252)
JPEG_QUALITY = 70
REQUEST_TIMEOUT = 60
MAX_TURNS = 5

# Voce (Kokoro): "if_sara" (femminile) oppure "im_nicola" (maschile)
ENABLE_TTS = True
TTS_VOICE = "if_sara"
TTS_SPEED = 1.0
TTS_DEVICE = None  # None = automatico; "cpu" per lasciare tutta la GPU al modello LLM

stop_event = threading.Event()
frame_lock = threading.Lock()
current_frame = None

session = requests.Session()
# Ignora la ricerca automatica dei proxy di sistema (su Windows rallenta le richieste)
session.trust_env = False

SYSTEM_PROMPT = (
    "Sei un assistente virtuale amichevole e alla mano, parli in modo informale "
    "usando il 'tu'. Rispondi in italiano. Usa le immagini e le info dal web "
    "solo quando presenti. Non inventare dati non presenti. Sii conciso: "
    "massimo 2-3 frasi, salvo richieste esplicite."
)

VISION_KEYWORDS = (
    "vedi", "guarda", "fotocamera", "webcam", "colore",
    "faccia", "mano", "schermo", "davanti",
)

WEB_KEYWORDS = (
    "chi è", "chi e", "cos'è", "cos e", "che cosa è", "notizie", "news",
    "meteo", "tempo", "prezzo", "quanto costa", "quanto fa", "risultati",
    "classifica", "punteggio", "oggi", "attuale", "attuali", "ultimo",
    "ultima", "recente", "recenti", "latest", "data di", "che ore",
    "quando è", "quando e",
)


def webcam_loop():
    global current_frame
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Errore: impossibile aprire la webcam.")
        stop_event.set()
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                current_frame = frame
            cv2.imshow("Mente IA - Buddy (fast)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()


@lru_cache(maxsize=32)
def web_search(query):
    try:
        print("[IA]: cerco su internet...")
        with DDGS(timeout=8) as ddgs:
            results = [
                r.get("body", "")[:300] for r in ddgs.text(query, max_results=3)
            ]
        results = [r.strip() for r in results if r and r.strip()]
        if results:
            return "Informazioni dal web:\n" + "\n".join(results)
    except Exception as exc:
        print(f"[Errore ricerca web]: {exc}")
    return ""


def needs_search(prompt_lower):
    return any(keyword in prompt_lower for keyword in WEB_KEYWORDS)


def needs_vision(prompt_lower):
    return any(keyword in prompt_lower for keyword in VISION_KEYWORDS)


def trim_history(history):
    limit = 1 + MAX_TURNS * 2
    if len(history) <= limit:
        return history
    return [history[0]] + history[-(MAX_TURNS * 2):]


def stream_answer(messages, tts=None):
    payload = {
        "model": "qwen-vl",
        "messages": messages,
        "max_tokens": 95,
        "temperature": 0.3,
        "stream": True,
        # il server riusa la parte di conversazione già elaborata
        "cache_prompt": True,
    }
    t0 = time.time()
    response = session.post(
        SERVER_URL, json=payload, stream=True, timeout=REQUEST_TIMEOUT
    )
    if response.status_code != 200:
        print(f"\nErrore dal server: {response.status_code} - {response.text}\n")
        return ""

    print("\n[Risposta IA]:\n", end="", flush=True)
    full_text = ""
    sentence = ""  # frase in costruzione, da mandare al TTS appena è completa
    token_count = 0
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if not decoded.startswith("data: "):
            continue
        data_str = decoded[6:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
            delta = (
                chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            )
        except (json.JSONDecodeError, ValueError, IndexError):
            continue
        if not delta:
            continue

        # Una frase è finita se termina con . ! ? e il pezzo successivo inizia con
        # uno spazio/a capo (così "3.5" o "es." a metà non vengono spezzati)
        if tts and sentence and delta[0].isspace():
            if sentence.rstrip()[-1:] in (".", "!", "?") or "\n" in delta:
                tts.say(sentence)
                sentence = ""

        full_text += delta
        sentence += delta
        token_count += 1
        print(delta, end="", flush=True)

    if tts and sentence.strip():
        tts.say(sentence)

    t1 = time.time()
    if token_count == 0 and not full_text:
        print("(nessun token ricevuto)")
    print(f"\n\nLatenza: {(t1 - t0):.2f}s | token: {token_count}\n" + "-" * 60)
    return full_text


def encode_frame(frame):
    # Nessun lock: il frame è già una copia locale, così la webcam non resta bloccata
    if frame.shape[1] != IMAGE_SIZE[0] or frame.shape[0] != IMAGE_SIZE[1]:
        frame = cv2.resize(frame, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )
    if not ok:
        return None
    return base64.b64encode(buffer).decode("ascii")


def main():
    global current_frame
    t = threading.Thread(target=webcam_loop, daemon=True)
    t.start()
    time.sleep(0.2)

    # Il modello Kokoro si carica in background: non ritarda l'avvio
    tts = None
    if ENABLE_TTS:
        tts = KokoroTTS(voice=TTS_VOICE, speed=TTS_SPEED, device=TTS_DEVICE)

    print("==========================================================")
    print("  MENTE IA MULTIMODALE - RTX 4060 (FAST BUILD)")
    print("==========================================================")
    print("Comandi: /clear pulisce lo storico, /mute attiva/disattiva la voce,")
    print("         /exit o q esce, q nel video chiude.\n")

    messages_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        while not stop_event.is_set():
            try:
                prompt = input("\n[Tu]: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nUscita in corso...")
                break
            if not prompt:
                continue
            if prompt.lower() in ("/exit", "q"):
                break
            if prompt.lower() == "/clear":
                messages_history = [{"role": "system", "content": SYSTEM_PROMPT}]
                if tts:
                    tts.stop()
                print("Storico pulito.")
                continue
            if prompt.lower() == "/mute":
                if tts:
                    tts.enabled = not tts.enabled
                    if not tts.enabled:
                        tts.stop()
                    print("Voce attivata." if tts.enabled else "Voce disattivata.")
                else:
                    print("Il TTS non è attivo.")
                continue

            # Se l'IA sta ancora parlando, la interrompo: è partita una nuova domanda
            if tts:
                tts.stop()

            prompt_lower = prompt.lower()
            search = needs_search(prompt_lower)
            vision = needs_vision(prompt_lower)

            final_prompt = prompt
            if search:
                results = web_search(prompt)
                if results:
                    final_prompt = f"{prompt}\n\n{results}"

            user_content_full = final_prompt
            slim_text = final_prompt
            if vision:
                with frame_lock:
                    frame = current_frame
                if frame is None:
                    print("Attendendo il primo fotogramma...")
                    continue
                img_b64 = encode_frame(frame)
                if img_b64 is None:
                    print("Errore nella codifica dell'immagine.")
                    continue
                user_content_full = [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                    {"type": "text", "text": final_prompt},
                ]
                slim_text = f"[immagine webcam allegata] {final_prompt}"
                print("[IA]: analizzo la webcam...")
            else:
                print("[IA]: penso alla risposta...")

            messages_history.append({"role": "user", "content": user_content_full})

            try:
                full_output = stream_answer(messages_history, tts)
            except requests.RequestException as exc:
                print(f"\nErrore di connessione al server: {exc}\n")
                full_output = ""

            if not full_output:
                # niente risposta: non sporco lo storico con messaggi vuoti
                messages_history.pop()
                continue

            messages_history[-1] = {"role": "user", "content": slim_text}
            messages_history.append({"role": "assistant", "content": full_output})
            messages_history = trim_history(messages_history)

    finally:
        stop_event.set()
        if tts:
            tts.close()
        t.join(timeout=2)


if __name__ == "__main__":
    main()
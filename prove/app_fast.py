import base64
import json
import threading
import time
from functools import lru_cache

import cv2
import requests
from duckduckgo_search import DDGS

SERVER_URL = "http://localhost:8080/v1/chat/completions"
IMAGE_SIZE = (336, 252)
JPEG_QUALITY = 60
REQUEST_TIMEOUT = 60
MAX_TURNS = 5

stop_event = threading.Event()
frame_lock = threading.Lock()
current_frame = None

session = requests.Session()

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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])

    if not cap.isOpened():
        print("Errore: impossibile aprire la webcam.")
        stop_event.set()
        return

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
        with DDGS() as ddgs:
            results = [r.get("body", "") for r in ddgs.text(query, max_results=3)]
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


def stream_answer(messages):
    payload = {
        "model": "qwen-vl",
        "messages": messages,
        "max_tokens": 95,
        "temperature": 0.3,
        "stream": True,
    }
    response = session.post(
        SERVER_URL, json=payload, stream=True, timeout=REQUEST_TIMEOUT
    )
    if response.status_code != 200:
        print(f"\nErrore dal server: {response.status_code} - {response.text}\n")
        return ""

    print("\n[Risposta IA]:\n", end="", flush=True)
    full_text = ""
    token_count = 0
    t0 = time.time()
    buffer = ""
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
        except (json.JSONDecodeError, ValueError):
            continue
        if not delta:
            continue
        buffer += delta
        full_text += delta
        token_count += 1
        if len(buffer) >= 12 or "\n" in buffer:
            print(buffer, end="", flush=True)
            buffer = ""
    if buffer:
        print(buffer, end="", flush=True)
    t1 = time.time()
    if token_count == 0 and not full_text:
        print("(nessun token ricevuto)")
    print(f"\n\nLatenza: {(t1 - t0):.2f}s | token: {token_count}\n" + "-" * 60)
    return full_text


def encode_frame(frame):
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

    print("==========================================================")
    print("  MENTE IA MULTIMODALE - RTX 4060 (FAST BUILD)")
    print("==========================================================")
    print("Comandi: /clear pulisce lo storico, /exit o q esce, q nel video chiude.\n")

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
                print("Storico pulito.")
                continue

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

            full_output = stream_answer(messages_history)

            messages_history[-1] = {"role": "user", "content": slim_text}
            messages_history.append({"role": "assistant", "content": full_output})
            messages_history = trim_history(messages_history)

    finally:
        stop_event.set()
        t.join(timeout=2)


if __name__ == "__main__":
    main()

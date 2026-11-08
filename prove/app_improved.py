import base64
import json
import re
import threading
import time

import cv2
import requests
from duckduckgo_search import DDGS

SERVER_URL = "http://localhost:8080/v1/chat/completions"
IMAGE_SIZE = (448, 336)
JPEG_QUALITY = 70
REQUEST_TIMEOUT = 90
MAX_TURNS = 6

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

ROUTER_SYSTEM = (
    "Sei un router rapido. Analizza il messaggio e rispondi SOLO con i token "
    "necessari, uno per riga:\n"
    "- [CERCA: query] se servono informazioni aggiornate dal web\n"
    "- [VEDI] se serve guardare la webcam/foto attuale\n"
    "Se nessuno serve, rispondi solo: NO"
)

VISION_KEYWORDS = {
    "vedi", "guarda", "fotocamera", "webcam", "colore",
    "faccia", "mano", "schermo", "davanti",
}


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
            cv2.imshow("Mente IA - Buddy", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()


def web_search(query):
    try:
        print("[IA]: cerco su internet...")
        with DDGS() as ddgs:
            results = [r.get("body", "") for r in ddgs.text(query, max_results=3)]
        results = [r for r in results if r]
        if results:
            return "Informazioni dal web:\n" + "\n".join(results)
    except Exception as exc:
        print(f"[Errore ricerca web]: {exc}")
    return ""


def router_decision(prompt):
    payload = {
        "model": "qwen-vl",
        "messages": [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 40,
        "temperature": 0.0,
        "stream": False,
    }
    try:
        res = session.post(SERVER_URL, json=payload, timeout=REQUEST_TIMEOUT)
        if res.status_code != 200:
            return None, False
        text = (
            res.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        ).strip()
        query_match = re.search(r"\[CERCA:\s*(.*?)\]", text, re.IGNORECASE | re.DOTALL)
        search_query = query_match.group(1).strip() if query_match else None
        needs_vision = bool(re.search(r"\[VEDI\]", text, re.IGNORECASE))
        return search_query, needs_vision
    except Exception as exc:
        print(f"[Errore router]: {exc}")
        return None, False


def trim_history(history):
    if len(history) <= 1 + MAX_TURNS * 2:
        return history
    return [history[0]] + history[-(MAX_TURNS * 2):]


def stream_answer(messages):
    payload = {
        "model": "qwen-vl",
        "messages": messages,
        "max_tokens": 120,
        "temperature": 0.3,
        "stream": True,
    }
    response = session.post(SERVER_URL, json=payload, stream=True, timeout=REQUEST_TIMEOUT)
    if response.status_code != 200:
        print(f"\nErrore dal server: {response.status_code} - {response.text}\n")
        return ""

    print("\n[Risposta IA]:\n", end="", flush=True)
    full_text = ""
    token_count = 0
    t0 = time.time()
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
        if delta:
            print(delta, end="", flush=True)
            full_text += delta
            token_count += 1
    t1 = time.time()
    if token_count == 0 and not full_text:
        print("(nessun token ricevuto)")
    print(f"\n\nLatenza: {(t1 - t0):.2f}s | token: {token_count}\n" + "-" * 60)
    return full_text


def main():
    global current_frame
    t = threading.Thread(target=webcam_loop, daemon=True)
    t.start()
    time.sleep(0.5)

    print("==========================================================")
    print("  MENTE IA MULTIMODALE - RTX 4060 (PROVA OTTIMIZZATA)")
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

            # Router con heuristic locale come fallback per la visione
            search_query, router_vision = router_decision(prompt)
            prompt_lower = prompt.lower()
            local_vision = any(kw in prompt_lower for kw in VISION_KEYWORDS)
            needs_vision = router_vision or local_vision

            final_prompt = prompt
            if search_query:
                web_info = web_search(search_query)
                if web_info:
                    final_prompt = f"{prompt}\n\n{web_info}"

            user_content_full = None
            slim_text = final_prompt
            if needs_vision:
                with frame_lock:
                    if current_frame is None:
                        print("Attendendo il primo fotogramma...")
                        continue
                    frame = current_frame.copy()
                ok, buffer = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                )
                if not ok:
                    print("Errore nella codifica dell'immagine.")
                    continue
                img_b64 = base64.b64encode(buffer).decode("utf-8")
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
                user_content_full = final_prompt
                print("[IA]: penso alla risposta...")

            messages_history.append({"role": "user", "content": user_content_full})

            full_output = stream_answer(messages_history)

            # compatta il turno con immagine per non tenere base64 nello storico
            messages_history[-1] = {"role": "user", "content": slim_text}
            messages_history.append({"role": "assistant", "content": full_output})
            messages_history = trim_history(messages_history)

    finally:
        stop_event.set()
        t.join(timeout=2)


if __name__ == "__main__":
    main()
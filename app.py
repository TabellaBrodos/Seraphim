import base64
import json
import threading
import time
import cv2
import requests
from duckduckgo_search import DDGS

SERVER_URL = "http://localhost:8080/v1/chat/completions"
IMAGE_SIZE = (448, 336)

current_frame = None
frame_lock = threading.Lock()
running = True

session = requests.Session()


def webcam_loop():
  global current_frame, running
  cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0])
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])

  if not cap.isOpened():
    print("Errore: Impossibile aprire la webcam.")
    running = False
    return

  while running:
    ret, frame = cap.read()
    if ret:
      with frame_lock:
        current_frame = frame.copy()
      cv2.imshow("Mente IA - Buddy", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
      running = False
      break

  cap.release()
  cv2.destroyAllWindows()


# Avvia la webcam in background
t = threading.Thread(target=webcam_loop, daemon=True)
t.start()

print("==========================================================")
print("  MENTE IA MULTIMODALE - RICERCA AUTONOMA (RTX 4060)")
print("==========================================================")
print("Parla normalmente. L'IA deciderà da sola se usare il web o la webcam.\n")

messages_history = [{
    "role": "system",
    "content": (
        "Sei un assistente e aiutante virtuale versatile, con la personalità"
        " di un amico fidato, simpatico e alla mano. Parla in modo informale e"
        " naturale (usa il 'tu'). Se l'utente ti passa un'immagine o risultati"
        " dal web, usali per rispondere accuratamente senza allucinazioni. Sii"
        " conciso (massimo 2-3 frasi)."
    ),
}]

VISION_KEYWORDS = [
    "vedi",
    "ved",
    "guard",
    "fotocamera",
    "webcam",
    "questo",
    "questa",
    "qui",
    "davanti",
    "colore",
    "c'è",
    "appa",
    "schermo",
    "faccia",
    "mano",
]


def web_search(query):
  try:
    print("[IA]: Cerco su internet...")
    with DDGS() as ddgs:
      results = [r.get("body", "") for r in ddgs.text(query, max_results=3)]
      if results:
        return " Informazioni dal web:\n" + "\n".join(results)
  except Exception as e:
    print(f"[Errore ricerca web]: {e}")
  return ""


try:
  while running:
    try:
      prompt = input("\n[Tu]: ")
    except (KeyboardInterrupt, EOFError):
      print("\nUscita in corso...")
      break

    if prompt.lower().strip() == "q":
      running = False
      break
    if not prompt.strip():
      continue

    router_payload = {
        "model": "qwen-vl",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Sei un router. Valuta se la richiesta richiede informazioni"
                    " aggiornate dal web o fatti attuali che potresti non"
                    " conoscere. Se sì, rispondi SOLO con [CERCA: query di"
                    " ricerca]. Se no, rispondi SOLO con NO."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 25,
        "temperature": 0.0,
    }

    router_res = session.post(SERVER_URL, json=router_payload)
    final_prompt = prompt
    if router_res.status_code == 200:
      router_out = (
          router_res.json()
          .get("choices", [{}])[0]
          .get("message", {})
          .get("content", "")
          .strip()
      )
      if "[CERCA:" in router_out:
        try:
          start = router_out.find("[CERCA:") + 7
          end = router_out.find("]", start)
          query = (
              router_out[start:end].strip()
              if end != -1
              else router_out[start:].strip()
          )
          web_info = web_search(query)
          if web_info:
            final_prompt = f"{prompt}\n\n{web_info}"
        except Exception:
          pass

    prompt_lower = prompt.lower()
    needs_vision = any(keyword in prompt_lower for keyword in VISION_KEYWORDS)

    if needs_vision:
      with frame_lock:
        if current_frame is None:
          print("Attendendo il primo fotogramma...")
          continue
        frame_to_send = current_frame.copy()

      resized_frame = cv2.resize(frame_to_send, IMAGE_SIZE)
      _, buffer = cv2.imencode(
          ".jpg", resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75]
      )
      img_b64 = base64.b64encode(buffer).decode("utf-8")

      user_content = [
          {
              "type": "image_url",
              "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
          },
          {"type": "text", "text": final_prompt},
      ]
      print("[IA]: Analizzo te e la webcam...")
    else:
      user_content = final_prompt
      print("[IA]: Penso alla risposta...")

    messages_history.append({"role": "user", "content": user_content})

    payload = {
        "model": "qwen-vl",
        "messages": messages_history,
        "max_tokens": 90,
        "temperature": 0.3,
        "stream": True,
    }

    t0 = time.time()
    response = session.post(SERVER_URL, json=payload, stream=True)

    if response.status_code == 200:
      print("\n[Risposta IA]:\n", end="")
      full_output_text = ""

      for line in response.iter_lines():
        if line:
          decoded_line = line.decode("utf-8")
          if decoded_line.startswith("data: "):
            data_str = decoded_line[6:].strip()
            if data_str == "[DONE]":
              break
            try:
              chunk_json = json.loads(data_str)
              delta = (
                  chunk_json.get("choices", [{}])[0]
                  .get("delta", {})
                  .get("content", "")
              )
              if delta:
                print(delta, end="", flush=True)
                full_output_text += delta
            except json.JSONDecodeError:
              pass

      t1 = time.time()
      print(f"\n\nLatenza totale: {(t1 - t0):.2f} secondi")
      print("-" * 60)

      messages_history.append(
          {"role": "assistant", "content": full_output_text}
      )

      if len(messages_history) > 11:
        messages_history = [messages_history[0]] + messages_history[-10:]

    else:
      print(f"Errore dal server: {response.status_code} - {response.text}")

finally:
  running = False
  time.sleep(0.5)
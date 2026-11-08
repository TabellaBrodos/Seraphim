import cv2

print("Tentativo di apertura webcam...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
  print("Errore: Impossibile connettersi alla webcam.")
  exit()

print("Webcam aperta! Premi 'q' nella finestra video per uscire.")

while True:
  ret, frame = cap.read()
  if not ret:
    print("Errore di lettura fotogramma.")
    break

  cv2.imshow("Test Webcam", frame)

  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()
print("Test completato con successo.")
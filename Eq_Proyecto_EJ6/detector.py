import cv2
from ultralytics import YOLO

# Cargar el modelo que entrenaste
model = YOLO("best.pt")

# Abrir la cámara web (0 es la cámara por defecto)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar la detección
    results = model(frame, conf=0.5)  # conf es el umbral de confianza (50%)

    # Dibujar las cajas y etiquetas sobre el cuadro
    annotated_frame = results[0].plot()

    # Mostrar en ventana
    cv2.imshow("Mi Detector Personalizado YOLOv11", annotated_frame)

    # Presiona la tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
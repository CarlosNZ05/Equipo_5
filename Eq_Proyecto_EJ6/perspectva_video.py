import cv2
import numpy as np

puntos = []

def clicks(event, x, y, flags, param):
    global puntos
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(puntos) < 4:
            puntos.append([x, y])


cap = cv2.VideoCapture(0)

cv2.namedWindow("Camara Web")
cv2.setMouseCallback("Camara Web", clicks)

ancho, alto = 480, 300

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_mostrar = frame.copy()

    # Dibujar los puntos que el usuario va marcando
    for pt in puntos:
        cv2.circle(frame_mostrar, (pt[0], pt[1]), 5, (0, 255, 0), -1)

    # Si se han seleccionado 4 puntos, aplicar la transformación en tiempo real
    if len(puntos) == 4:
        pts1 = np.float32(puntos)
        pts2 = np.float32([
            [0, 0],
            [ancho, 0],
            [0, alto],
            [ancho, alto]
        ])
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        resultado = cv2.warpPerspective(frame, M, (ancho, alto))
        
        cv2.imshow("Video Transformado", resultado)

    cv2.imshow("Camara Web", frame_mostrar)

    key = cv2.waitKey(1) & 0xFF
    # Presiona 'n' para limpiar los puntos
    if key == ord('n'):
        puntos = []
        cv2.destroyWindow("Video Transformado")
    # Presiona ESC para salir
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np

puntos = []

def clicks(event, x, y, flags, param):
    global puntos
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(imagen_mostrar, (x, y), 5, (0, 255, 0), -1)
        puntos.append([x, y])


imagen = cv2.imread("compu.jpg") 
imagen_mostrar = imagen.copy()

cv2.namedWindow("Imagen")
cv2.setMouseCallback("Imagen", clicks)


ancho, alto = 480, 300

while True:
    cv2.imshow("Imagen", imagen_mostrar)
    

    if len(puntos) == 4:
        pts1 = np.float32(puntos)
        pts2 = np.float32([
            [0, 0],          # Superior Izquierdo
            [ancho, 0],      # Superior Derecho
            [0, alto],       # Inferior Izquierdo
            [ancho, alto]    # Inferior Derecho
        ])
        

        M = cv2.getPerspectiveTransform(pts1, pts2)
        

        resultado = cv2.warpPerspective(imagen, M, (ancho, alto))
        
        cv2.imshow("Imagen Transformada", resultado)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):
        puntos = []
        imagen_mostrar = imagen.copy()
        cv2.destroyWindow("Imagen Transformada")
    elif key == 27:
        break

cv2.destroyAllWindows()
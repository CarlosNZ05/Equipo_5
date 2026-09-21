import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image = cv2.imread('auto003.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.blur(gray, (3,3))
canny = cv2.Canny(gray, 150, 200)
canny = cv2.dilate(canny, None, iterations=1)

cnts, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print(f'Total de contornos encontrados: {len(cnts)}')

for c in cnts:
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    epsilon = 0.09 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)

    # DEBUG: imprime TODOS los candidatos con 4 lados, sin importar el área
    if len(approx) == 4:
        aspect_ratio = float(w) / h
        print(f'  candidato: area={area:.0f}, aspect_ratio={aspect_ratio:.2f}, pos=({x},{y})')

    if len(approx) == 4 and area > 500:  # umbral bajado desde 9000
        aspect_ratio = float(w) / h
        if 2.0 < aspect_ratio < 5.5:  # rango más amplio
            placaImg = gray[y:y+h, x:x+w]
            text = pytesseract.image_to_string(placaImg, config='--psm 11')
            text = text.strip()

            if text:
                print('placa=', text)
                cv2.imshow('placa', placaImg)
                cv2.moveWindow('placa', 780, 10)
                cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 3)
                cv2.putText(image, text, (x-20, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,255,0), 2)

cv2.imshow('Canny', canny)
cv2.moveWindow('Canny', 45, 500)
cv2.imshow('Image', image)
cv2.moveWindow('Image', 45, 10)
cv2.waitKey(0)
cv2.destroyAllWindows()
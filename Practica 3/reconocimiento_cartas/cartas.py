import glob
import os
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


ANCHO_CARTA, ALTO_CARTA = 300, 420
LADO_MAX_DETECCION = 1000
AREA_MIN_REL, AREA_MAX_REL = 0.004, 0.92
PROPORCION_CARTA = (1.0, 2.6)
PROPORCION_VISIBLE = (1.05, 2.20)
SATURACION_MAX_BLANCO = 80


EXTENSIONES_IMAGEN = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CONFIANZA_SUFICIENTE = 0.75
ANGULOS_TOLERANCIA = (-10, -5, 5, 10)
FRACCIONES_SEMILLA = (0.5, 0.65, 0.8)

REGIONES = [
    ((0.015, 0.20), (0.015, 0.30)),
    ((0.01, 0.27), (0.01, 0.36)),
    ((0.01, 0.15), (0.01, 0.27)),
    ((0.005, 0.25), (0.005, 0.34)),
    ((0.005, 0.30), (0.005, 0.40)),
]


ROI = ((0.01, 0.25), (0.01, 0.32))

TAM_VALOR = (50, 70)
TAM_PALO = (50, 50)

UMBRAL_VALOR = 0.45
UMBRAL_PALO = 0.45
PENALIZACION_COLOR = 0.12
SOLAPE_DUPLICADO = 0.5

METODOS = ("auto", "umbral", "blanco", "bordes")
PALOS = {
    "espadas": "Espadas",
    "corazones": "Corazones",
    "diamantes": "Diamantes",
    "treboles": "Tréboles",
}
PALOS_ROJOS = {"corazones", "diamantes"}
PALOS_NEGROS = {"espadas", "treboles"}



@dataclass
class Resultado:
    valor: str
    palo: str
    score_valor: float
    score_palo: float

    @property
    def valida(self) -> bool:
        return self.score_valor >= UMBRAL_VALOR and self.score_palo >= UMBRAL_PALO

    @property
    def nombre(self) -> str:
        return f"{self.valor} de {PALOS[self.palo]}"

    @property
    def clave(self) -> str:
        return f"{self.valor}-{self.palo}"


@dataclass
class Indice:
    img_valor: np.ndarray
    img_palo: np.ndarray
    rojo: bool
    roi: Tuple[int, int, int, int]
    binaria: np.ndarray


@dataclass
class Deteccion:
    cuad: np.ndarray
    carta: np.ndarray
    resultado: Optional[Resultado]


@dataclass
class Plantillas:
    valores: Dict[str, List[np.ndarray]]
    palos: Dict[str, List[np.ndarray]]



def sin_acentos(texto: str) -> str:
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")


def normalizar_palo(texto: str) -> str:
    return sin_acentos(texto.strip().lower())


def _longitudes_lados(cuad) -> Tuple[float, ...]:
    p = np.asarray(cuad, dtype=np.float32).reshape(4, 2)
    return tuple(float(np.linalg.norm(p[(i + 1) % 4] - p[i])) for i in range(4))


def _proporcion(cuad) -> float:
    lados = _longitudes_lados(cuad)
    a = (lados[0] + lados[2]) / 2.0
    b = (lados[1] + lados[3]) / 2.0
    return max(a, b) / max(min(a, b), 1e-6)


def ordenar_vertices(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    centro = pts.mean(axis=0)
    angulos = np.arctan2(pts[:, 1] - centro[1], pts[:, 0] - centro[0])
    pts = pts[np.argsort(angulos)]
    lados = _longitudes_lados(pts)
    if (lados[0] + lados[2]) > (lados[1] + lados[3]):
        pts = np.roll(pts, -1, axis=0)
    return np.ascontiguousarray(pts, dtype=np.float32)



def _mascara(img: np.ndarray, metodo: str) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    if metodo == "umbral":
        gris = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        _, mask = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    elif metodo == "blanco":
        hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
        _, sat, val = cv2.split(hsv)
        umbral_v, _ = cv2.threshold(val, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        mask = ((val >= 0.85 * umbral_v) & (sat <= SATURACION_MAX_BLANCO)).astype(np.uint8) * 255

    elif metodo == "bordes":
        gris = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        bordes = cv2.Canny(gris, 50, 150)
        mask = cv2.dilate(bordes, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(mask)
        if conts:
            cv2.drawContours(mask, conts, -1, 255, cv2.FILLED)
        return cv2.erode(mask, kernel, iterations=2)

    else:
        raise ValueError(f"Método '{metodo}' no válido. Opciones: {', '.join(METODOS)}")

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)



def _cuadrilatero(contorno: np.ndarray) -> Optional[np.ndarray]:
    casco = cv2.convexHull(contorno)
    area = cv2.contourArea(casco)
    if area <= 0:
        return None
    perimetro = cv2.arcLength(casco, True)
    for eps in (0.02, 0.04, 0.06):
        aprox = cv2.approxPolyDP(casco, eps * perimetro, True)
        if len(aprox) == 4 and cv2.contourArea(aprox) / area >= 0.85:
            return aprox.reshape(4, 2).astype(np.float32)
    rect = cv2.minAreaRect(casco)
    area_rect = rect[1][0] * rect[1][1]
    if area_rect > 0 and area / area_rect >= 0.85:
        return cv2.boxPoints(rect).astype(np.float32)
    return None


def _cuadrilatero_valido(contorno: np.ndarray, area_img: float) -> Optional[np.ndarray]:
    if cv2.contourArea(contorno) < AREA_MIN_REL * area_img:
        return None
    cuad = _cuadrilatero(contorno)
    if cuad is None:
        return None
    cuad = ordenar_vertices(cuad)
    if PROPORCION_CARTA[0] <= _proporcion(cuad) <= PROPORCION_CARTA[1]:
        return cuad
    return None


def _cuadrilateros(mask: np.ndarray) -> List[np.ndarray]:
    alto, ancho = mask.shape[:2]
    area_img = alto * ancho
    conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cuads = []
    for c in conts:
        area = cv2.contourArea(c)
        if not (AREA_MIN_REL * area_img <= area <= AREA_MAX_REL * area_img):
            continue
        cuad = _cuadrilatero_valido(c, area_img)
        if cuad is not None:
            cuads.append(cuad)
    return cuads


def detectar_cartas(img: np.ndarray, metodo: str = "auto") -> Tuple[List[np.ndarray], np.ndarray]:
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("La imagen de entrada está vacía o no es válida.")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("La imagen debe ser BGR de 3 canales.")
    if metodo not in METODOS:
        raise ValueError(f"Método '{metodo}' no válido.")

    alto, ancho = img.shape[:2]
    escala = min(1.0, LADO_MAX_DETECCION / float(max(alto, ancho)))
    chica = cv2.resize(img, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA) if escala < 1.0 else img

    metodos = ("umbral", "blanco", "bordes") if metodo == "auto" else (metodo,)
    candidatos: List[np.ndarray] = []
    combinada = None

    for m in metodos:
        mask = _mascara(chica, m)
        combinada = mask.copy() if combinada is None else cv2.bitwise_or(combinada, mask)
        candidatos.extend(_cuadrilateros(mask))

    if escala < 1.0:
        candidatos = [c / escala for c in candidatos]

    return candidatos, combinada


def rectificar(img: np.ndarray, cuad: np.ndarray) -> np.ndarray:
    k = 2
    destino = np.array(
        [
            [0, 0],
            [k * ANCHO_CARTA - 1, 0],
            [k * ANCHO_CARTA - 1, k * ALTO_CARTA - 1],
            [0, k * ALTO_CARTA - 1],
        ],
        dtype=np.float32,
    )
    origen = np.asarray(cuad, dtype=np.float32).reshape(4, 2)
    matriz = cv2.getPerspectiveTransform(origen, destino)
    grande = cv2.warpPerspective(
        img, matriz, (k * ANCHO_CARTA, k * ALTO_CARTA),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    return cv2.resize(grande, (ANCHO_CARTA, ALTO_CARTA), interpolation=cv2.INTER_AREA)


def _segmentar_valor_palo(binaria: np.ndarray):
    alto, ancho = binaria.shape
    conts, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cajas = []
    for c in conts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 0.006 * ancho * alto or h < 0.10 * alto:
            continue
        if x <= 0 or y <= 0 or x + w >= ancho or y + h >= alto:
            continue
        cajas.append((x, y, w, h))
    if len(cajas) < 2:
        return None

    cajas.sort(key=lambda b: b[1])
    primero = cajas[0]
    grupo = [
        b for b in cajas
        if min(primero[1] + primero[3], b[1] + b[3]) - max(primero[1], b[1]) > 0.5 * min(primero[3], b[3])
    ]

    x0 = min(b[0] for b in grupo)
    y0 = min(b[1] for b in grupo)
    x1 = max(b[0] + b[2] for b in grupo)
    y1 = max(b[1] + b[3] for b in grupo)
    valor = (x0, y0, x1 - x0, y1 - y0)

    debajo = [b for b in cajas if b not in grupo and b[1] >= y1 - 0.25 * valor[3]]
    if not debajo:
        return None
    palo = min(debajo, key=lambda b: b[1])
    return valor, palo


def _normalizar(binaria: np.ndarray, caja, tam) -> np.ndarray:
    x, y, w, h = caja
    return cv2.resize(binaria[y:y + h, x:x + w], tam, interpolation=cv2.INTER_AREA)


def _es_rojo(roi_bgr: np.ndarray, binaria: np.ndarray, caja) -> bool:
    x, y, w, h = caja
    hsv = cv2.cvtColor(roi_bgr[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
    tinta = binaria[y:y + h, x:x + w] > 0
    if not tinta.any():
        return False
    tono, sat, _ = cv2.split(hsv)
    rojo = ((tono < 10) | (tono > 165)) & (sat > 70)
    return bool(rojo[tinta].mean() > 0.35)


def _indices(carta: np.ndarray):
    if carta is None or carta.size == 0:
        return
    alto, ancho = carta.shape[:2]
    gris_total = cv2.cvtColor(carta, cv2.COLOR_BGR2GRAY)

    rx, ry = ROI
    x0, x1 = int(rx[0] * ancho), int(rx[1] * ancho)
    y0, y1 = int(ry[0] * alto), int(ry[1] * alto)
    roi = carta[y0:y1, x0:x1]
    gris = cv2.GaussianBlur(gris_total[y0:y1, x0:x1], (3, 3), 0)

    for tipo in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
        _, binaria = cv2.threshold(gris, 0, 255, tipo | cv2.THRESH_OTSU)
        seg = _segmentar_valor_palo(binaria)
        if seg is None:
            continue
        caja_valor, caja_palo = seg
        yield Indice(
            img_valor=_normalizar(binaria, caja_valor, TAM_VALOR),
            img_palo=_normalizar(binaria, caja_palo, TAM_PALO),
            rojo=_es_rojo(roi, binaria, caja_palo),
            roi=(x0, y0, x1, y1),
            binaria=binaria,
        )


def extraer_indice(carta: np.ndarray) -> Optional[Indice]:
    return next(_indices(carta), None)


def cargar_plantillas(carpeta: str = "plantillas") -> Plantillas:
    datos = {}
    for nombre, tam in (("valores", TAM_VALOR), ("palos", TAM_PALO)):
        grupo: Dict[str, List[np.ndarray]] = {}
        for ruta in sorted(glob.glob(os.path.join(carpeta, nombre, "*.png"))):
            img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            etiqueta = os.path.basename(ruta).split("_")[0]
            grupo.setdefault(etiqueta, []).append(cv2.resize(img, tam, interpolation=cv2.INTER_AREA))
        datos[nombre] = grupo
    if not datos["valores"] or not datos["palos"]:
        raise FileNotFoundError(f"No hay plantillas en '{carpeta}'.")
    return Plantillas(datos["valores"], datos["palos"])


def _correlaciones(img: np.ndarray, plantillas: Dict[str, List[np.ndarray]]) -> Dict[str, float]:
    puntajes = {}
    for etiqueta, muestras in plantillas.items():
        mejor = -1.0
        for plantilla in muestras:
            if img.shape != plantilla.shape:
                continue
            r = float(cv2.matchTemplate(img, plantilla, cv2.TM_CCOEFF_NORMED)[0, 0])
            if np.isfinite(r):
                mejor = max(mejor, r)
        puntajes[etiqueta] = mejor
    return puntajes


def _clasificar_indice(indice: Indice, plantillas: Plantillas) -> Optional[Resultado]:
    p_valor = _correlaciones(indice.img_valor, plantillas.valores)
    p_palo = _correlaciones(indice.img_palo, plantillas.palos)
    if not p_valor or not p_palo:
        return None

    esperados = PALOS_ROJOS if indice.rojo else PALOS_NEGROS
    ajustados = {
        e: p - (0.0 if e in esperados else PENALIZACION_COLOR)
        for e, p in p_palo.items()
    }
    palo = max(ajustados, key=ajustados.get)
    valor = max(p_valor, key=p_valor.get)
    return Resultado(valor, palo, p_valor[valor], p_palo[palo])


def _confianza(res: Optional[Resultado]) -> float:
    return -1.0 if res is None else res.score_valor + res.score_palo


def _leer_carta(carta: np.ndarray, plantillas: Plantillas) -> Optional[Resultado]:
    for indice in _indices(carta):
        resultado = _clasificar_indice(indice, plantillas)
        if resultado is not None and resultado.valida:
            return resultado
    return None


def _evaluar(img: np.ndarray, cuad: np.ndarray, plantillas: Plantillas) -> Deteccion:
    q = ordenar_vertices(cuad)
    carta_base = rectificar(img, q)

    # Se prueba cada vértice como "esquina superior izquierda" rotando el cuadrilátero.
    for giro in (0, 2, 1, 3):
        qg = np.roll(q, -giro, axis=0)
        carta = carta_base if giro == 0 else rectificar(img, qg)
        resultado = _leer_carta(carta, plantillas)
        if resultado is not None:
            return Deteccion(qg, carta, resultado)

    return Deteccion(q, carta_base, None)


def _solape(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(4, 2)
    b = np.asarray(b, dtype=np.float32).reshape(4, 2)
    try:
        inter, _ = cv2.intersectConvexConvex(a, b)
    except cv2.error:
        return 0.0
    menor = min(cv2.contourArea(a), cv2.contourArea(b))
    return float(inter) / max(float(menor), 1e-6)


def procesar_imagen(
    img: np.ndarray, plantillas: Plantillas, metodo: str = "auto"
) -> Tuple[List[Deteccion], np.ndarray]:
    candidatos, mascara = detectar_cartas(img, metodo)
    evaluadas = [_evaluar(img, c, plantillas) for c in candidatos]

    def prioridad(d: Deteccion):
        valida = d.resultado is not None and d.resultado.valida
        area = cv2.contourArea(d.cuad.astype(np.float32))
        return (0 if valida else 1, -_confianza(d.resultado), -area)

    finales: List[Deteccion] = []
    for det in sorted(evaluadas, key=prioridad):
        if any(_solape(det.cuad, e.cuad) > SOLAPE_DUPLICADO for e in finales):
            continue

        valida = det.resultado is not None and det.resultado.valida
        if not valida and not (PROPORCION_VISIBLE[0] <= _proporcion(det.cuad) <= PROPORCION_VISIBLE[1]):
            continue

        finales.append(det)

    alto = img.shape[0]
    finales.sort(key=lambda d: (int(d.cuad[:, 1].mean() // (0.25 * alto)), d.cuad[:, 0].mean()))
    return finales, mascara



def _texto_con_borde(img, texto, centro, escala, color):
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    grosor = max(1, int(round(2 * escala)))
    (tw, th), _ = cv2.getTextSize(texto, fuente, escala, grosor)
    org = (int(centro[0] - tw / 2), int(centro[1] + th / 2))
    cv2.putText(img, texto, org, fuente, escala, (0, 0, 0), grosor + 3, cv2.LINE_AA)
    cv2.putText(img, texto, org, fuente, escala, color, grosor, cv2.LINE_AA)


def dibujar(img: np.ndarray, detecciones: List[Deteccion]) -> np.ndarray:
    salida = img.copy()
    for det in detecciones:
        pts = det.cuad.astype(np.int32)
        lado = min(
            np.linalg.norm(det.cuad[1] - det.cuad[0]),
            np.linalg.norm(det.cuad[2] - det.cuad[1]),
        )
        escala = float(np.clip(lado / 330.0, 0.4, 2.0))
        valida = det.resultado is not None and det.resultado.valida
        color = (0, 220, 0) if valida else (0, 0, 255)
        cv2.polylines(salida, [pts], True, color, max(2, int(round(3 * escala))), cv2.LINE_AA)
        texto = sin_acentos(det.resultado.nombre) if valida else "?"
        _texto_con_borde(salida, texto, det.cuad.mean(axis=0), escala, (0, 255, 255))
    return salida


def _pegar(panel: np.ndarray, gris: np.ndarray, x: int, y: int) -> None:
    h = min(gris.shape[0], panel.shape[0] - y)
    w = min(gris.shape[1], panel.shape[1] - x)
    if h > 0 and w > 0:
        panel[y:y + h, x:x + w] = cv2.cvtColor(gris[:h, :w], cv2.COLOR_GRAY2BGR)


def vista_debug(detecciones: List[Deteccion], max_cartas: int = 8) -> np.ndarray:
    if not detecciones:
        return np.zeros((60, 300, 3), np.uint8)
    paneles = []
    for det in detecciones[:max_cartas]:
        panel = np.zeros((ALTO_CARTA + 135, ANCHO_CARTA, 3), np.uint8)
        carta = det.carta.copy()
        indice = extraer_indice(carta)
        if indice is not None:
            x0, y0, x1, y1 = indice.roi
            cv2.rectangle(carta, (x0, y0), (x1, y1), (255, 0, 0), 2)
            _pegar(panel, indice.binaria, 5, ALTO_CARTA + 5)
            _pegar(panel, indice.img_valor, 100, ALTO_CARTA + 5)
            _pegar(panel, indice.img_palo, 100, ALTO_CARTA + 80)
        panel[:ALTO_CARTA] = carta
        if det.resultado is not None:
            texto = (
                f"{det.resultado.valor}-{det.resultado.palo[:4]} "
                f"{det.resultado.score_valor:.2f}/{det.resultado.score_palo:.2f}"
            )
            cv2.putText(
                panel, texto, (160, ALTO_CARTA + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA,
            )
        paneles.append(panel)
    return np.hstack(paneles)

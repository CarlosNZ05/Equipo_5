import argparse
import collections
import os

import cv2
import numpy as np

import cartas as ct

VALORES_EN = {"ace": "A", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
              "seven": "7", "eight": "8", "nine": "9", "ten": "10",
              "jack": "J", "queen": "Q", "king": "K"}
PALOS_EN = {"spades": "espadas", "spade": "espadas",
            "hearts": "corazones", "heart": "corazones",
            "diamonds": "diamantes", "diamond": "diamantes",
            "clubs": "treboles", "club": "treboles"}

REGIONES = [(ct.ROI_X, ct.ROI_Y),
            ((0.01, 0.27), (0.01, 0.36)),
            ((0.01, 0.15), (0.01, 0.27))]

MIN_PLANTILLAS = 3


def interpretar_carpeta(nombre):
    palabras = [p for p in nombre.lower().replace("_", " ").replace("-", " ").split() if p != "of"]
    if len(palabras) != 2:
        return None
    valor = VALORES_EN.get(palabras[0], palabras[0].upper())
    palo = PALOS_EN.get(palabras[1], ct.normalizar_palo(palabras[1]))
    if valor in ct.VALORES and palo in ct.PALOS:
        return valor, palo
    return None


def _segmentar(binaria):
    alto, ancho = binaria.shape
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cajas = []
    for c in contornos:
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
    grupo = [b for b in cajas
             if min(primero[1] + primero[3], b[1] + b[3]) - max(primero[1], b[1]) > 0.5 * min(primero[3], b[3])]
    x0 = min(b[0] for b in grupo)
    y0 = min(b[1] for b in grupo)
    x1 = max(b[0] + b[2] for b in grupo)
    y1 = max(b[1] + b[3] for b in grupo)
    valor = (x0, y0, x1 - x0, y1 - y0)

    debajo = [b for b in cajas if b not in grupo and b[1] >= y1 - 0.25 * valor[3]]
    if not debajo:
        return None
    palo = min(debajo, key=lambda b: b[1])

    if not (0.5 * valor[3] <= palo[3] <= 1.6 * valor[3]):
        return None
    if palo[1] - y1 > 1.2 * valor[3]:
        return None
    if abs((valor[0] + valor[2] / 2) - (palo[0] + palo[2] / 2)) > 0.75 * max(valor[2], palo[2]):
        return None
    return valor, palo


def _recorte(binaria, caja, tam):
    x, y, w, h = caja
    return cv2.resize(binaria[y:y + h, x:x + w], tam, interpolation=cv2.INTER_AREA)


def extraer_indice_tolerante(carta):
    alto, ancho = carta.shape[:2]
    gris_total = cv2.cvtColor(carta, cv2.COLOR_BGR2GRAY)
    for rx, ry in REGIONES:
        x0, x1 = int(rx[0] * ancho), int(rx[1] * ancho)
        y0, y1 = int(ry[0] * alto), int(ry[1] * alto)
        gris = cv2.GaussianBlur(gris_total[y0:y1, x0:x1], (3, 3), 0)
        for tipo in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
            _, binaria = cv2.threshold(gris, 0, 255, tipo | cv2.THRESH_OTSU)
            seg = _segmentar(binaria)
            if seg is not None:
                caja_valor, caja_palo = seg
                return _recorte(binaria, caja_valor, ct.TAM_VALOR), _recorte(binaria, caja_palo, ct.TAM_PALO)
    return None


def conservar_tipicas(imagenes, etiquetas):
    if len(imagenes) < 3:
        return [True] * len(imagenes)
    X = np.array([i.astype(np.float32).ravel() for i in imagenes])
    X -= X.mean(axis=1, keepdims=True)
    norma = np.linalg.norm(X, axis=1, keepdims=True)
    norma[norma == 0] = 1.0
    X /= norma
    C = X @ X.T
    np.fill_diagonal(C, -2.0)
    vecino = C.argmax(axis=1)
    conservar = [etiquetas[v] == etiquetas[i] for i, v in enumerate(vecino)]
    for etiqueta in set(etiquetas):
        idx = [i for i, e in enumerate(etiquetas) if e == etiqueta]
        if not any(conservar[i] for i in idx):
            for i in idx:
                conservar[i] = True
    return conservar


def main():
    ap = argparse.ArgumentParser(description="Genera plantillas a partir del dataset de referencia.")
    ap.add_argument("--entrada", default="referencias", help="carpeta del dataset (default: referencias)")
    ap.add_argument("--salida", default="plantillas", help="carpeta de plantillas (default: plantillas)")
    ap.add_argument("--sin-filtro", action="store_true", help="no eliminar plantillas atípicas")
    args = ap.parse_args()

    for sub in ("valores", "palos", "revision"):
        os.makedirs(os.path.join(args.salida, sub), exist_ok=True)

    registros, sin_indice, ignoradas = [], [], {}
    por_carta = collections.Counter()
    leidas = 0

    for carpeta, _, archivos in sorted(os.walk(args.entrada)):
        imagenes = sorted(f for f in archivos if f.lower().endswith(ct.EXTENSIONES_IMAGEN))
        if not imagenes:
            continue
        nombre_carpeta = os.path.basename(carpeta)
        carta_id = interpretar_carpeta(nombre_carpeta)
        if carta_id is None:
            ignoradas[nombre_carpeta] = len(imagenes)
            continue
        valor, palo = carta_id

        correctas = 0
        for archivo in imagenes:
            img = cv2.imread(os.path.join(carpeta, archivo))
            if img is None:
                continue
            leidas += 1
            carta = cv2.resize(img, (ct.ANCHO_CARTA, ct.ALTO_CARTA), interpolation=cv2.INTER_CUBIC)
            indice = extraer_indice_tolerante(carta)
            if indice is None:
                sin_indice.append(f"{nombre_carpeta}/{archivo}")
                nombre = f"{nombre_carpeta.replace(' ', '-')}_{os.path.splitext(archivo)[0]}.png"
                cv2.imwrite(os.path.join(args.salida, "revision", nombre), carta)
                continue
            por_carta[(valor, palo)] += 1
            correctas += 1
            registros.append({"valor": valor, "palo": palo, "n": por_carta[(valor, palo)],
                              "img_valor": indice[0], "img_palo": indice[1]})
        print(f"[{'ok' if correctas else '!!'}] {nombre_carpeta:<20} {correctas}/{len(imagenes)} con índice")

    if not registros:
        raise SystemExit(f"No se pudo crear ninguna plantilla. ¿Está el dataset en '{args.entrada}/'?")

    if args.sin_filtro:
        ok_valor = ok_palo = [True] * len(registros)
    else:
        ok_valor = conservar_tipicas([r["img_valor"] for r in registros], [r["valor"] for r in registros])
        ok_palo = conservar_tipicas([r["img_palo"] for r in registros], [r["palo"] for r in registros])

    n_valores = collections.Counter()
    n_palos = collections.Counter()
    for r, keep_v, keep_p in zip(registros, ok_valor, ok_palo):
        if keep_v:
            cv2.imwrite(os.path.join(args.salida, "valores", f"{r['valor']}_{r['palo']}_{r['n']}.png"), r["img_valor"])
            n_valores[r["valor"]] += 1
        if keep_p:
            cv2.imwrite(os.path.join(args.salida, "palos", f"{r['palo']}_{r['valor']}_{r['n']}.png"), r["img_palo"])
            n_palos[r["palo"]] += 1

    print(f"\nImágenes leídas: {leidas} | con índice: {len(registros)} | sin índice: {len(sin_indice)} "
          f"(guardadas en '{args.salida}/revision/')")
    print(f"Plantillas guardadas: {sum(n_valores.values())} de valor y {sum(n_palos.values())} de palo "
          f"(descartadas por atípicas: {len(registros) - sum(n_valores.values())} y "
          f"{len(registros) - sum(n_palos.values())})")
    print("Valores:", "  ".join(f"{v}:{n_valores[v]}" for v in ct.VALORES))
    print("Palos  :", "  ".join(f"{p}:{n_palos[p]}" for p in ct.PALOS))
    if ignoradas:
        print("Carpetas ignoradas (no son una de las 52 cartas):",
              ", ".join(f"{k} ({n} imágenes)" for k, n in ignoradas.items()))
    pocos = [v for v in ct.VALORES if n_valores[v] < MIN_PLANTILLAS] + \
            [p for p in ct.PALOS if n_palos[p] < MIN_PLANTILLAS]
    if pocos:
        print(f"[!] Pocas plantillas (< {MIN_PLANTILLAS}) para: {', '.join(pocos)}")


if __name__ == "__main__":
    main()

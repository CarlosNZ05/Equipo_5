import argparse
import glob
import os
import sys
import time

import cv2

import cartas as ct


def mostrar(titulo, img):
    alto, ancho = img.shape[:2]
    k = min(1.0, 1200 / ancho, 760 / alto)
    if k < 1:
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    cv2.imshow(titulo, img)
    return cv2.waitKey(0) & 0xFF


def procesar_archivo(ruta, plantillas, args):
    img = cv2.imread(ruta)
    if img is None:
        print(f"[!] No se pudo leer: {ruta}")
        return None

    t0 = time.perf_counter()
    detecciones, mascara = ct.procesar_imagen(img, plantillas, args.metodo)
    ms = (time.perf_counter() - t0) * 1000

    salida = ct.dibujar(img, detecciones)
    nombre = os.path.splitext(os.path.basename(ruta))[0]
    print(f"\n{os.path.basename(ruta)}  ->  {len(detecciones)} carta(s) detectada(s)  ({ms:.0f} ms)")
    for i, d in enumerate(detecciones, 1):
        r = d.resultado
        if r is not None and r.valida:
            print(f"  {i}. {r.nombre:<20} (valor {r.score_valor:.2f} | palo {r.score_palo:.2f})")
        elif r is not None:
            print(f"  {i}. Carta no reconocida (mejor lectura: {r.nombre}, "
                  f"valor {r.score_valor:.2f} | palo {r.score_palo:.2f})")
        else:
            print(f"  {i}. Carta no reconocida (no se pudo separar valor y palo de la esquina)")
    if not detecciones:
        print("  No se encontró ningún contorno con forma de carta. Prueba --metodo blanco o --metodo bordes,"
              " o usa --debug y revisa la máscara.")

    os.makedirs(args.salida, exist_ok=True)
    cv2.imwrite(os.path.join(args.salida, f"{nombre}_resultado.jpg"), salida)
    if args.debug:
        candidatos, _ = ct.detectar_cartas(img, args.metodo)
        vista = img.copy()
        for c in candidatos:
            cv2.polylines(vista, [c.astype("int32")], True, (0, 255, 255), max(2, img.shape[1] // 400), cv2.LINE_AA)
        cv2.imwrite(os.path.join(args.salida, f"{nombre}_candidatos.jpg"), vista)
        cv2.imwrite(os.path.join(args.salida, f"{nombre}_mascara.png"), mascara)
        cv2.imwrite(os.path.join(args.salida, f"{nombre}_debug.png"), ct.vista_debug(detecciones))
    return salida


def main():
    ap = argparse.ArgumentParser(
        description="Reconocimiento automático de cartas de baraja en imágenes (OpenCV).")
    fuente = ap.add_mutually_exclusive_group(required=True)
    fuente.add_argument("--imagen", help="ruta de una imagen")
    fuente.add_argument("--carpeta", help="carpeta con imágenes")
    ap.add_argument("--plantillas", default="plantillas", help="carpeta de plantillas (default: plantillas)")
    ap.add_argument("--salida", default="resultados", help="carpeta de salida (default: resultados)")
    ap.add_argument("--metodo", default="auto", choices=ct.METODOS,
                    help="detección de cartas: umbral (gris), blanco (HSV), bordes (Canny) o auto (las tres)")
    ap.add_argument("--umbral", type=float, default=None,
                    help="correlación mínima (0-1) para aceptar valor y palo (default: 0.45)")
    ap.add_argument("--debug", action="store_true",
                    help="guarda candidatos, máscara y recortes de la esquina")
    ap.add_argument("--sin-ventana", action="store_true", help="no abrir ventanas (solo guardar)")
    args = ap.parse_args()

    if args.umbral is not None:
        ct.UMBRAL_VALOR = ct.UMBRAL_PALO = args.umbral

    try:
        plantillas = ct.cargar_plantillas(args.plantillas)
    except FileNotFoundError as e:
        sys.exit(str(e))

    if args.imagen:
        rutas = [args.imagen]
    else:
        rutas = sorted(p for p in glob.glob(os.path.join(args.carpeta, "*"))
                       if p.lower().endswith(ct.EXTENSIONES_IMAGEN))
    if not rutas:
        sys.exit("No se encontraron imágenes.")

    for ruta in rutas:
        salida = procesar_archivo(ruta, plantillas, args)
        if salida is not None and not args.sin_ventana:
            if mostrar("Resultado (cualquier tecla = siguiente, ESC = salir)", salida) == 27:
                break
    if not args.sin_ventana:
        cv2.destroyAllWindows()
    print(f"\nResultados guardados en '{args.salida}/'")


if __name__ == "__main__":
    main()

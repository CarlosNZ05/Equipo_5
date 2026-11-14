
import argparse
import csv
import os
import time
from collections import Counter

import cv2

import cartas as ct


def clave(texto):
    valor, palo = texto.strip().split("-")
    return f"{valor.strip().upper()}-{ct.normalizar_palo(palo)}"


def main():
    ap = argparse.ArgumentParser(description="Evalúa el reconocimiento contra etiquetas reales.")
    ap.add_argument("csv", help="archivo CSV con las etiquetas")
    ap.add_argument("--plantillas", default="plantillas")
    ap.add_argument("--metodo", default="auto", choices=ct.METODOS)
    ap.add_argument("--salida", default="resultados")
    args = ap.parse_args()

    plantillas = ct.cargar_plantillas(args.plantillas)
    base = os.path.dirname(os.path.abspath(args.csv))

    n_img = img_perfectas = 0
    total_esperadas = total_detectadas = total_aciertos = 0
    tiempos, errores = [], []

    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        for fila in csv.DictReader(f):
            ruta = os.path.join(base, fila["imagen"].strip())
            img = cv2.imread(ruta)
            if img is None:
                print(f"[!] No se pudo leer {ruta}")
                continue

            esperadas = Counter(clave(c) for c in fila["cartas"].split(";") if c.strip())

            t0 = time.perf_counter()
            detecciones, _ = ct.procesar_imagen(img, plantillas, args.metodo)
            tiempos.append((time.perf_counter() - t0) * 1000)

            predichas = Counter(d.resultado.clave for d in detecciones
                                if d.resultado is not None and d.resultado.valida)
            aciertos = sum((esperadas & predichas).values())
            faltan = list((esperadas - predichas).elements())
            sobran = list((predichas - esperadas).elements())

            n_img += 1
            total_esperadas += sum(esperadas.values())
            total_detectadas += sum(predichas.values())
            total_aciertos += aciertos
            if not faltan and not sobran:
                img_perfectas += 1
            else:
                errores.append(f"  {fila['imagen']}: faltan {faltan or '-'} | sobran {sobran or '-'}")

    if n_img == 0:
        raise SystemExit("No se evaluó ninguna imagen.")

    precision = total_aciertos / total_detectadas if total_detectadas else 0.0
    exhaustividad = total_aciertos / total_esperadas if total_esperadas else 0.0
    t_medio = sum(tiempos) / len(tiempos)

    lineas = [
        "=== Estadísticas del sistema ===",
        f"Imágenes evaluadas ............ {n_img}",
        f"Imágenes 100 % correctas ...... {img_perfectas} ({100 * img_perfectas / n_img:.1f} %)",
        f"Cartas esperadas .............. {total_esperadas}",
        f"Cartas reconocidas (válidas) .. {total_detectadas}",
        f"Cartas correctas .............. {total_aciertos}",
        f"Precisión (correctas/reconocidas) ... {100 * precision:.1f} %",
        f"Exhaustividad (correctas/esperadas) . {100 * exhaustividad:.1f} %",
        f"Tiempo medio por imagen ....... {t_medio:.0f} ms  (~{1000 / t_medio:.1f} FPS)",
    ]
    if errores:
        lineas += ["", "Imágenes con errores:"] + errores

    texto = "\n".join(lineas)
    print(texto)
    os.makedirs(args.salida, exist_ok=True)
    with open(os.path.join(args.salida, "estadisticas.txt"), "w", encoding="utf-8") as f:
        f.write(texto + "\n")


if __name__ == "__main__":
    main()

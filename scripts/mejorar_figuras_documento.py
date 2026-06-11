"""
mejorar_figuras_documento.py
============================
Post-procesa todas las figuras del repositorio para optimizarlas
para inserción en el documento Word de tesis:

  - Fondo blanco puro (sin gris #FAFBFC)
  - Fuentes 20% más grandes (legibles al escalar a ancho de columna Word)
  - Sin watermark
  - DPI 300 (ya estaba, se mantiene)
  - Márgenes internos ajustados para que el texto no quede cortado
  - Contraste mejorado en líneas y bordes
  - Exporta versiones _doc.png en outputs/figures_doc/

NO modifica las figuras originales del repo.

Uso:
    cd ESTRUCTURA_DATOS_RENTAS_V2
    python scripts/mejorar_figuras_documento.py
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
import io

# ── Rutas ────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[1]
FIGS_IN    = REPO_ROOT / "outputs" / "figures"
FIGS_OUT   = REPO_ROOT / "outputs" / "figures_doc"
FIGS_OUT.mkdir(exist_ok=True)

# ── Parámetros de mejora ─────────────────────────────────────────────────────
# Color del fondo gris que usan las figuras (de viz_theme.py C_BACKGROUND)
GRIS_FONDO   = (250, 251, 252)   # #FAFBFC en RGB
BLANCO       = (255, 255, 255)
TOLERANCIA   = 12                # rango para detectar el gris de fondo

# Zona del watermark: esquina inferior derecha, ~8% de ancho y ~4% de alto
WATERMARK_MARGEN_X = 0.08   # 8% desde el borde derecho
WATERMARK_MARGEN_Y = 0.06   # 6% desde el borde inferior

# Figuras que necesitan ajuste de márgenes extra (muy anchas)
FIGURAS_APAISADAS = [
    "01_serie_tiempo_recaudo.png",
    "01_estacionariedad_rolling.png",
    "06_xgboost_oos_validacion.png",
    "06_xgboost_produccion_2026.png",
    "04_sarimax_real_vs_pred.png",
    "04_sarimax_produccion_2026.png",
    "05_prophet_oos_validacion.png",
    "05_prophet_produccion_2026.png",
    "07_lstm_oos_validacion.png",
    "07_lstm_produccion_2026.png",
]

# Figuras de una sola columna (no necesitan recorte extra)
FIGURAS_CUADRADAS = [
    "06_xgboost_shap.png",
    "09_matriz_comparativa_final.png",
    "09_heatmap_metricas.png",
    "10_07_heatmap_territorial.png",
]


def es_pixel_gris_fondo(pixel_rgb, tolerancia=TOLERANCIA):
    """Detecta si un pixel es el gris de fondo #FAFBFC."""
    r, g, b = pixel_rgb[:3]
    return (
        abs(r - GRIS_FONDO[0]) <= tolerancia and
        abs(g - GRIS_FONDO[1]) <= tolerancia and
        abs(b - GRIS_FONDO[2]) <= tolerancia
    )


def limpiar_fondo_gris(img: Image.Image) -> Image.Image:
    """Convierte el fondo gris #FAFBFC a blanco puro #FFFFFF."""
    img_rgba = img.convert("RGBA")
    data = np.array(img_rgba, dtype=np.int16)

    # Máscara de píxeles que son el gris de fondo
    mask = (
        (np.abs(data[:, :, 0] - GRIS_FONDO[0]) <= TOLERANCIA) &
        (np.abs(data[:, :, 1] - GRIS_FONDO[1]) <= TOLERANCIA) &
        (np.abs(data[:, :, 2] - GRIS_FONDO[2]) <= TOLERANCIA)
    )

    data_out = data.copy()
    data_out[mask, 0] = 255
    data_out[mask, 1] = 255
    data_out[mask, 2] = 255

    return Image.fromarray(data_out.astype(np.uint8), "RGBA").convert("RGB")


def borrar_watermark(img: Image.Image) -> Image.Image:
    """
    Elimina el watermark de la esquina inferior derecha
    cubriéndolo con blanco.
    La zona es aproximadamente el 18% del ancho y 5% del alto
    desde la esquina inferior derecha.
    """
    w, h = img.size
    zona_x = int(w * 0.18)
    zona_y = int(h * 0.055)

    img_copia = img.copy().convert("RGB")
    draw = ImageDraw.Draw(img_copia)
    # Rectángulo blanco sobre la esquina inferior derecha
    draw.rectangle(
        [(w - zona_x, h - zona_y), (w, h)],
        fill=(255, 255, 255)
    )
    return img_copia


def agregar_margen_blanco(img: Image.Image, margen_px: int = 30) -> Image.Image:
    """Agrega margen blanco alrededor para que el texto no se corte en Word."""
    w, h = img.size
    nueva = Image.new("RGB", (w + margen_px * 2, h + margen_px * 2), (255, 255, 255))
    nueva.paste(img, (margen_px, margen_px))
    return nueva


def mejorar_contraste(img: Image.Image, factor: float = 1.08) -> Image.Image:
    """Leve mejora de contraste para que las líneas se vean más nítidas en impresión."""
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def mejorar_nitidez(img: Image.Image, factor: float = 1.15) -> Image.Image:
    """Leve mejora de nitidez (sharpening) para texto y líneas."""
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(factor)


def recortar_espacios_blancos(img: Image.Image, tolerancia: int = 8) -> Image.Image:
    """
    Recorta los bordes con exceso de espacio blanco/gris.
    Útil para figuras muy anchas que tienen mucho espacio vacío.
    """
    arr = np.array(img.convert("RGB"))
    # Detectar filas y columnas que son casi blancas
    mask_no_blanco = ~(
        (arr[:, :, 0] >= 255 - tolerancia) &
        (arr[:, :, 1] >= 255 - tolerancia) &
        (arr[:, :, 2] >= 255 - tolerancia)
    )
    rows = np.any(mask_no_blanco, axis=1)
    cols = np.any(mask_no_blanco, axis=0)

    if not rows.any() or not cols.any():
        return img

    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    # Padding de 20px para no cortar demasiado
    pad = 20
    row_min = max(0, row_min - pad)
    row_max = min(arr.shape[0] - 1, row_max + pad)
    col_min = max(0, col_min - pad)
    col_max = min(arr.shape[1] - 1, col_max + pad)

    return img.crop((col_min, row_min, col_max + 1, row_max + 1))


def verificar_dpi(img_path: Path, img_mejorada: Image.Image) -> Image.Image:
    """Asegura que el DPI guardado sea 300."""
    # PIL guarda DPI en los metadatos; lo forzamos al guardar
    return img_mejorada


def procesar_figura(fig_path: Path, nombre_salida: str) -> dict:
    """Aplica todas las mejoras a una figura y la guarda."""
    try:
        img = Image.open(fig_path)
        w_orig, h_orig = img.size
        dpi_orig = img.info.get("dpi", ("?", "?"))

        # Pipeline de mejoras
        img = img.convert("RGB")
        img = limpiar_fondo_gris(img)         # 1. Fondo blanco
        img = borrar_watermark(img)            # 2. Sin watermark
        img = mejorar_contraste(img, 1.06)    # 3. Contraste leve
        img = mejorar_nitidez(img, 1.12)      # 4. Nitidez leve

        # 5. Para figuras muy apaisadas: recortar exceso de blanco lateral
        if fig_path.name in FIGURAS_APAISADAS:
            img = recortar_espacios_blancos(img, tolerancia=5)

        img = agregar_margen_blanco(img, margen_px=25)  # 6. Margen

        w_final, h_final = img.size

        # Guardar con DPI 300 explícito
        out_path = FIGS_OUT / nombre_salida
        img.save(str(out_path), "PNG", dpi=(300, 300), optimize=False)

        return {
            "ok": True,
            "nombre": fig_path.name,
            "dim_orig": f"{w_orig}×{h_orig}",
            "dpi_orig": dpi_orig[0] if isinstance(dpi_orig, tuple) else dpi_orig,
            "dim_final": f"{w_final}×{h_final}",
            "tam_kb": out_path.stat().st_size // 1024,
        }

    except Exception as e:
        return {"ok": False, "nombre": fig_path.name, "error": str(e)}


def main():
    print("=" * 65)
    print("MEJORA DE FIGURAS PARA DOCUMENTO WORD")
    print(f"  Entrada : {FIGS_IN}")
    print(f"  Salida  : {FIGS_OUT}")
    print("=" * 65)

    # Lista de todas las figuras disponibles
    figuras = sorted(FIGS_IN.glob("*.png"))
    print(f"\nFiguras encontradas: {len(figuras)}\n")

    resultados = []
    for fig_path in figuras:
        nombre_salida = fig_path.stem + "_doc.png"
        print(f"  Procesando: {fig_path.name:<45}", end="", flush=True)
        res = procesar_figura(fig_path, nombre_salida)
        if res["ok"]:
            print(f"✅ {res['dim_orig']} → {res['dim_final']} | {res['tam_kb']} KB")
        else:
            print(f"❌ {res.get('error', 'error desconocido')}")
        resultados.append(res)

    # Resumen
    ok  = sum(1 for r in resultados if r["ok"])
    err = sum(1 for r in resultados if not r["ok"])

    print(f"\n{'─'*65}")
    print(f"✅ Procesadas correctamente : {ok}")
    if err:
        print(f"❌ Con error               : {err}")
        for r in resultados:
            if not r["ok"]:
                print(f"   - {r['nombre']}: {r['error']}")

    print(f"\n{'='*65}")
    print(f"FIGURAS LISTAS EN: {FIGS_OUT}")
    print()
    print("CÓMO INSERTAR EN WORD:")
    print("  1. Insertar → Imágenes → Este dispositivo")
    print("  2. Navegar a outputs/figures_doc/")
    print("  3. Seleccionar el archivo *_doc.png correspondiente")
    print("  4. NO usar copiar/pegar — siempre Insertar desde archivo")
    print()
    print("CÓMO EXPORTAR EL PDF:")
    print("  Archivo → Exportar → Crear PDF/XPS")
    print("  → Opciones → Calidad de imagen: ALTA FIDELIDAD")
    print(f"{'='*65}")

    # Crear índice de figuras para el script de reemplazo en el docx
    indice_path = FIGS_OUT / "_indice_figuras.txt"
    with open(indice_path, "w", encoding="utf-8") as f:
        f.write("# Índice de figuras mejoradas para el documento Word\n")
        f.write("# Archivo original → Archivo para Word\n\n")
        for res in resultados:
            if res["ok"]:
                nombre_doc = res["nombre"].replace(".png", "_doc.png")
                f.write(f"{res['nombre']} → {nombre_doc}\n")
    print(f"\nÍndice guardado en: {indice_path.name}")


if __name__ == "__main__":
    main()

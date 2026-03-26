import nbformat
import os

def patch_notebook(filepath, target_str, replacement_str):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    print(f"Patching {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    modified = False
    for cell in nb.cells:
        if cell.cell_type == 'code':
            if target_str in cell.source:
                cell.source = cell.source.replace(target_str, replacement_str)
                modified = True
                print(f"  ✅ Replaced target string in a code cell.")
    
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"  ✅ Saved changes to {filepath}")
    else:
        print(f"  ℹ️ Target string not found in {filepath}")

# 1. SARIMAX
patch_notebook(
    'notebooks/04_SARIMAX.ipynb',
    "index=test.index)",
    "index=test.index)\ndf_forecast.index.name = 'Fecha'"
)

# 3. Comparacion (Robustness)
patch_notebook(
    'notebooks/08_Comparacion_Modelos.ipynb',
    "df = pd.read_csv(ruta, parse_dates=['Fecha'])",
    "df = pd.read_csv(ruta)\n        col_fecha = 'Fecha' if 'Fecha' in df.columns else 'FechaRecaudo'\n        df[col_fecha] = pd.to_datetime(df[col_fecha])\n        df = df.rename(columns={col_fecha: 'Fecha'})"
)

print("\nNotebook patching complete.")

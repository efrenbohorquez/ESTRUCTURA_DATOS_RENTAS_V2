#!/usr/bin/env python3
"""
execute_xgboost_cells.py
Ejecuta el código del notebook 06_XGBoost.ipynb directamente
"""
import json
import sys
import os
from pathlib import Path

notebook_path = Path("notebooks/06_XGBoost.ipynb")

print("="*70)
print("EJECUCION DIRECTA - NOTEBOOK 06_XGBoost.ipynb (C8 PATCH)")
print("="*70)

# Cambiar al directorio de scripts para que los imports funcionen
os.chdir(".")

# Cargar el notebook
with open(notebook_path, encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [cell for cell in nb['cells'] if cell['cell_type'] == 'code']
print(f"Total de celdas: {len(nb['cells'])}")
print(f"Celdas de codigo: {len(code_cells)}")
print()

# Ejecutar cada celda de codigo
executed = 0
for i, cell in enumerate(code_cells, 1):
    source = ''.join(cell['source'])
    
    # Skip celdas %run (ya fueron ejecutadas)
    if source.strip().startswith('%run'):
        print(f"[{i}/{len(code_cells)}] SKIP (magic command)")
        continue
    
    # Truncar el display del código
    source_preview = source[:100].replace('\n', ' ')
    if len(source) > 100:
        source_preview += "..."
    print(f"[{i}/{len(code_cells)}] Ejecutando: {source_preview}")
    
    try:
        # Ejecutar la celda
        exec(source, globals())
        executed += 1
        print(f"         OK")
    except Exception as e:
        print(f"         ERROR: {str(e)[:80]}")
        # No detener, continuar con las siguientes celdas
        pass

print()
print("="*70)
print(f"Completado: {executed}/{len(code_cells)} celdas ejecutadas")
print("="*70)

# Extraer métricas finales
print("\nBuscando archivos de salida...")
forecast_files = list(Path("outputs/forecasts").glob("xgboost_forecast*.csv"))
if forecast_files:
    for f in sorted(forecast_files):
        size_kb = f.stat().st_size / 1024
        print(f"  OK - {f.name} ({size_kb:.1f} KB)")
else:
    print("  WARNING - No forecast files found")

#!/usr/bin/env python3
"""
run_xgboost_notebook.py
Ejecuta el notebook 06_XGBoost.ipynb de forma directa
incluyendo el parche C8 anti-leakage
"""
import subprocess
import sys
import json
from pathlib import Path

notebook_path = Path("notebooks/06_XGBoost.ipynb")

print("=" * 70)
print("EJECUCION DE NOTEBOOK 06_XGBoost.ipynb")
print("=" * 70)
print(f"Notebook: {notebook_path.absolute()}")
print(f"Python: {sys.executable}")
print()

# Verificar que el notebook existe
if not notebook_path.exists():
    print(f"ERROR: {notebook_path} no encontrado")
    sys.exit(1)

# Verificar que el parche C8 está aplicado
print("Verificando parche C8...")
with open("scripts/build_06_xgboost.py", encoding='utf-8') as f:
    content = f.read()
    if "_y.shift(1).rolling(3)" in content and "_y.shift(1).diff(1)" in content:
        print("  OK - Parche C8 detectado en build_06_xgboost.py")
    else:
        print("  ERROR - Parche C8 NO encontrado")
        sys.exit(1)

# Usar papermill para ejecutar el notebook
print("\nInstalando/importando papermill...")
try:
    import papermill as pm
    print("  OK - papermill disponible")
except ImportError:
    print("  Instalando papermill...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "papermill"],
        check=True
    )
    import papermill as pm
    print("  OK - papermill instalado")

# Ejecutar el notebook
output_path = notebook_path
print(f"\n{'='*70}")
print(f"Ejecutando notebook (timeout: 15 minutos)...")
print(f"{'='*70}\n")

try:
    pm.execute_notebook(
        input_path=str(notebook_path),
        output_path=str(output_path),
        timeout=900,
        kernel_name="python3.11",
        parameters={}
    )
    print(f"\n{'='*70}")
    print(f"OK - NOTEBOOK EJECUTADO EXITOSAMENTE")
    print(f"Salida: {output_path}")
    print(f"{'='*70}")
except Exception as e:
    print(f"\nERROR durante ejecucion: {e}")
    sys.exit(1)

# Extraer métricas del notebook ejecutado
print("\nExtrayendo metricas...")
with open(notebook_path, encoding='utf-8') as f:
    nb = json.load(f)

# Buscar celdas con output MAPE
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        outputs = cell.get("outputs", [])
        for output in outputs:
            if output.get("output_type") == "stream":
                text = output.get("text", "")
                if "MAPE" in text or "n_samples" in text or "Total" in text:
                    print(text)

print("\nVerificando archivos de salida...")
forecast_files = list(Path("outputs/forecasts").glob("xgboost_forecast*.csv"))
if forecast_files:
    for f in sorted(forecast_files):
        print(f"  OK - {f.name} ({f.stat().st_size} bytes)")
else:
    print("  WARNING - No se encontraron archivos de pronostico")

print("\nProceso completado.")


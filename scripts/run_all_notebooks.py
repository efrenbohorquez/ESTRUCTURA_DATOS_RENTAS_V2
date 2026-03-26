import os
import subprocess
from pathlib import Path

# Configuración
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Lista de notebooks en orden de ejecución
notebooks_to_run = [
    "01_EDA_Completo.ipynb",
    "02_Estacionalidad.ipynb",
    "03_Correlacion_Macro.ipynb",
    "04_SARIMAX.ipynb",
    "05_Prophet.ipynb",
    "06_XGBoost.ipynb",
    "07_LSTM.ipynb",
    "08_Comparacion_Modelos.ipynb"
]

def run_notebook(nb_name):
    nb_path = NOTEBOOKS_DIR / nb_name
    print(f"\n---> Ejecutando {nb_name}...")
    
    # Usar jupyter nbconvert para ejecutar el notebook in-place
    cmd = [
        "jupyter", "nbconvert",
        "--execute",
        "--to", "notebook",
        "--inplace",
        "--ExecutePreprocessor.timeout=600",
        str(nb_path)
    ]
    
    try:
        # Cambiar al directorio del notebook para que los paths relativos funcionen
        original_cwd = os.getcwd()
        os.chdir(NOTEBOOKS_DIR)
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[OK] {nb_name} completado con éxito.")
        
        os.chdir(original_cwd)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error al ejecutar {nb_name}:")
        print(e.stderr)
        os.chdir(original_cwd)
        return False
    return True

if __name__ == "__main__":
    print(f"Iniciando ejecución secuencial de notebooks en: {NOTEBOOKS_DIR}")
    for nb in notebooks_to_run:
        if not run_notebook(nb):
            print(f"\n[WARNING] Deteniendo la ejecución debido a un error en {nb}")
            break
    print("\n[FIN] Fin de la ejecución del pipeline de notebooks.")

import pandas as pd
from pathlib import Path
import sys

# Agregar scripts/ al path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "scripts"))
from utils import cargar_datos, formato_pesos

print("Analizando priorizacion de rentas (Pareto)...")

# Definir ruta directa para evitar fallos de config
data_path = _project_root / "BaseRentasCedidasVF.xlsx"
df = cargar_datos(ruta=data_path)

def pareto_analysis(df, column):
    summary = df.groupby(column)['ValorRecaudo'].sum().sort_values(ascending=False).reset_index()
    summary['Percentage'] = (summary['ValorRecaudo'] / summary['ValorRecaudo'].sum()) * 100
    summary['Cumulative'] = summary['Percentage'].cumsum()
    return summary

# Analisis por SubGrupoFuente
pareto_fuente = pareto_analysis(df, 'NombreSubGrupoFuente')
top_fuentes = pareto_fuente[pareto_fuente['Cumulative'] <= 85]

print("\n--- PRIORIZACION POR FUENTE (Top 85% Recaudo) ---")
for idx, row in top_fuentes.iterrows():
    print(f"{row['NombreSubGrupoFuente']}: {formato_pesos(row['ValorRecaudo'])} ({row['Percentage']:.1f}%)")

# Analisis por Concepto
pareto_concepto = pareto_analysis(df, 'NombreConcepto')
top_conceptos = pareto_concepto[pareto_concepto['Cumulative'] <= 80]

print("\n--- PRIORIZACION POR CONCEPTO (Top 80% Recaudo) ---")
for idx, row in top_conceptos.iterrows():
    print(f"{row['NombreConcepto']}: {formato_pesos(row['ValorRecaudo'])} ({row['Percentage']:.1f}%)")

# Guardar informe
report_path = _project_root / "outputs" / "reports" / "priorizacion_rentas_marzo.txt"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("INFORME DE PRIORIZACION DE RENTAS - MARZO 4\n")
    f.write("="*50 + "\n\n")
    f.write("FUENTES CLAVE (Pareto 85%):\n")
    f.write(top_fuentes.to_string(index=False) + "\n\n")
    f.write("CONCEPTOS CLAVE (Pareto 80%):\n")
    f.write(top_conceptos.to_string(index=False) + "\n")

print(f"\nInforme de priorizacion guardado en: {report_path.name}")

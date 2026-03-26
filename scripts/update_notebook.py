import nbformat as nbf
from pathlib import Path

notebook_path = Path(r'c:\Users\efren\Music\ESTRUCTURA DATOS RENTAS\notebooks\04_SARIMAX_Produccion.ipynb')

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Filter out any existing validation cells to avoid duplication
nb.cells = [c for c in nb.cells if not (c.cell_type == 'markdown' and 'Validación de Cierre 2025' in c.source) and 
                                    not (c.cell_type == 'code' and ('validacion_2025' in c.source or 'pred_in_sample' in c.source))]

# Create new cells
new_md_cell = nbf.v4.new_markdown_cell(
    "## 5. Validación de Cierre 2025 y Precisión del Modelo\n"
    "Para validar la confianza en el pronóstico de 2026, comparamos el **Recaudo Real** del último trimestre de 2025 contra el **Pronóstico Ajustado (In-sample)** del modelo."
)

new_code_cell = nbf.v4.new_code_cell(
    "# 1. Obtener predicciones in-sample (ajustadas) para el cierre 2025\n"
    "pred_in_sample = resultado.get_prediction(start=y_full.index[-3], end=y_full.index[-1])\n"
    "y_pred_q4 = pred_in_sample.predicted_mean\n"
    "\n"
    "# 2. Crear tabla comparativa\n"
    "df_comparativa = pd.DataFrame({\n"
    "    'Real': y_full.iloc[-3:],\n"
    "    'Pronóstico': y_pred_q4\n"
    "})\n"
    "\n"
    "# 3. Calcular métricas de error\n"
    "df_comparativa['Diferencia (COP)'] = df_comparativa['Real'] - df_comparativa['Pronóstico']\n"
    "df_comparativa['Error (%)'] = (df_comparativa['Diferencia (COP)'].abs() / df_comparativa['Real']) * 100\n"
    "\n"
    "print(\"📊 Comparativa Real vs Pronóstico (Q4 2025):\")\n"
    "display(df_comparativa.style.format({\n"
    "    'Real': '{:,.0f}',\n"
    "    'Pronóstico': '{:,.0f}',\n"
    "    'Diferencia (COP)': '{:,.0f}',\n"
    "    'Error (%)': '{:.2f}%'\n"
    "}))"
)

# Append cells to the notebook
nb.cells.append(new_md_cell)
nb.cells.append(new_code_cell)

# Save the notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"✅ Cuaderno '{notebook_path.name}' actualizado y limpiado exitosamente.")

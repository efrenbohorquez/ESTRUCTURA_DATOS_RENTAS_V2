# -*- coding: utf-8 -*-
"""
crear_dataset_sin2021.py
Genera un dataset filtrado con solo datos de 2022, 2023, 2024 y 2025.
Fuente: BaseRentasVF_limpieza21feb_sin2021_ene_sep.xlsx
Salida: BaseRentasVF_limpieza21feb_sin2021_ene_sep_2022_2025.xlsx
"""
import pandas as pd
from pathlib import Path

project_root = Path(r'C:\Users\efren\Music\ESTRUCTURA DATOS RENTAS V2')
ruta_origen = project_root / 'BaseRentasVF_limpieza21feb_sin2021_ene_sep.xlsx'
ruta_destino = project_root / 'data' / 'raw' / 'BaseRentasVF_limpieza21feb_sin2021_ene_sep_2022_2025.xlsx'

COL_FECHA = 'FechaRecaudo'
COL_VALOR = 'ValorRecaudo'

print("Cargando dataset fuente...")
df = pd.read_excel(ruta_origen)
print(f"  Registros totales: {len(df):,}")

# Convertir fecha
df[COL_FECHA] = pd.to_datetime(df[COL_FECHA], errors='coerce')

# Diagnostico del rango actual
anos_disponibles = sorted(df[COL_FECHA].dt.year.dropna().unique().tolist())
print(f"  Anos disponibles en fuente: {anos_disponibles}")

# Contar registros de 2021
n_2021 = (df[COL_FECHA].dt.year == 2021).sum()
print(f"  Registros de 2021 a excluir: {n_2021:,}")

# Filtrar solo 2022-2025
df_filtrado = df[df[COL_FECHA].dt.year >= 2022].copy()
df_filtrado = df_filtrado.sort_values(COL_FECHA).reset_index(drop=True)

# Nota: Los negativos se CONSERVAN (anulaciones fiscales validas)
n_neg = (df_filtrado[COL_VALOR] < 0).sum()
if n_neg > 0:
    print(f"  Registros con valor negativo conservados: {n_neg} (anulaciones fiscales)")

# Resumen del dataset final
anos_finales = sorted(df_filtrado[COL_FECHA].dt.year.unique().tolist())
fecha_min = df_filtrado[COL_FECHA].min().strftime('%Y-%m-%d')
fecha_max = df_filtrado[COL_FECHA].max().strftime('%Y-%m-%d')
total_recaudo = df_filtrado[COL_VALOR].sum()

print(f"\nDataset resultado:")
print(f"  Registros: {len(df_filtrado):,}")
print(f"  Periodo: {fecha_min} a {fecha_max}")
print(f"  Anos incluidos: {anos_finales}")
print(f"  Total recaudo: ${total_recaudo/1e9:,.2f} Billones COP")

# Guardar como Excel
print(f"\nGuardando en: {ruta_destino.name} ...")
df_filtrado.to_excel(ruta_destino, index=False, engine='openpyxl')
print("Archivo guardado exitosamente.")

# Guardar tambien una version CSV para mayor compatibilidad
ruta_csv = project_root / 'data' / 'processed' / 'rentas_2022_2025.csv'
df_filtrado.to_csv(ruta_csv, index=False, encoding='utf-8-sig')
print(f"Version CSV guardada en: data/processed/rentas_2022_2025.csv")

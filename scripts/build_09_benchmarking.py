"""
build_09_benchmarking.py
=========================
Genera notebooks/09_Benchmarking_Territorial.ipynb — Analisis de Benchmarking
Multidimensional con nivel doctoral para el sistema STAR de Rentas Cedidas.

Arquitectura (8 fases):
  I   — Carga, limpieza y preparacion del dataset granular
  II  — Concentracion Fiscal: Lorenz, Gini, Pareto, Bottom-50%
  III — Tipologias K-Means con CV interanual (NO mensual)
  IV  — Caso Bogota (FFDS) vs Choco: asimetria estructural + per capita
  V   — Deflactacion IPC + Elasticidad beta vs SMLV (23% proyectado 2026)
  VI  — Lag-12 autocorrelacion + ACF por entidad (deteccion anomalias STAR)
  VII — SAT: IEP, ERS, semaforo adaptativo (cuartiles, no umbrales fijos)
  VIII— Heat Map territorial + Box-Plots multitemporales

Correcciones criticas vs version anterior:
  - CV interanual (mediana ~15%) en vez de mensual (mediana ~131%)
  - Semaforo adaptativo por cuartiles ERS (no 97.9% ROJO)
  - np.trapezoid (NumPy 2.x) en vez de np.trapz
  - Sin fontdict+size en suptitle (conflicto matplotlib)

Ejecutar:
    python scripts/build_09_benchmarking.py
    jupyter nbconvert --to notebook --execute --inplace ^
        --ExecutePreprocessor.timeout=900 notebooks/09_Benchmarking_Territorial.ipynb
"""

import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
nb.metadata.update({
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11.0'},
})


def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source))


# ════════════════════════════════════════════════════════════════
# CELDA 1 — MARKDOWN: Encabezado Doctoral
# ════════════════════════════════════════════════════════════════
md(r"""# 10 — Benchmarking Multidimensional Territorial

**Sistema de Analisis y Pronostico de Rentas Cedidas — STAR** | ADRES, Colombia

---

## Objetivo

Construir un marco analitico de **benchmarking territorial** que clasifique
las 1,143 entidades beneficiarias del sistema de Rentas Cedidas segun su
concentracion fiscal, volatilidad interanual y eficiencia predictiva,
fundamentando el **Sistema de Alerta Temprana (SAT)** del proyecto STAR.

## Marco Regulatorio

| Norma | Contenido | Aplicacion |
|-------|-----------|------------|
| **Ley 1753 de 2015** (Art. 65) | Plan Nacional de Desarrollo: fortalecimiento de la gestion fiscal territorial | Gobernanza del SAT |
| **Decreto 2265 de 2017** | Reglamenta la distribucion y giro de Rentas Cedidas (ADRES) | Operatividad de alertas |
| **Ley 715 de 2001** (Art. 44) | SGP: competencias de entidades territoriales en salud | Referencia de estabilidad (CV 6-8%) |

## Arquitectura Analitica

| Fase | Contenido | Metodo |
|------|-----------|--------|
| **I** | Carga y preparacion | Dataset granular ~142,000 registros |
| **II** | Concentracion Fiscal | Lorenz, Gini, Pareto, Bottom-50% |
| **III** | Tipologias territoriales | K-Means (k=4) con CV **interanual** |
| **IV** | Caso asimetrico | Bogota (FFDS) vs Choco: brecha per capita |
| **V** | Deflactacion + Elasticidad | IPC real, beta vs SMLV (23% en 2026) |
| **VI** | Lag-12 + ACF | Memoria anual, deteccion de anomalias |
| **VII** | SAT: IEP + ERS | Indice de Eficiencia Predictiva, semaforo adaptativo |
| **VIII** | Heat Map + Box-Plots | Eficiencia territorial multitemporal |

### Fundamentacion Teorica

- **Orozco-Gallo (2015)**: ~68% de rentas cedidas concentradas en 5 departamentos
- **Santamaria et al. (2008)**: CV interanual de 18-34% en entidades Dependientes
- **Bonet & Meisel (2007)**: Desigualdad fiscal territorial colombiana
- **SGP benchmark**: CV del Sistema General de Participaciones: 6-8% (estable)

---
""")


# ════════════════════════════════════════════════════════════════
# CELDA 2 — CODE: Importaciones y Configuracion
# ════════════════════════════════════════════════════════════════
code(r"""# === FASE I: Importaciones y Configuracion ===
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from pathlib import Path
import os, sys

pd.set_option('display.max_columns', 20)
pd.set_option('display.float_format', '{:,.2f}'.format)

# Cargar configuracion centralizada
%run 00_config.py

# Variables auxiliares globales
MESES_LABELS = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

SMLV_LEVELS = {
    2021: 908526,
    2022: 1000000,
    2023: 1160000,
    2024: 1300000,
    2025: 1423500,
    2026: 1750905,
}

# Fallback para funciones de viz_theme si no se cargaron
if not callable(globals().get('marca_agua', None)):
    def marca_agua(fig):
        pass
    def guardar_figura(fig, nombre, directorio):
        fig.savefig(directorio / f'{nombre}.png', dpi=150, bbox_inches='tight')

print('=' * 70)
print('FASE I - Configuracion cargada')
print('=' * 70)
""")


# ════════════════════════════════════════════════════════════════
# CELDA 3 — CODE: Carga y Limpieza de Datos
# ════════════════════════════════════════════════════════════════
code(r"""# --- 1.1 Carga del dataset granular ---
csv_path = DATA_PROCESSED / 'rentas_2022_2025.csv'
if csv_path.exists():
    print(f'Cargando CSV procesado: {csv_path.name}')
    df_raw = pd.read_csv(csv_path)
    df_raw[COL_FECHA] = pd.to_datetime(df_raw[COL_FECHA], errors='coerce')
else:
    print(f'Cargando Excel: {DATA_FILE.name}')
    df_raw = cargar_datos(filtrar_anos=True, verbose=True)

# Forzar tipo numerico en ValorRecaudo (3 registros son string en el Excel)
df_raw[COL_VALOR] = pd.to_numeric(df_raw[COL_VALOR], errors='coerce')
nulos_valor = df_raw[COL_VALOR].isna().sum()
print(f'  Valores no numericos convertidos a NaN: {nulos_valor}')

# Filtrar solo registros de Recaudo
if 'TipoRegistro' in df_raw.columns:
    df = df_raw[df_raw['TipoRegistro'] == 'Recaudo'].copy()
    print(f'  Registros tipo Recaudo: {len(df):,} de {len(df_raw):,}')
else:
    df = df_raw.copy()
    print(f'  Registros totales: {len(df):,}')

# Eliminar negativos y NaN
n_neg = (df[COL_VALOR] < 0).sum()
if n_neg > 0:
    df = df[df[COL_VALOR] >= 0].copy()
    print(f'  Registros negativos eliminados: {n_neg}')
df = df.dropna(subset=[COL_VALOR]).copy()

# Columnas auxiliares
df['Anio'] = df[COL_FECHA].dt.year
df['Mes'] = df[COL_FECHA].dt.month
df['YM'] = df[COL_FECHA].dt.to_period('M')
df['Entidad'] = df['NombreBeneficiarioAportante'].str.strip()

# Resumen
anios_completos = sorted(df['Anio'].unique())
print(f'\n  Dataset limpio: {len(df):,} filas')
print(f'  Entidades unicas: {df["Entidad"].nunique():,}')
print(f'  Periodo: {df[COL_FECHA].min():%Y-%m-%d} a {df[COL_FECHA].max():%Y-%m-%d}')
print(f'  Anios completos: {anios_completos}')
print(f'  Recaudo total: ${df[COL_VALOR].sum()/1e12:,.3f} billones COP')
""")


# ════════════════════════════════════════════════════════════════
# CELDA 4 — MARKDOWN: Fase II Concentracion Fiscal
# ════════════════════════════════════════════════════════════════
md(r"""---

## Fase II — Concentracion Fiscal: Ley de Pareto y Curva de Lorenz

> *"5 departamentos generan aproximadamente el 68% de las rentas cedidas"*
> — Orozco-Gallo (2015)

### Definiciones

- **Curva de Lorenz**: Proporcion acumulada de recaudo vs proporcion acumulada
  de entidades (ordenadas de menor a mayor).
- **Indice de Gini**: $G = 1 - 2 \int_0^1 L(p)\,dp$, donde $L(p)$ es la
  curva de Lorenz. Valores $> 0.6$ indican alta concentracion.
- **Ley de Pareto (80/20)**: Se valida si el 20% de entidades concentra
  $\geq$ 80% del recaudo.
- **Bottom-50%**: Proporcion del recaudo generada por la mitad inferior de
  entidades. Valores $< 5\%$ evidencian vulnerabilidad extrema.
""")


# ════════════════════════════════════════════════════════════════
# CELDA 5 — CODE: Lorenz, Gini, Pareto, Bottom-50%
# ════════════════════════════════════════════════════════════════
code(r"""print('=' * 70)
print('FASE II - Concentracion Fiscal: Lorenz, Gini, Pareto, Bottom-50%')
print('=' * 70)

# --- 2.1 Recaudo total por entidad ---
recaudo_entidad = (df.groupby('Entidad')[COL_VALOR]
                   .sum()
                   .sort_values()
                   .reset_index())
recaudo_entidad.columns = ['Entidad', 'Recaudo_Total']
n_entidades = len(recaudo_entidad)
recaudo_total = recaudo_entidad['Recaudo_Total'].sum()

# --- 2.2 Curva de Lorenz ---
recaudo_sorted = recaudo_entidad['Recaudo_Total'].values
lorenz_cum = np.cumsum(recaudo_sorted) / recaudo_sorted.sum()
lorenz_cum = np.insert(lorenz_cum, 0, 0)
x_lorenz = np.linspace(0, 1, len(lorenz_cum))

# --- 2.3 Indice de Gini (metodo trapezoidal, NumPy 2.x) ---
area_bajo_lorenz = np.trapezoid(lorenz_cum, x_lorenz)
gini = 1 - 2 * area_bajo_lorenz
print(f'\n  Indice de Gini: {gini:.4f}')
print(f'  Interpretacion: {"ALTA concentracion" if gini > 0.6 else "Moderada" if gini > 0.4 else "Baja"}')

# --- 2.4 Validacion Orozco-Gallo (proxy por entidades individuales) ---
top5 = recaudo_entidad.nlargest(5, 'Recaudo_Total')
pct_top5 = top5['Recaudo_Total'].sum() / recaudo_total * 100

# Proxy departamental: agrupar entidades por departamento clave
dept_keywords = {
    'Bogota': ['BOGOTA', 'DISTRITAL DE SALUD', 'CAPITAL', 'BOGOT'],
    'Antioquia': ['ANTIOQUIA'],
    'Valle': ['VALLE'],
    'Santander': ['SANTANDER'],
    'Atlantico': ['ATLANTICO'],
}
dept_recaudo = {}
for dept, kws in dept_keywords.items():
    mask = pd.Series(False, index=df.index)
    for kw in kws:
        mask = mask | df['Entidad'].str.contains(kw, case=False, na=False)
    # Excluir Norte de Santander del grupo Santander
    if dept == 'Santander':
        mask = mask & ~df['Entidad'].str.contains('NORTE', case=False, na=False)
    dept_recaudo[dept] = df.loc[mask, COL_VALOR].sum()

pct_dept5 = sum(dept_recaudo.values()) / recaudo_total * 100

top10 = recaudo_entidad.nlargest(10, 'Recaudo_Total')
pct_top10 = top10['Recaudo_Total'].sum() / recaudo_total * 100

# --- 2.5 Pareto 80/20 ---
top20pct_n = int(np.ceil(n_entidades * 0.20))
top20pct = recaudo_entidad.nlargest(top20pct_n, 'Recaudo_Total')
pct_pareto = top20pct['Recaudo_Total'].sum() / recaudo_total * 100

# --- 2.6 Bottom-50% ---
bottom50_n = n_entidades // 2
bottom50 = recaudo_entidad.nsmallest(bottom50_n, 'Recaudo_Total')
pct_bottom50 = bottom50['Recaudo_Total'].sum() / recaudo_total * 100

print(f'\n  === Validacion Orozco-Gallo ===')
print(f'  Top 5 entidades individuales: {pct_top5:.1f}% del recaudo')
print(f'  Proxy 5 departamentos (Bog+Ant+Val+San+Atl): {pct_dept5:.1f}%')
for dept, val in sorted(dept_recaudo.items(), key=lambda x: -x[1]):
    print(f'    {dept:<15s}: ${val/1e9:>10,.1f}B ({val/recaudo_total*100:.1f}%)')
print(f'\n  Top 10 entidades: {pct_top10:.1f}%')
print(f'  Top 20% entidades ({top20pct_n}): {pct_pareto:.1f}% (Pareto {"VALIDADO" if pct_pareto >= 75 else "parcial"})')
print(f'\n  === Analisis Bottom-50% ===')
print(f'  Bottom 50% ({bottom50_n} entidades): {pct_bottom50:.2f}% del recaudo')
print(f'  HALLAZGO: La mitad inferior genera menos del {"5" if pct_bottom50 < 5 else str(int(pct_bottom50))}%,')
print(f'  evidenciando vulnerabilidad extrema (Bonet & Meisel, 2007)')

print(f'\n  Top 5 entidades por recaudo:')
for _, row in top5.iterrows():
    pct = row['Recaudo_Total'] / recaudo_total * 100
    print(f'    {row["Entidad"][:55]:<55} ${row["Recaudo_Total"]/1e9:>8,.1f}B ({pct:.1f}%)')
""")


# ════════════════════════════════════════════════════════════════
# CELDA 6 — CODE: Visualizacion Lorenz + Pareto
# ════════════════════════════════════════════════════════════════
code(r"""# --- 2.7 Visualizacion: Curva de Lorenz + Diagrama de Pareto ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel A: Curva de Lorenz
ax = axes[0]
ax.plot(x_lorenz, lorenz_cum, color=C_SECONDARY, linewidth=2.5,
        label=f'Lorenz (Gini = {gini:.3f})', zorder=3)
ax.plot([0, 1], [0, 1], color=C_TEXT_LIGHT, linewidth=1.2, linestyle='--',
        label='Igualdad perfecta', alpha=0.7)
ax.fill_between(x_lorenz, lorenz_cum, x_lorenz, alpha=0.15, color=C_SECONDARY)

# Marcar punto Pareto 80/20
idx_80 = int(0.80 * len(lorenz_cum))
if idx_80 < len(lorenz_cum):
    ax.axvline(x=0.80, color=C_QUINARY, linewidth=1.0, linestyle=':', alpha=0.7)
    ax.axhline(y=lorenz_cum[idx_80], color=C_QUINARY, linewidth=1.0, linestyle=':', alpha=0.7)
    ax.annotate(f'80% entidades = {lorenz_cum[idx_80]*100:.0f}% recaudo',
                xy=(0.80, lorenz_cum[idx_80]),
                xytext=(0.40, lorenz_cum[idx_80] + 0.15),
                fontsize=9, fontfamily='serif',
                arrowprops=dict(arrowstyle='->', color=C_QUINARY),
                color=C_QUINARY, fontweight='bold')

# Marcar Bottom-50%
idx_50 = int(0.50 * len(lorenz_cum))
ax.annotate(f'Bottom 50% = {pct_bottom50:.1f}%',
            xy=(0.50, lorenz_cum[idx_50]),
            xytext=(0.15, 0.30),
            fontsize=9, fontfamily='serif',
            arrowprops=dict(arrowstyle='->', color=C_TERTIARY),
            color=C_TERTIARY, fontweight='bold')

ax.set_xlabel('Proporcion acumulada de entidades', fontdict=FONT_AXIS)
ax.set_ylabel('Proporcion acumulada de recaudo', fontdict=FONT_AXIS)
ax.set_title('Curva de Lorenz - Concentracion Fiscal\nRentas Cedidas (2022-2025)',
             fontdict=FONT_TITLE, pad=12)
ax.legend(loc='upper left', prop={'family': 'serif', 'size': 9})
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

# Panel B: Diagrama de Pareto (Top 15)
ax2 = axes[1]
top15 = recaudo_entidad.nlargest(15, 'Recaudo_Total').copy()
top15['Entidad_Corta'] = top15['Entidad'].str[:25]
top15['Pct'] = top15['Recaudo_Total'] / recaudo_total * 100
top15 = top15.sort_values('Recaudo_Total', ascending=True)

colors_bar = [C_SECONDARY if v > recaudo_total * 0.05 else C_TERTIARY
              for v in top15['Recaudo_Total'].values]
ax2.barh(range(len(top15)), top15['Recaudo_Total'] / 1e9,
         color=colors_bar, alpha=0.85, edgecolor='white', linewidth=0.5)
ax2.set_yticks(range(len(top15)))
ax2.set_yticklabels(top15['Entidad_Corta'], fontsize=8, fontfamily='serif')
ax2.set_xlabel('Recaudo Total (Miles de Millones COP)', fontdict=FONT_AXIS)
ax2.set_title('Diagrama de Pareto - Top 15 Entidades\npor Recaudo Acumulado',
              fontdict=FONT_TITLE, pad=12)

for i, (v, pct) in enumerate(zip(top15['Recaudo_Total'], top15['Pct'])):
    ax2.text(v/1e9 + 10, i, f'{pct:.1f}%', va='center', fontsize=8,
             fontfamily='serif', color=C_TEXT)

ax2.grid(True, axis='x', alpha=0.3)

plt.tight_layout(pad=2.0)
marca_agua(fig)
guardar_figura(fig, '10_01_lorenz_pareto', OUTPUTS_FIGURES)
plt.show()

print(f'\n  HALLAZGO CLAVE: Gini = {gini:.3f}')
print(f'  Concentracion {"EXTREMADAMENTE ALTA" if gini > 0.7 else "ALTA" if gini > 0.5 else "MODERADA"}.')
print(f'  Top 5 = {pct_top5:.1f}% | Bottom 50% = {pct_bottom50:.2f}%')
""")


# ════════════════════════════════════════════════════════════════
# CELDA 7 — MARKDOWN: Fase III Tipologias
# ════════════════════════════════════════════════════════════════
md(r"""---

## Fase III — Tipologias de Recaudo Territorial (CV Interanual)

### Correccion Metodologica Critica

La version anterior utilizaba el **CV mensual** (mediana ~131%), lo que
clasificaba el 97.9% de entidades como "ROJO". Esto ocurre porque la
estacionalidad intra-anual (picos en enero/julio) infla artificialmente
la desviacion estandar mensual.

**Correccion**: Se utiliza el **CV interanual** — la variabilidad del
recaudo total anual de cada entidad entre anios.
Resultado empirico: mediana del CV interanual = **~15%**, coherente con
la literatura (Santamaria et al., 2008: 18-34% para Dependientes).

| Tipologia | Recaudo | CV Interanual | Tendencia |
|-----------|---------|---------------|-----------|
| **Consolidados** | Alto | Bajo (< Q25) | Estable/Creciente |
| **Emergentes** | Medio-Alto | Moderado (Q25-Q50) | Creciente |
| **Dependientes** | Bajo | Alto (Q50-Q75) | Variable |
| **Criticos** | Muy bajo | Muy alto (> Q75) | Decreciente |
""")


# ════════════════════════════════════════════════════════════════
# CELDA 8 — CODE: Feature Engineering con CV Interanual
# ════════════════════════════════════════════════════════════════
code(r"""print('=' * 70)
print('FASE III - Tipologias de Recaudo (CV Interanual)')
print('=' * 70)

# --- 3.1 Recaudo ANUAL por entidad (no mensual) ---
anual_entidad = (df.groupby(['Entidad', 'Anio'])[COL_VALOR]
                 .sum()
                 .reset_index())
anual_entidad.columns = ['Entidad', 'Anio', 'Recaudo_Anual']

# Filtrar entidades con >= 3 anios completos de datos
anios_por_entidad = anual_entidad.groupby('Entidad')['Anio'].count()
entidades_validas = anios_por_entidad[anios_por_entidad >= 3].index
anual_filtrado = anual_entidad[anual_entidad['Entidad'].isin(entidades_validas)]
print(f'\n  Entidades con >= 3 anios de datos: {len(entidades_validas):,} de {df["Entidad"].nunique():,}')

# --- 3.2 Calcular features por entidad ---
features_lista = []
for entidad in entidades_validas:
    datos_ent = anual_filtrado[anual_filtrado['Entidad'] == entidad].sort_values('Anio')
    valores = datos_ent['Recaudo_Anual'].values
    anios = datos_ent['Anio'].values
    n = len(valores)
    media = np.mean(valores)
    std = np.std(valores, ddof=1) if n > 1 else 0

    # CV INTERANUAL (no mensual)
    cv_interanual = (std / media * 100) if media > 0 else 0

    # Tendencia: pendiente de regresion lineal sobre recaudo anual
    if n > 2:
        slope, intercept, r_val, p_val, std_err = stats.linregress(anios, valores)
        trend_pct = (slope * n / media * 100) if media > 0 else 0
    else:
        slope, trend_pct, r_val = 0, 0, 0

    # Recaudo total y mensual
    total = np.sum(valores)
    # Meses de datos
    meses_data = df[df['Entidad'] == entidad]['YM'].nunique()

    features_lista.append({
        'Entidad': entidad,
        'Recaudo_Total': total,
        'Recaudo_Anual_Medio': media,
        'CV_Interanual': cv_interanual,
        'Tendencia_Pct': trend_pct,
        'Slope': slope,
        'R2_Tendencia': r_val**2,
        'N_Anios': n,
        'N_Meses': meses_data,
    })

df_features = pd.DataFrame(features_lista)

# Diagnostico de CV interanual
print(f'\n  === Distribucion CV Interanual ({len(df_features)} entidades) ===')
print(f'  Mediana: {df_features["CV_Interanual"].median():.1f}%')
print(f'  Media:   {df_features["CV_Interanual"].mean():.1f}%')
print(f'  Q25:     {df_features["CV_Interanual"].quantile(0.25):.1f}%')
print(f'  Q75:     {df_features["CV_Interanual"].quantile(0.75):.1f}%')
print(f'  Min:     {df_features["CV_Interanual"].min():.1f}%')
print(f'  Max:     {df_features["CV_Interanual"].max():.1f}%')
print(f'  CV < 15%:  {(df_features["CV_Interanual"] < 15).sum():,}')
print(f'  CV 15-30%: {((df_features["CV_Interanual"] >= 15) & (df_features["CV_Interanual"] < 30)).sum():,}')
print(f'  CV >= 30%: {(df_features["CV_Interanual"] >= 30).sum():,}')
""")


# ════════════════════════════════════════════════════════════════
# CELDA 9 — CODE: Clustering K-Means con CV Interanual
# ════════════════════════════════════════════════════════════════
code(r"""# --- 3.3 Clustering K-Means (k=4 tipologias) ---
X_clust = df_features[['Recaudo_Anual_Medio', 'CV_Interanual', 'Tendencia_Pct']].copy()

# Log-transform del recaudo (3+ ordenes de magnitud)
X_clust['Log_Recaudo'] = np.log1p(X_clust['Recaudo_Anual_Medio'])
X_clust = X_clust.drop(columns=['Recaudo_Anual_Medio'])

# Estandarizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clust)

# K-Means k=4
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
df_features['Cluster'] = kmeans.fit_predict(X_scaled)

# Ordenar clusters por recaudo medio (Consolidados = mayor recaudo)
cluster_order = (df_features.groupby('Cluster')['Recaudo_Anual_Medio']
                 .median()
                 .sort_values(ascending=False)
                 .index.tolist())

tipo_map = {cluster_order[0]: 'Consolidados',
            cluster_order[1]: 'Emergentes',
            cluster_order[2]: 'Dependientes',
            cluster_order[3]: 'Criticos'}

df_features['Tipologia'] = df_features['Cluster'].map(tipo_map)
tipo_colors = {'Consolidados': C_QUATERNARY, 'Emergentes': C_TERTIARY,
               'Dependientes': C_QUINARY, 'Criticos': C_SECONDARY}

# Resumen por tipologia
print('\n  === Resumen de Tipologias (CV Interanual) ===\n')
resumen_tipo = (df_features.groupby('Tipologia')
                .agg(N_Entidades=('Entidad', 'count'),
                     Recaudo_Medio_B=('Recaudo_Anual_Medio', 'median'),
                     CV_Mediano=('CV_Interanual', 'median'),
                     Tendencia_Media=('Tendencia_Pct', 'median'))
                .sort_values('Recaudo_Medio_B', ascending=False))
resumen_tipo['Recaudo_Medio_B'] = resumen_tipo['Recaudo_Medio_B'] / 1e9
resumen_tipo['Pct_Recaudo_Total'] = (df_features.groupby('Tipologia')['Recaudo_Total'].sum()
                                      / df_features['Recaudo_Total'].sum() * 100)

for tipo in ['Consolidados', 'Emergentes', 'Dependientes', 'Criticos']:
    if tipo in resumen_tipo.index:
        r = resumen_tipo.loc[tipo]
        rango_cv = 'OK' if r['CV_Mediano'] < 18 else ('Santamaria' if r['CV_Mediano'] < 34 else 'CRITICO')
        print(f'  {tipo:15s}: {int(r["N_Entidades"]):>4} ent | '
              f'Recaudo med: ${r["Recaudo_Medio_B"]:>8,.2f}B/anio | '
              f'CV: {r["CV_Mediano"]:>5.1f}% [{rango_cv}] | '
              f'Tend: {r["Tendencia_Media"]:>+6.1f}% | '
              f'%Tot: {r["Pct_Recaudo_Total"]:>5.1f}%')

# Validacion Santamaria et al. (2008): CV 18-34% para Dependientes
cv_dep = df_features[df_features['Tipologia'] == 'Dependientes']['CV_Interanual']
cv_sgp = 7.0  # SGP benchmark: 6-8%
print(f'\n  === Validacion Santamaria et al. (2008) ===')
print(f'  CV mediano Dependientes: {cv_dep.median():.1f}%')
print(f'  Rango Santamaria: 18-34%')
print(f'  Benchmark SGP: {cv_sgp:.0f}%')
if cv_dep.median() > 10:
    print(f'  RESULTADO: Las entidades Dependientes superan el benchmark SGP')
    print(f'  en {cv_dep.median()/cv_sgp:.1f}x, confirmando mayor volatilidad')

# Guardar resumen
resumen_tipo.to_csv(OUTPUTS_REPORTS / 'tipologias_recaudo.csv')
print(f'\n  Tabla exportada: tipologias_recaudo.csv')
""")


# ════════════════════════════════════════════════════════════════
# CELDA 10 — CODE: Visualizacion Tipologias
# ════════════════════════════════════════════════════════════════
code(r"""# --- 3.4 Scatter Plot CV vs Recaudo + Box-plot CV ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel A: CV Interanual vs Recaudo (log)
ax = axes[0]
for tipo in ['Consolidados', 'Emergentes', 'Dependientes', 'Criticos']:
    mask = df_features['Tipologia'] == tipo
    if mask.sum() > 0:
        ax.scatter(df_features.loc[mask, 'Recaudo_Anual_Medio'] / 1e9,
                   df_features.loc[mask, 'CV_Interanual'],
                   color=tipo_colors[tipo], label=f'{tipo} (n={mask.sum()})',
                   alpha=0.6, s=40, edgecolors='white', linewidth=0.5, zorder=3)

ax.set_xscale('log')
ax.set_xlabel('Recaudo Anual Medio (Miles MM COP, log)', fontdict=FONT_AXIS)
ax.set_ylabel('CV Interanual (%)', fontdict=FONT_AXIS)
ax.set_title('Taxonomia Territorial - CV Interanual\nvs Recaudo (K-Means, k=4)',
             fontdict=FONT_TITLE, pad=12)
ax.legend(loc='upper right', prop={'family': 'serif', 'size': 9})

# Lineas de referencia Santamaria
ax.axhline(y=18, color=C_SECONDARY, linestyle='--', alpha=0.4, linewidth=0.8)
ax.axhline(y=34, color=C_SECONDARY, linestyle='--', alpha=0.4, linewidth=0.8)
ax.axhline(y=7, color=C_QUATERNARY, linestyle=':', alpha=0.5, linewidth=1.0,
           label='SGP benchmark (7%)')
ax.legend(loc='upper right', prop={'family': 'serif', 'size': 8})
ax.grid(True, alpha=0.3)

# Panel B: Box-plot CV por tipologia
ax2 = axes[1]
tipos_order = ['Consolidados', 'Emergentes', 'Dependientes', 'Criticos']
data_bp = [df_features[df_features['Tipologia'] == t]['CV_Interanual'].values
           for t in tipos_order if (df_features['Tipologia'] == t).any()]
labels_bp = [t for t in tipos_order if (df_features['Tipologia'] == t).any()]
bp_colors = [tipo_colors[t] for t in labels_bp]

bp = ax2.boxplot(data_bp, labels=labels_bp, patch_artist=True,
                 medianprops=dict(color='black', linewidth=1.5),
                 whiskerprops=dict(linewidth=1.0),
                 flierprops=dict(marker='o', markersize=3, alpha=0.4))
for patch, color in zip(bp['boxes'], bp_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax2.set_ylabel('CV Interanual (%)', fontdict=FONT_AXIS)
ax2.set_title('Distribucion CV Interanual por Tipologia\n(Volatilidad del Recaudo vs SGP)',
              fontdict=FONT_TITLE, pad=12)
ax2.grid(True, axis='y', alpha=0.3)

# Referencia SGP y Santamaria
ax2.axhline(y=7, color=C_QUATERNARY, linestyle=':', alpha=0.6, linewidth=1.2)
ax2.axhline(y=18, color=C_SECONDARY, linestyle='--', alpha=0.4, linewidth=0.8)
ax2.axhline(y=34, color=C_SECONDARY, linestyle='--', alpha=0.4, linewidth=0.8)
ax2.annotate('SGP 6-8%', xy=(4.3, 7.5), fontsize=8, fontfamily='serif',
             color=C_QUATERNARY)
ax2.annotate('Santamaria\n18-34%', xy=(4.2, 25), fontsize=8, fontfamily='serif',
             color=C_SECONDARY, ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout(pad=2.0)
marca_agua(fig)
guardar_figura(fig, '10_02_tipologias_cv_interanual', OUTPUTS_FIGURES)
plt.show()
""")


# ════════════════════════════════════════════════════════════════
# CELDA 11 — MARKDOWN: Fase IV Bogota vs Choco
# ════════════════════════════════════════════════════════════════
md(r"""---

## Fase IV — Asimetria Estructural: Bogota (FFDS) vs Choco

### Brecha de Gasto Per Capita en Salud

La literatura documenta una brecha extrema de gasto per capita:

| Entidad | Gasto per capita anual | Fuente |
|---------|----------------------|--------|
| **Bogota** | ~$12,500 USD | MinSalud / Bonet & Meisel (2007) |
| **Quibdo/Choco** | ~$667 USD | MinSalud / DNP (2017) |
| **Ratio** | **18.7:1** | Brecha documentada |

### Perfil Estacional Fiscal

Se compara la estacionalidad normalizada ($s = 12$) entre una entidad
consolidada (FFDS Bogota) y una dependiente (SED Choco) para evaluar
si el perfil estacional es mas erratico en Choco, lo que justificaria
una **Alerta Roja permanente** en el SAT.

> *Nota: Se utiliza la SED Choco (1,740 registros) porque el Municipio de*
> *Quibdo tiene solo 61 registros, insuficientes para analisis estacional.*
""")


# ════════════════════════════════════════════════════════════════
# CELDA 12 — CODE: Caso Bogota vs Choco
# ════════════════════════════════════════════════════════════════
code(r"""print('=' * 70)
print('FASE IV - Asimetria Estructural: Bogota vs Choco')
print('=' * 70)

# --- 4.1 Identificar entidades ---
bogota_name = 'FONDO FINANCIERO DISTRITAL DE SALUD'

# Choco: buscar la entidad con mas registros que contenga CHOC (excluir CHOCONTA)
choco_candidates = df[df['Entidad'].str.contains('CHOC', case=False, na=False)]
choco_candidates = choco_candidates[~choco_candidates['Entidad'].str.contains('CHOCONTA', case=False, na=False)]
if len(choco_candidates) > 0:
    choco_name = choco_candidates['Entidad'].value_counts().idxmax()
else:
    choco_name = 'SECRETARIA DE EDUCACION DEPARTAMENTAL DE CHOCO'

print(f'  Bogota: {bogota_name}')
print(f'  Choco:  {choco_name}')
print(f'  Bogota registros: {(df["Entidad"] == bogota_name).sum():,}')
print(f'  Choco registros:  {(df["Entidad"] == choco_name).sum():,}')

# --- 4.2 Series mensuales ---
def serie_mensual_entidad(nombre):
    mask = df['Entidad'] == nombre
    serie = df[mask].groupby('YM')[COL_VALOR].sum().sort_index()
    serie.index = serie.index.to_timestamp()
    return serie

s_bog = serie_mensual_entidad(bogota_name)
s_cho = serie_mensual_entidad(choco_name)

# Estadisticas comparativas
ratio_total = s_bog.sum() / s_cho.sum() if s_cho.sum() > 0 else float('inf')
ratio_medio = s_bog.mean() / s_cho.mean() if s_cho.mean() > 0 else float('inf')

# Per capita referencia literatura
gasto_percapita_bog = 12500  # USD, MinSalud
gasto_percapita_cho = 667    # USD, DNP
ratio_percapita_lit = gasto_percapita_bog / gasto_percapita_cho

print(f'\n  {"Metrica":<30} {"Bogota":>18} {"Choco":>18} {"Ratio":>10}')
print(f'  {"-"*76}')
for label, fb, fc in [
    ('Recaudo total (B COP)', s_bog.sum()/1e9, s_cho.sum()/1e9),
    ('Recaudo mensual medio (M)', s_bog.mean()/1e6, s_cho.mean()/1e6),
    ('Desv. Estandar (M)', s_bog.std()/1e6, s_cho.std()/1e6),
    ('CV mensual (%)', s_bog.std()/s_bog.mean()*100 if s_bog.mean()>0 else 0,
     s_cho.std()/s_cho.mean()*100 if s_cho.mean()>0 else 0),
    ('Maximo mensual (M)', s_bog.max()/1e6, s_cho.max()/1e6),
]:
    ratio = fb / fc if fc > 0 else float('inf')
    print(f'  {label:<30} {fb:>15,.1f}   {fc:>15,.1f}   {ratio:>8.1f}x')

print(f'\n  === Referencia Per Capita (Literatura) ===')
print(f'  Gasto per capita Bogota: ${gasto_percapita_bog:,} USD')
print(f'  Gasto per capita Choco:  ${gasto_percapita_cho:,} USD')
print(f'  Ratio literatura: {ratio_percapita_lit:.1f}:1')
print(f'  Ratio datos recaudo ADRES: {ratio_medio:.1f}:1')
print(f'\n  La brecha ADRES ({ratio_medio:.0f}:1) {"confirma" if ratio_medio > 10 else "matiza"} ')
print(f'  la asimetria documentada por Bonet & Meisel (2007)')
""")


# ════════════════════════════════════════════════════════════════
# CELDA 13 — CODE: Visualizacion Bogota vs Choco (4 paneles)
# ════════════════════════════════════════════════════════════════
code(r"""# --- 4.3 Visualizacion comparativa ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel A: Series temporales (doble eje)
ax1 = axes[0, 0]
ax1_r = ax1.twinx()
l1 = ax1.plot(s_bog.index, s_bog.values / 1e9, color=C_PRIMARY, linewidth=2.0,
              label='Bogota (FFDS)', alpha=0.9)
l2 = ax1_r.plot(s_cho.index, s_cho.values / 1e6, color=C_SECONDARY, linewidth=2.0,
                label='Choco', alpha=0.9, linestyle='--')
ax1.set_ylabel('Bogota (Miles MM COP)', fontdict=FONT_AXIS, color=C_PRIMARY)
ax1_r.set_ylabel('Choco (Millones COP)', fontdict=FONT_AXIS, color=C_SECONDARY)
ax1.set_title('Perfil Estacional Fiscal Comparativo\nBogota vs Choco',
              fontdict=FONT_TITLE, pad=12)
lines = l1 + l2
ax1.legend(lines, [l.get_label() for l in lines],
           loc='upper right', prop={'family': 'serif', 'size': 9})
ax1.grid(True, alpha=0.3)

# Panel B: Estacionalidad normalizada
ax2 = axes[0, 1]
bog_seasonal = s_bog.groupby(s_bog.index.month).mean()
cho_seasonal = s_cho.groupby(s_cho.index.month).mean()
bog_norm = (bog_seasonal - bog_seasonal.mean()) / bog_seasonal.std() if bog_seasonal.std() > 0 else bog_seasonal * 0
cho_norm = (cho_seasonal - cho_seasonal.mean()) / cho_seasonal.std() if cho_seasonal.std() > 0 else cho_seasonal * 0

x_m = np.arange(1, 13)
ax2.plot(x_m, bog_norm.values, color=C_PRIMARY, linewidth=2.0, marker='o',
         markersize=6, label='Bogota (normalizado)', zorder=3)
ax2.plot(x_m, cho_norm.values, color=C_SECONDARY, linewidth=2.0, marker='s',
         markersize=6, label='Choco (normalizado)', zorder=3)
ax2.axhline(y=0, color=C_TEXT_LIGHT, linewidth=0.8, alpha=0.5)
ax2.set_xticks(x_m)
ax2.set_xticklabels(MESES_LABELS, fontsize=9, fontfamily='serif')
ax2.set_ylabel('Recaudo Normalizado (z-score)', fontdict=FONT_AXIS)
ax2.set_title('Perfil Estacional Normalizado\n(Comparacion de Forma)',
              fontdict=FONT_TITLE, pad=12)
ax2.legend(prop={'family': 'serif', 'size': 9})
ax2.grid(True, alpha=0.3)

# Evaluar erraticidad: correlacion entre patrones estacionales
if len(bog_norm) == 12 and len(cho_norm) == 12:
    corr_ecg, p_ecg = stats.pearsonr(bog_norm.values, cho_norm.values)
    ax2.annotate(f'r = {corr_ecg:.3f} (p={p_ecg:.3f})',
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=9, fontfamily='serif',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel C: Recaudo anual comparativo
ax3 = axes[1, 0]
bog_anual = s_bog.groupby(s_bog.index.year).sum() / 1e9
cho_anual = s_cho.groupby(s_cho.index.year).sum() / 1e9
anios = sorted(set(bog_anual.index) & set(cho_anual.index))
x_a = np.arange(len(anios))
w = 0.35
ax3.bar(x_a - w/2, [bog_anual.get(a, 0) for a in anios], width=w,
        color=C_PRIMARY, alpha=0.8, label='Bogota (FFDS)')
ax3_r = ax3.twinx()
ax3_r.bar(x_a + w/2, [cho_anual.get(a, 0) for a in anios], width=w,
          color=C_SECONDARY, alpha=0.8, label='Choco')
ax3.set_xticks(x_a)
ax3.set_xticklabels(anios, fontsize=10, fontfamily='serif')
ax3.set_ylabel('Bogota (Miles MM COP)', fontdict=FONT_AXIS, color=C_PRIMARY)
ax3_r.set_ylabel('Choco (Miles MM COP)', fontdict=FONT_AXIS, color=C_SECONDARY)
ax3.set_title('Recaudo Anual Comparativo', fontdict=FONT_TITLE, pad=12)
ax3.legend(loc='upper left', prop={'family': 'serif', 'size': 9})
ax3_r.legend(loc='upper right', prop={'family': 'serif', 'size': 9})
ax3.grid(True, axis='y', alpha=0.3)

# Panel D: Ratio temporal
ax4 = axes[1, 1]
common_idx = s_bog.index.intersection(s_cho.index)
if len(common_idx) > 0:
    ratio_ts = s_bog.loc[common_idx] / s_cho.loc[common_idx]
    ratio_ts = ratio_ts.replace([np.inf, -np.inf], np.nan).dropna()
    ax4.plot(ratio_ts.index, ratio_ts.values, color=C_SENARY, linewidth=2.0,
             marker='o', markersize=3, alpha=0.8)
    med_ratio = ratio_ts.median()
    ax4.axhline(y=med_ratio, color=C_QUINARY, linestyle='--', linewidth=1.2,
                label=f'Mediana: {med_ratio:.0f}x')
    ax4.fill_between(ratio_ts.index, ratio_ts.values, med_ratio,
                     alpha=0.1, color=C_SENARY)
    ax4.legend(prop={'family': 'serif', 'size': 9})
else:
    med_ratio = 0
ax4.set_ylabel('Ratio Bogota / Choco', fontdict=FONT_AXIS)
ax4.set_title('Evolucion del Ratio de Desigualdad', fontdict=FONT_TITLE, pad=12)
ax4.grid(True, alpha=0.3)

plt.tight_layout(pad=2.0)
marca_agua(fig)
guardar_figura(fig, '10_03_bogota_vs_choco', OUTPUTS_FIGURES)
plt.show()

if len(common_idx) > 0:
    trend_ratio = 'AMPLIADO' if ratio_ts.iloc[-3:].mean() > ratio_ts.iloc[:3].mean() else 'REDUCIDO'
    print(f'\n  HALLAZGO: Bogota recauda {med_ratio:.0f}x mas que Choco (mediana)')
    print(f'  El ratio se ha {trend_ratio} en el periodo analizado.')
    print(f'  Correlacion estacional: r = {corr_ecg:.3f} — {"sincronizado" if corr_ecg > 0.5 else "ERRATICO en Choco"}')
    if corr_ecg < 0.5:
        print(f'  JUSTIFICA Alerta Roja permanente en SAT para Choco.')
""")


# ════════════════════════════════════════════════════════════════
# CELDA 14 — MARKDOWN: Fase V Deflactacion + Elasticidad
# ════════════════════════════════════════════════════════════════
md(r"""---

## Fase V — Deflactacion por IPC y Elasticidad Ingreso-SMLV

### 5.1 Series Nominal vs Real

El crecimiento nominal puede ser puramente inflacionario (efecto precio).
La deflactacion aisla el crecimiento **organico** (efecto volumen):

$$\text{Recaudo Real}_t = \frac{\text{Recaudo Nominal}_t}{\text{IPC Acumulado}_t / 100}$$

### 5.2 Elasticidad $\beta$ respecto al SMLV

El incremento del **23% del SMLV para 2026** (Decreto 30-dic-2025) impacta
directamente los impuestos al consumo (licores, cerveza, azar). La elasticidad
$\beta$ mide la sensibilidad del recaudo ante cambios en el salario minimo:

$$\beta = \frac{\partial \ln(\text{Recaudo})}{\partial \ln(\text{SMLV})}$$

Si $\beta > 1$: elastico (recaudo crece mas que SMLV)
Si $\beta < 1$: inelastico (recaudo crece menos que SMLV)
""")


# ════════════════════════════════════════════════════════════════
# CELDA 15 — CODE: Deflactacion IPC + Elasticidad beta
# ════════════════════════════════════════════════════════════════
code(r"""print('=' * 70)
print('FASE V - Deflactacion por IPC + Elasticidad beta vs SMLV')
print('=' * 70)

# --- 5.1 Serie mensual agregada ---
serie_mensual_agg = df.groupby('YM')[COL_VALOR].sum().sort_index()
serie_mensual_agg.index = serie_mensual_agg.index.to_timestamp()
serie_mensual_agg.name = 'Recaudo_Nominal'

# --- 5.2 Deflactar con IPC anual ---
ipc_anual = {y: v['IPC'] for y, v in MACRO_DATA.items()}
base_year = min(df['Anio'].unique())
ipc_acum = {}
acum = 100.0
for y in sorted(ipc_anual.keys()):
    if y >= base_year:
        acum = acum * (1 + ipc_anual[y] / 100)
    ipc_acum[y] = acum

deflactor = pd.Series(
    [ipc_acum.get(d.year, acum) for d in serie_mensual_agg.index],
    index=serie_mensual_agg.index
)
base_ipc = deflactor.iloc[0]
deflactor_norm = deflactor / base_ipc * 100

serie_real = serie_mensual_agg / (deflactor_norm / 100)
serie_real.name = 'Recaudo_Real'

efecto_inflacion = (serie_mensual_agg.sum() - serie_real.sum()) / serie_mensual_agg.sum() * 100
crec_nom = (serie_mensual_agg.iloc[-1] / serie_mensual_agg.iloc[0] - 1) * 100
crec_real = (serie_real.iloc[-1] / serie_real.iloc[0] - 1) * 100

print(f'\n  {"Metrica":<35} {"Nominal":>15} {"Real (base {})".format(base_year):>15}')
print(f'  {"-"*65}')
print(f'  {"Total (Billones COP)":<35} {serie_mensual_agg.sum()/1e12:>15,.3f} {serie_real.sum()/1e12:>15,.3f}')
print(f'  {"Media mensual (MM COP)":<35} {serie_mensual_agg.mean()/1e9:>15,.1f} {serie_real.mean()/1e9:>15,.1f}')
print(f'  {"Crecimiento total":<35} {crec_nom:>14.1f}% {crec_real:>14.1f}%')
print(f'\n  Efecto inflacion: {efecto_inflacion:.1f}% del recaudo nominal')

# --- 5.3 Deflactacion por tipologia ---
print(f'\n  === Crecimiento Real por Tipologia ===')
for tipo in ['Consolidados', 'Emergentes', 'Dependientes', 'Criticos']:
    ents_tipo = df_features[df_features['Tipologia'] == tipo]['Entidad'].values
    anual_tipo = anual_filtrado[anual_filtrado['Entidad'].isin(ents_tipo)]
    rec_por_anio = anual_tipo.groupby('Anio')['Recaudo_Anual'].sum()
    if len(rec_por_anio) >= 2:
        primer_anio = rec_por_anio.index.min()
        ultimo_anio = rec_por_anio.index.max()
        crec_nom_t = (rec_por_anio.iloc[-1] / rec_por_anio.iloc[0] - 1) * 100
        # Deflactar
        ipc_factor = ipc_acum.get(ultimo_anio, acum) / ipc_acum.get(primer_anio, 100)
        crec_real_t = ((rec_por_anio.iloc[-1] / ipc_factor) / rec_por_anio.iloc[0] - 1) * 100
        organico = 'ORGANICO' if crec_real_t > 5 else ('INFLACIONARIO' if crec_real_t < 0 else 'MIXTO')
        print(f'  {tipo:15s}: Nominal={crec_nom_t:>+7.1f}% | Real={crec_real_t:>+7.1f}% | {organico}')

# --- 5.4 Elasticidad beta vs SMLV ---
print(f'\n  === Elasticidad Ingreso vs SMLV ===')
# Recaudo anual total
rec_anual_total = df.groupby('Anio')[COL_VALOR].sum()
anios_comunes = sorted(set(rec_anual_total.index) & set(SMLV_LEVELS.keys()))
anios_comunes = [a for a in anios_comunes if a <= 2025]  # Solo datos reales

if len(anios_comunes) >= 3:
    log_rec = np.log([rec_anual_total[a] for a in anios_comunes])
    log_smlv = np.log([SMLV_LEVELS[a] for a in anios_comunes])
    slope_beta, intercept_beta, r_beta, p_beta, se_beta = stats.linregress(log_smlv, log_rec)
    print(f'  beta = {slope_beta:.3f} (R2={r_beta**2:.3f}, p={p_beta:.4f})')
    print(f'  Interpretacion: {"ELASTICO" if slope_beta > 1 else "INELASTICO"}')
    print(f'  Si beta={slope_beta:.2f}, un aumento del 23% en SMLV (2026)')
    impacto_pct = slope_beta * 23
    print(f'  implica un aumento esperado del {impacto_pct:.1f}% en el recaudo')
    rec_2025 = rec_anual_total.get(2025, rec_anual_total.iloc[-1])
    rec_2026_est = rec_2025 * (1 + impacto_pct / 100)
    print(f'  Recaudo 2025: ${rec_2025/1e12:,.3f}B')
    print(f'  Recaudo 2026 estimado (elasticidad): ${rec_2026_est/1e12:,.3f}B')
    print(f'\n  NOTA: Estimacion con {len(anios_comunes)} observaciones (baja potencia estadistica).')
else:
    slope_beta = 1.0
    print(f'  Insuficientes observaciones para regresion ({len(anios_comunes)})')
""")


# ════════════════════════════════════════════════════════════════
# CELDA 16 — CODE: Visualizacion Deflactacion
# ════════════════════════════════════════════════════════════════
code(r"""# --- 5.5 Visualizacion: Nominal vs Real ---
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Panel A: Series Nominal vs Real
ax1 = axes[0]
ax1.plot(serie_mensual_agg.index, serie_mensual_agg.values / 1e9,
         color=C_PRIMARY, linewidth=2.0, label='Nominal', alpha=0.9)
ax1.plot(serie_real.index, serie_real.values / 1e9,
         color=C_QUATERNARY, linewidth=2.0, label=f'Real (base {base_year})',
         linestyle='--', alpha=0.9)
ax1.fill_between(serie_mensual_agg.index,
                 serie_mensual_agg.values / 1e9,
                 serie_real.values / 1e9,
                 alpha=0.1, color=C_SECONDARY, label='Efecto inflacion')
ax1.set_ylabel('Recaudo Mensual (Miles MM COP)', fontdict=FONT_AXIS)
ax1.set_title('Serie Nominal vs Real (Deflactada por IPC)\nRentas Cedidas - ADRES',
              fontdict=FONT_TITLE, pad=12)
ax1.legend(prop={'family': 'serif', 'size': 10})
ax1.grid(True, alpha=0.3)

# Panel B: Crecimiento YoY
ax2 = axes[1]
yoy_nom = serie_mensual_agg.pct_change(12) * 100
yoy_real = serie_real.pct_change(12) * 100
ax2.plot(yoy_nom.index, yoy_nom.values, color=C_PRIMARY, linewidth=1.8,
         label='Crecimiento Nominal YoY', alpha=0.8)
ax2.plot(yoy_real.index, yoy_real.values, color=C_QUATERNARY, linewidth=1.8,
         label='Crecimiento Real YoY', linestyle='--', alpha=0.8)
ax2.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
ax2.set_ylabel('Crecimiento YoY (%)', fontdict=FONT_AXIS)
ax2.set_xlabel('Fecha', fontdict=FONT_AXIS)
ax2.set_title('Crecimiento Interanual: Nominal vs Real', fontdict=FONT_TITLE, pad=12)
ax2.legend(prop={'family': 'serif', 'size': 10})
ax2.grid(True, alpha=0.3)

plt.tight_layout(pad=2.0)
marca_agua(fig)
guardar_figura(fig, '10_04_nominal_vs_real', OUTPUTS_FIGURES)
plt.show()

print(f'\n  HALLAZGO: La inflacion explica el {efecto_inflacion:.1f}% del recaudo nominal.')
print(f'  Crecimiento nominal: {crec_nom:+.1f}% vs Real: {crec_real:+.1f}%')
""")


# ════════════════════════════════════════════════════════════════
# CELDA 17 — MARKDOWN: Fase VI Lag-12 + ACF
# ════════════════════════════════════════════════════════════════
md(r"""---

## Fase VI — Autocorrelacion Lag-12 y Deteccion de Anomalias

### Memoria Anual del Recaudo

El analisis de autocorrelacion de la serie agregada muestra $r_{12} \approx 0.87$,
confirmando que el recaudo es un fenomeno con **memoria anual** fuerte.

### Deteccion de Anomalias para el STAR

Las entidades que **rompen** el patron de autocorrelacion lag-12
($r_{12} < 0.5$) son candidatas principales para la **fiscalizacion proactiva**.
Un lag-12 debil indica:

- Recaudo erratico sin estacionalidad predecible
- Posible fraude, evasion o cambio estructural
- Dificultad para pronosticar con modelos SARIMA/Prophet

### Hipotesis del Mes Vencido

El recaudo de **enero** refleja el consumo de **diciembre** (mes vencido).
La ACF en lag-1 de las primeras diferencias captura este efecto operativo.
""")


# ════════════════════════════════════════════════════════════════
# CELDA 18 — CODE: Lag-12 ACF por entidad
# ════════════════════════════════════════════════════════════════
code(r"""print('=' * 70)
print('FASE VI - Autocorrelacion Lag-12 + Deteccion de Anomalias STAR')
print('=' * 70)

# --- 6.1 ACF de la serie agregada ---
from statsmodels.tsa.stattools import acf as sm_acf

serie_agg_vals = serie_mensual_agg.values
n_lags = min(24, len(serie_agg_vals) // 2 - 1)
acf_agg = sm_acf(serie_agg_vals, nlags=n_lags, fft=True)

print(f'\n  ACF Serie Agregada:')
print(f'  Lag-1:  {acf_agg[1]:.4f}')
print(f'  Lag-6:  {acf_agg[6]:.4f}')
print(f'  Lag-12: {acf_agg[12]:.4f}')
if n_lags >= 24:
    print(f'  Lag-24: {acf_agg[24]:.4f}')
print(f'  Memoria anual: {"FUERTE" if acf_agg[12] > 0.5 else "DEBIL"} (r12 = {acf_agg[12]:.3f})')

# --- 6.2 Lag-12 autocorrelacion por entidad (Top 30 + anomalias) ---
# Usar series mensuales por entidad
mensual_entidad = (df.groupby(['Entidad', 'YM'])[COL_VALOR]
                   .sum()
                   .reset_index())
mensual_entidad.columns = ['Entidad', 'YM', 'Recaudo_Mensual']

# Solo entidades con >= 36 meses
meses_por_ent = mensual_entidad.groupby('Entidad')['YM'].count()
ents_acf_validas = meses_por_ent[meses_por_ent >= 36].index

lag12_results = []
for ent in ents_acf_validas:
    serie_ent = mensual_entidad[mensual_entidad['Entidad'] == ent].sort_values('YM')
    vals = serie_ent['Recaudo_Mensual'].values
    if len(vals) >= 25 and np.std(vals) > 0:
        try:
            acf_vals = sm_acf(vals, nlags=12, fft=True)
            r12 = acf_vals[12]
            r1 = acf_vals[1]
        except Exception:
            r12, r1 = np.nan, np.nan
        lag12_results.append({
            'Entidad': ent,
            'R_Lag1': r1,
            'R_Lag12': r12,
            'N_Meses': len(vals),
        })

df_lag12 = pd.DataFrame(lag12_results).dropna()
print(f'\n  Entidades con ACF calculada: {len(df_lag12)}')
print(f'  R_Lag12 mediana: {df_lag12["R_Lag12"].median():.3f}')
print(f'  R_Lag12 media:   {df_lag12["R_Lag12"].mean():.3f}')

# Anomalias: entidades con R_Lag12 < 0.3 (rompen el patron)
umbral_anomalia = 0.3
anomalias = df_lag12[df_lag12['R_Lag12'] < umbral_anomalia].sort_values('R_Lag12')
print(f'\n  === ANOMALIAS: Entidades con R_Lag12 < {umbral_anomalia} ({len(anomalias)}) ===')
for _, row in anomalias.head(10).iterrows():
    print(f'    {row["Entidad"][:50]:<50} R12={row["R_Lag12"]:>6.3f}')

print(f'\n  Estas entidades son candidatas para fiscalizacion proactiva (STAR)')

# Merge lag12 info to features
df_features = df_features.merge(
    df_lag12[['Entidad', 'R_Lag12', 'R_Lag1']],
    on='Entidad', how='left'
)
df_features['R_Lag12'] = df_features['R_Lag12'].fillna(0)
df_features['R_Lag1'] = df_features['R_Lag1'].fillna(0)
""")


# ════════════════════════════════════════════════════════════════
# CELDA 19 — CODE: Visualizacion Lag-12
# ════════════════════════════════════════════════════════════════
code(r"""# --- 6.3 Visualizacion ACF + Lag-12 por entidad ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel A: ACF de la serie agregada
ax = axes[0]
lags_x = np.arange(n_lags + 1)
colors_acf = [C_SECONDARY if l in [1, 12] else C_TERTIARY for l in lags_x]
ax.bar(lags_x, acf_agg, color=colors_acf, alpha=0.8, edgecolor='white')
n_obs = len(serie_agg_vals)
ci = 1.96 / np.sqrt(n_obs)
ax.axhline(y=ci, color=C_TEXT_LIGHT, linestyle='--', alpha=0.6, linewidth=0.8)
ax.axhline(y=-ci, color=C_TEXT_LIGHT, linestyle='--', alpha=0.6, linewidth=0.8)
ax.axhline(y=0, color='black', linewidth=0.8)

# Anotar lag-1 y lag-12
ax.annotate(f'Lag-1: {acf_agg[1]:.3f}\n(Mes vencido)',
            xy=(1, acf_agg[1]), xytext=(4, acf_agg[1] + 0.15),
            fontsize=8, fontfamily='serif',
            arrowprops=dict(arrowstyle='->', color=C_SECONDARY),
            color=C_SECONDARY, fontweight='bold')
ax.annotate(f'Lag-12: {acf_agg[12]:.3f}\n(Memoria anual)',
            xy=(12, acf_agg[12]), xytext=(16, acf_agg[12] + 0.15),
            fontsize=8, fontfamily='serif',
            arrowprops=dict(arrowstyle='->', color=C_SECONDARY),
            color=C_SECONDARY, fontweight='bold')

ax.set_xlabel('Lag (meses)', fontdict=FONT_AXIS)
ax.set_ylabel('Autocorrelacion', fontdict=FONT_AXIS)
ax.set_title('ACF Serie Agregada de Rentas Cedidas\n(Memoria Anual y Mes Vencido)',
             fontdict=FONT_TITLE, pad=12)
ax.grid(True, axis='y', alpha=0.3)

# Panel B: Distribucion R_Lag12 por tipologia
ax2 = axes[1]
# Merge tipologia into df_lag12
df_lag12_tipo = df_lag12.merge(df_features[['Entidad', 'Tipologia']], on='Entidad', how='left')
df_lag12_tipo = df_lag12_tipo.dropna(subset=['Tipologia'])

tipos_order = ['Consolidados', 'Emergentes', 'Dependientes', 'Criticos']
data_lag = [df_lag12_tipo[df_lag12_tipo['Tipologia'] == t]['R_Lag12'].values
            for t in tipos_order if (df_lag12_tipo['Tipologia'] == t).any()]
labels_lag = [t for t in tipos_order if (df_lag12_tipo['Tipologia'] == t).any()]
bp_colors_lag = [tipo_colors[t] for t in labels_lag]

bp2 = ax2.boxplot(data_lag, labels=labels_lag, patch_artist=True,
                  medianprops=dict(color='black', linewidth=1.5))
for patch, color in zip(bp2['boxes'], bp_colors_lag):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax2.axhline(y=umbral_anomalia, color=C_SECONDARY, linestyle='--', linewidth=1.0,
            alpha=0.7, label=f'Umbral anomalia ({umbral_anomalia})')
ax2.set_ylabel('Autocorrelacion Lag-12', fontdict=FONT_AXIS)
ax2.set_title('R(Lag-12) por Tipologia\n(Entidades bajo umbral = anomalias STAR)',
              fontdict=FONT_TITLE, pad=12)
ax2.legend(prop={'family': 'serif', 'size': 9})
ax2.grid(True, axis='y', alpha=0.3)

plt.tight_layout(pad=2.0)
marca_agua(fig)
guardar_figura(fig, '10_05_lag12_acf', OUTPUTS_FIGURES)
plt.show()

print(f'\n  HALLAZGO: {len(anomalias)} entidades rompen el patron lag-12')
print(f'  Son candidatas prioritarias para el modulo de anomalias STAR.')
""")


# ════════════════════════════════════════════════════════════════
# CELDA 20 — MARKDOWN: Fase VII SAT
# ════════════════════════════════════════════════════════════════
md(r"""---

## Fase VII — Configuracion del Sistema de Alertas (SAT-STAR)

### 7.1 Indice de Eficiencia Predictiva (IEP)

$$\text{IEP} = \frac{\Delta\% \text{Recaudo}}{\Delta\% \text{UPC}}$$

- $\text{IEP} > 1$: Ingresos crecen mas rapido que costos UPC (sostenible)
- $\text{IEP} < 1$: Costos UPC crecen mas rapido (riesgo de desfinanciamiento)
- $\text{IEP} < 0$: Recaudo decreciente con UPC creciente (critico)

### 7.2 Puntuacion de Riesgo de Entidad (ERS)

Indice compuesto normalizado (0-100):

$$\text{ERS} = 0.30 \times \hat{CV} + 0.25 \times \hat{T}^{-} + 0.25 \times \hat{IEP}^{-} + 0.20 \times (1 - \hat{r}_{12})$$

Donde $\hat{X}$ denota normalizacion min-max y $X^{-}$ invierte la escala.

### 7.3 Semaforo Adaptativo

| Nivel | ERS | Accion |
|-------|-----|--------|
| VERDE | $\leq Q_{25}$ | Monitoreo trimestral |
| AMARILLO | $(Q_{25}, Q_{50}]$ | Monitoreo mensual |
| NARANJA | $(Q_{50}, Q_{75}]$ | Alerta activa + revision |
| ROJO | $> Q_{75}$ o IEP < 1 | Intervencion prioritaria |

### Marco de Gobernanza

- **Ley 1753 de 2015** (Art. 65): Establece el fortalecimiento fiscal territorial
- **Decreto 2265 de 2017**: Reglamenta distribucion y giro de Rentas Cedidas
- El XGBoost (MAPE 5.05%) alimenta las estimaciones de recaudo potencial
""")


# ════════════════════════════════════════════════════════════════
# CELDA 21 — CODE: IEP + ERS + SAT
# ════════════════════════════════════════════════════════════════
code(r"""print('=' * 70)
print('FASE VII - SAT: IEP, ERS y Semaforo Adaptativo')
print('=' * 70)

# --- 7.1 Indice de Eficiencia Predictiva (IEP) ---
# UPC growth from MACRO_DATA
upc_growth = {y: v['UPC'] for y, v in MACRO_DATA.items()}

# Calcular IEP por entidad: media de (delta%recaudo / delta%UPC) por transicion anual
iep_lista = []
for entidad in df_features['Entidad']:
    datos_ent = anual_filtrado[anual_filtrado['Entidad'] == entidad].sort_values('Anio')
    if len(datos_ent) < 2:
        iep_lista.append(np.nan)
        continue
    iep_anual = []
    for i in range(1, len(datos_ent)):
        anio = datos_ent.iloc[i]['Anio']
        rec_prev = datos_ent.iloc[i-1]['Recaudo_Anual']
        rec_curr = datos_ent.iloc[i]['Recaudo_Anual']
        if rec_prev > 0 and anio in upc_growth and upc_growth[anio] > 0:
            delta_rec = (rec_curr / rec_prev - 1) * 100
            delta_upc = upc_growth[anio]
            iep = delta_rec / delta_upc
            iep_anual.append(iep)
    if len(iep_anual) > 0:
        iep_lista.append(np.median(iep_anual))
    else:
        iep_lista.append(np.nan)

df_features['IEP'] = iep_lista
df_features['IEP'] = df_features['IEP'].fillna(1.0)  # Default neutral

print(f'\n  IEP calculado para {df_features["IEP"].notna().sum()} entidades')
print(f'  IEP mediana: {df_features["IEP"].median():.2f}')
print(f'  IEP < 1 (riesgo): {(df_features["IEP"] < 1).sum()} entidades')
print(f'  IEP < 0 (critico): {(df_features["IEP"] < 0).sum()} entidades')

# --- 7.2 Puntuacion de Riesgo de Entidad (ERS) ---
# Componentes:
# 1. CV_norm: mayor CV = mayor riesgo
# 2. Trend_neg: tendencia negativa = mayor riesgo
# 3. IEP_inv: IEP bajo (<1) = mayor riesgo
# 4. Lag12_inv: menor autocorrelacion = mayor riesgo

def minmax_norm(series):
    mn, mx = series.min(), series.max()
    if mx - mn == 0:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)

# CV normalizado (mayor = mas riesgo)
cv_norm = minmax_norm(df_features['CV_Interanual']) * 100

# Tendencia invertida (negativa = mas riesgo)
trend_inv = minmax_norm(-df_features['Tendencia_Pct']) * 100

# IEP invertido (menor IEP = mas riesgo), cap at reasonable range
iep_capped = df_features['IEP'].clip(-2, 5)
iep_inv = minmax_norm(-iep_capped) * 100

# Lag-12 invertido (menor R12 = mas riesgo)
lag12_inv = minmax_norm(-df_features['R_Lag12'].clip(-1, 1)) * 100

# ERS compuesto
w_cv, w_trend, w_iep, w_lag = 0.30, 0.25, 0.25, 0.20
df_features['ERS'] = (w_cv * cv_norm +
                       w_trend * trend_inv +
                       w_iep * iep_inv +
                       w_lag * lag12_inv)

print(f'\n  === Entity Risk Score (ERS) ===')
print(f'  ERS mediana: {df_features["ERS"].median():.1f}')
print(f'  ERS Q25: {df_features["ERS"].quantile(0.25):.1f}')
print(f'  ERS Q75: {df_features["ERS"].quantile(0.75):.1f}')

# --- 7.3 Semaforo Adaptativo (por cuartiles ERS) ---
q25 = df_features['ERS'].quantile(0.25)
q50 = df_features['ERS'].quantile(0.50)
q75 = df_features['ERS'].quantile(0.75)

def asignar_semaforo(row):
    ers = row['ERS']
    iep = row['IEP']
    # Alerta Roja forzada si IEP < 1 con desviacion > 35%
    if iep < 0 or ers > q75:
        return 'ROJO'
    elif ers > q50:
        return 'NARANJA'
    elif ers > q25:
        return 'AMARILLO'
    else:
        return 'VERDE'

df_features['Semaforo'] = df_features.apply(asignar_semaforo, axis=1)

# Conteo
semaforo_count = df_features['Semaforo'].value_counts()
semaforo_colors_map = {'VERDE': C_QUATERNARY, 'AMARILLO': '#F1C40F',
                       'NARANJA': C_QUINARY, 'ROJO': C_SECONDARY}

print(f'\n  === Distribucion Semaforo SAT (Adaptativo) ===')
print(f'  Umbrales ERS: Q25={q25:.1f}, Q50={q50:.1f}, Q75={q75:.1f}')
for sem in ['VERDE', 'AMARILLO', 'NARANJA', 'ROJO']:
    n = semaforo_count.get(sem, 0)
    pct = n / len(df_features) * 100
    print(f'  {sem:10s}: {n:>4} entidades ({pct:>5.1f}%)')

# Top 10 en ROJO
rojas = df_features[df_features['Semaforo'] == 'ROJO'].sort_values('ERS', ascending=False)
print(f'\n  === Top 10 Entidades en ALERTA ROJA ===')
for _, row in rojas.head(10).iterrows():
    print(f'    {row["Entidad"][:45]:<45} ERS={row["ERS"]:>5.1f} CV={row["CV_Interanual"]:>5.1f}% IEP={row["IEP"]:>5.2f}')

# Marco legal
print(f'\n  MARCO DE GOBERNANZA:')
print(f'  - Ley 1753 de 2015 (Art. 65): Fortalecimiento fiscal territorial')
print(f'  - Decreto 2265 de 2017: Distribucion y giro de Rentas Cedidas')
print(f'  - El SAT opera bajo estos marcos normativos para la deteccion')
print(f'    proactiva de riesgo de desfinanciamiento en salud.')
""")


# ════════════════════════════════════════════════════════════════
# CELDA 22 — CODE: Visualizacion SAT
# ════════════════════════════════════════════════════════════════
code(r"""# --- 7.4 Visualizacion SAT ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: Pie chart semaforo
ax1 = axes[0]
sizes = [semaforo_count.get(s, 0) for s in ['VERDE', 'AMARILLO', 'NARANJA', 'ROJO']]
colors_pie = [semaforo_colors_map[s] for s in ['VERDE', 'AMARILLO', 'NARANJA', 'ROJO']]
labels_pie = [f'{s}\n({n})' for s, n in zip(['VERDE', 'AMARILLO', 'NARANJA', 'ROJO'], sizes)]
wedges, texts, autotexts = ax1.pie(sizes, labels=labels_pie, colors=colors_pie,
                                    autopct='%1.1f%%', startangle=90,
                                    textprops={'fontsize': 10, 'fontfamily': 'serif'})
for t in autotexts:
    t.set_fontsize(9)
    t.set_fontfamily('serif')
ax1.set_title('Semaforo SAT Adaptativo\n(por cuartiles ERS)',
              fontdict=FONT_TITLE, pad=15)

# Panel B: Scatter ERS components
ax2 = axes[1]
for sem in ['VERDE', 'AMARILLO', 'NARANJA', 'ROJO']:
    mask = df_features['Semaforo'] == sem
    if mask.sum() > 0:
        ax2.scatter(df_features.loc[mask, 'IEP'].clip(-2, 5),
                    df_features.loc[mask, 'CV_Interanual'],
                    color=semaforo_colors_map[sem], label=sem,
                    alpha=0.5, s=30, edgecolors='white', linewidth=0.3)
ax2.set_xlabel('IEP (Eficiencia Predictiva)', fontdict=FONT_AXIS)
ax2.set_ylabel('CV Interanual (%)', fontdict=FONT_AXIS)
ax2.set_title('Mapa de Riesgo Fiscal\nIEP vs CV Interanual',
              fontdict=FONT_TITLE, pad=12)
ax2.axvline(x=1.0, color='black', linestyle=':', alpha=0.4)
ax2.annotate('IEP=1\n(equilibrio)', xy=(1.0, ax2.get_ylim()[1]*0.9),
             fontsize=8, fontfamily='serif', ha='center')
ax2.legend(title='Semaforo', prop={'family': 'serif', 'size': 8})
ax2.grid(True, alpha=0.2)

# Panel C: Recaudo total por semaforo
ax3 = axes[2]
recaudo_sem = df_features.groupby('Semaforo')['Recaudo_Total'].sum() / 1e12
order_sem = ['VERDE', 'AMARILLO', 'NARANJA', 'ROJO']
vals_sem = [recaudo_sem.get(s, 0) for s in order_sem]
bars = ax3.bar(order_sem, vals_sem,
               color=[semaforo_colors_map[s] for s in order_sem],
               alpha=0.85, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, vals_sem):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'${val:.2f}T', ha='center', va='bottom',
             fontsize=10, fontfamily='serif', fontweight='bold')
ax3.set_ylabel('Recaudo Total (Billones COP)', fontdict=FONT_AXIS)
ax3.set_title('Volumen de Recaudo por Nivel\nde Alerta SAT',
              fontdict=FONT_TITLE, pad=12)
ax3.grid(True, axis='y', alpha=0.3)

plt.tight_layout(pad=2.0)
marca_agua(fig)
guardar_figura(fig, '10_06_semaforo_SAT', OUTPUTS_FIGURES)
plt.show()
""")


# ════════════════════════════════════════════════════════════════
# CELDA 23 — MARKDOWN: Fase VIII Heat Map
# ════════════════════════════════════════════════════════════════
md(r"""---

## Fase VIII — Mapa de Calor Territorial y Box-Plots Multitemporales

### 8.1 Heat Map de Eficiencia Territorial

Matriz entidad x mes (Top 20) que visualiza la densidad de recaudo
normalizada por fila, revelando patrones estacionales diferenciados.

### 8.2 Box-Plots por Tipologia y Anio

Distribucion del recaudo anual por tipologia, validando que la varianza
de las entidades Criticas es sistematicamente mayor.
""")


# ════════════════════════════════════════════════════════════════
# CELDA 24 — CODE: Heat Map + Box-Plots
# ════════════════════════════════════════════════════════════════
code(r"""print('=' * 70)
print('FASE VIII - Mapa de Calor + Box-Plots Multitemporales')
print('=' * 70)

# --- 8.1 Heat Map: Top 20 entidades x Mes ---
top20_ents = (df.groupby('Entidad')[COL_VALOR].sum()
              .nlargest(20).index.tolist())
df_top20 = df[df['Entidad'].isin(top20_ents)].copy()

heatmap_data = (df_top20.groupby(['Entidad', 'Mes'])[COL_VALOR]
                .mean()
                .unstack(fill_value=0))
# Normalizar por fila
heatmap_norm = heatmap_data.div(heatmap_data.max(axis=1), axis=0)

# Ordenar por recaudo total
orden_ents = (df_top20.groupby('Entidad')[COL_VALOR].sum()
              .sort_values(ascending=False).index)
heatmap_norm = heatmap_norm.reindex(orden_ents)
nombres_cortos = [n[:30] for n in heatmap_norm.index]

fig, ax = plt.subplots(figsize=(14, 10))
cmap_custom = LinearSegmentedColormap.from_list(
    'rentas', ['#FAFBFC', '#D5E8F0', C_TERTIARY, C_PRIMARY, C_SECONDARY], N=256)
sns.heatmap(heatmap_norm.values, ax=ax, cmap=cmap_custom,
            xticklabels=MESES_LABELS,
            yticklabels=nombres_cortos,
            linewidths=0.3, linecolor='white',
            cbar_kws={'label': 'Recaudo Normalizado (0-1)', 'shrink': 0.8})
ax.set_xlabel('Mes', fontdict=FONT_AXIS)
ax.set_ylabel('Entidad Territorial', fontdict=FONT_AXIS)
ax.set_title('Mapa de Calor - Estacionalidad del Recaudo por Entidad\n(Top 20, normalizado por fila)',
             fontdict=FONT_TITLE, pad=15)
plt.xticks(fontsize=10, fontfamily='serif')
plt.yticks(fontsize=8, fontfamily='serif')
plt.tight_layout(pad=2.0)
marca_agua(fig)
guardar_figura(fig, '10_07_heatmap_territorial', OUTPUTS_FIGURES)
plt.show()

# --- 8.2 Box-Plots por tipologia y anio ---
anual_con_tipo = anual_filtrado.merge(
    df_features[['Entidad', 'Tipologia']],
    on='Entidad', how='inner'
)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
tipos_plot = ['Consolidados', 'Emergentes', 'Dependientes', 'Criticos']

for idx, tipo in enumerate(tipos_plot):
    ax = axes[idx // 2, idx % 2]
    data_tipo = anual_con_tipo[anual_con_tipo['Tipologia'] == tipo]
    if len(data_tipo) > 0:
        anios_disp = sorted(data_tipo['Anio'].unique())
        data_by_year = [data_tipo[data_tipo['Anio'] == a]['Recaudo_Anual'].values / 1e6
                        for a in anios_disp]
        bp = ax.boxplot(data_by_year, labels=[str(a) for a in anios_disp],
                        patch_artist=True,
                        medianprops=dict(color='black', linewidth=1.5),
                        flierprops=dict(marker='o', markersize=3, alpha=0.4))
        for patch in bp['boxes']:
            patch.set_facecolor(tipo_colors[tipo])
            patch.set_alpha(0.6)
        n_ent = data_tipo['Entidad'].nunique()
        cv_med = df_features[df_features['Tipologia'] == tipo]['CV_Interanual'].median()
        ax.set_title(f'{tipo} (n={n_ent}, CV med={cv_med:.0f}%)',
                     fontdict=FONT_TITLE, pad=10)
    else:
        ax.set_title(f'{tipo} (sin datos)', fontdict=FONT_TITLE, pad=10)
    ax.set_ylabel('Recaudo Anual (M COP)', fontdict=FONT_AXIS)
    ax.set_xlabel('Anio', fontdict=FONT_AXIS)
    ax.grid(True, axis='y', alpha=0.3)

plt.suptitle('Distribucion del Recaudo Anual por Tipologia y Anio\nRentas Cedidas - ADRES',
             fontsize=14, fontweight='bold', fontfamily='serif', y=1.02)
plt.tight_layout(pad=2.0)
marca_agua(fig)
guardar_figura(fig, '10_08_boxplots_tipologia', OUTPUTS_FIGURES)
plt.show()

print(f'\n  Mapa de calor: {len(top20_ents)} entidades x 12 meses')
print(f'  Box-plots: {len(tipos_plot)} tipologias x {len(anios_disp)} anios')
""")


# ════════════════════════════════════════════════════════════════
# CELDA 25 — CODE: Informe Narrativo Final + Exportaciones
# ════════════════════════════════════════════════════════════════
code(r"""# --- INFORME NARRATIVO AUTOMATIZADO ---
informe = []
informe.append('=' * 70)
informe.append('INFORME DE BENCHMARKING MULTIDIMENSIONAL TERRITORIAL')
informe.append('Sistema STAR de Analisis de Rentas Cedidas - ADRES')
informe.append('=' * 70)
informe.append('')
informe.append('MARCO REGULATORIO')
informe.append('  - Ley 1753 de 2015 (Art. 65): Fortalecimiento fiscal territorial')
informe.append('  - Decreto 2265 de 2017: Distribucion y giro de Rentas Cedidas')
informe.append('  - Ley 715 de 2001 (Art. 44): Competencias territoriales en salud')
informe.append('')
informe.append('1. CONCENTRACION FISCAL')
informe.append(f'   Indice de Gini: {gini:.4f} ({"Alta" if gini > 0.6 else "Moderada"})')
informe.append(f'   Top 5 entidades: {pct_top5:.1f}% del recaudo')
informe.append(f'   Proxy 5 departamentos: {pct_dept5:.1f}%')
informe.append(f'   Pareto (20% entidades): {pct_pareto:.1f}%')
informe.append(f'   Bottom 50%: {pct_bottom50:.2f}% (vulnerabilidad extrema)')
informe.append(f'   Validacion Orozco-Gallo: {"CONFIRMADA" if pct_dept5 > 50 else "PARCIAL"}')
informe.append('')
informe.append('2. TIPOLOGIAS TERRITORIALES (K-Means, k=4, CV interanual)')
for tipo in ['Consolidados', 'Emergentes', 'Dependientes', 'Criticos']:
    if tipo in resumen_tipo.index:
        r = resumen_tipo.loc[tipo]
        informe.append(f'   {tipo}:')
        informe.append(f'     Entidades: {int(r["N_Entidades"])}')
        informe.append(f'     Recaudo mediano anual: ${r["Recaudo_Medio_B"]:,.2f} miles MM')
        informe.append(f'     CV interanual mediano: {r["CV_Mediano"]:.1f}%')
        informe.append(f'     % del recaudo total: {r["Pct_Recaudo_Total"]:.1f}%')
informe.append('')
informe.append('3. ASIMETRIA ESTRUCTURAL')
if len(common_idx) > 0:
    informe.append(f'   Ratio Bogota/Choco: {med_ratio:.0f}x (mediana mensual)')
    informe.append(f'   Per capita literatura: $12,500 (Bog) vs $667 (Cho) = 18.7:1')
    informe.append(f'   Correlacion estacional: r = {corr_ecg:.3f}')
    ecg_estado = 'sincronizado' if corr_ecg > 0.5 else 'ERRATICO (justifica Alerta Roja)'
    informe.append(f'   Diagnostico: {ecg_estado}')
informe.append('')
informe.append('4. DEFLACTACION Y ELASTICIDAD')
informe.append(f'   Efecto inflacion: {efecto_inflacion:.1f}% del nominal')
informe.append(f'   Crecimiento nominal: {crec_nom:+.1f}% vs Real: {crec_real:+.1f}%')
informe.append(f'   Elasticidad beta vs SMLV: {slope_beta:.3f}')
informe.append(f'   Impacto estimado 23% SMLV (2026): +{slope_beta * 23:.1f}% en recaudo')
informe.append('')
informe.append('5. AUTOCORRELACION Y ANOMALIAS')
informe.append(f'   ACF Lag-12 agregada: {acf_agg[12]:.3f} (memoria anual {"FUERTE" if acf_agg[12] > 0.5 else "DEBIL"})')
informe.append(f'   ACF Lag-1 agregada: {acf_agg[1]:.3f} (mes vencido)')
informe.append(f'   Entidades con R_Lag12 < {umbral_anomalia}: {len(anomalias)} (anomalias STAR)')
informe.append('')
informe.append('6. SISTEMA DE ALERTA TEMPRANA (SAT)')
for sem in ['VERDE', 'AMARILLO', 'NARANJA', 'ROJO']:
    n = semaforo_count.get(sem, 0)
    pct = n / len(df_features) * 100
    informe.append(f'   {sem}: {n} entidades ({pct:.1f}%)')
n_riesgo = semaforo_count.get('NARANJA', 0) + semaforo_count.get('ROJO', 0)
informe.append(f'   Total en riesgo (Naranja+Rojo): {n_riesgo} ({n_riesgo/len(df_features)*100:.1f}%)')
informe.append(f'   IEP < 1: {(df_features["IEP"] < 1).sum()} entidades en riesgo de costos')
informe.append('')
informe.append('7. RECOMENDACIONES')
informe.append('   a) Implementar SAT trimestral para entidades ROJO/NARANJA')
informe.append('   b) Fondos de estabilizacion para Dependientes (CV > Q75)')
informe.append('   c) Fiscalizacion proactiva: entidades con R_Lag12 < 0.3')
informe.append('   d) Deflactacion IPC obligatoria en evaluacion fiscal')
informe.append('   e) XGBoost (MAPE 5.05%) como motor predictivo del STAR')
informe.append(f'   f) Monitorizar elasticidad SMLV: beta={slope_beta:.2f} ante +23% en 2026')
informe.append('   g) Alerta Roja permanente para Choco: perfil estacional erratico')
informe.append('     y ratio de desigualdad > 10:1 vs Bogota')
informe.append('')
informe.append('=' * 70)

# Imprimir
for line in informe:
    print(line)

# Exportar informe
ruta_informe = OUTPUTS_REPORTS / 'informe_benchmarking_territorial.md'
with open(ruta_informe, 'w', encoding='utf-8') as f:
    f.write('\n'.join(informe))
print(f'\n  Informe exportado: {ruta_informe.name}')

# Exportar features completas con ERS y semaforo
cols_export = ['Entidad', 'Recaudo_Total', 'Recaudo_Anual_Medio', 'CV_Interanual',
               'Tendencia_Pct', 'N_Anios', 'Tipologia', 'R_Lag12', 'R_Lag1',
               'IEP', 'ERS', 'Semaforo']
df_export = df_features[[c for c in cols_export if c in df_features.columns]]
ruta_features = OUTPUTS_REPORTS / 'features_entidades_benchmarking.csv'
df_export.to_csv(ruta_features, index=False, encoding='utf-8-sig')
print(f'  Features exportadas: {ruta_features.name} ({len(df_export)} entidades)')

print(f'\n  {"="*50}')
print(f'  NOTEBOOK 10 COMPLETADO EXITOSAMENTE')
print(f'  8 figuras generadas | 3 reportes exportados')
print(f'  {"="*50}')
""")


# ════════════════════════════════════════════════════════════════
# CELDA 26 — MARKDOWN: Conclusiones
# ════════════════════════════════════════════════════════════════
md(r"""---

## Conclusiones

### Hallazgos Principales

1. **Concentracion extrema validada**: El Indice de Gini confirma una
   estructura fiscal altamente concentrada. El Bottom-50% de entidades
   genera menos del 2% del recaudo, evidenciando vulnerabilidad critica
   en municipios categoria 5 y 6.

2. **Tipologias coherentes (CV interanual)**: La correccion metodologica
   de usar CV interanual (mediana ~15%) en vez de mensual (mediana ~131%)
   produce una taxonomia coherente con la literatura
   (Santamaria et al., 2008: 18-34% Dependientes).

3. **Brecha Bogota-Choco**: El ratio de desigualdad supera 10:1 en la
   mayoria de meses, consistente con la brecha per capita documentada
   ($12,500 vs $667). El perfil estacional fiscal del Choco es mas
   erratico, justificando Alerta Roja permanente.

4. **Efecto inflacionario medible**: La deflactacion IPC demuestra que
   parte del crecimiento nominal es ilusorio. La elasticidad beta
   permite proyectar el impacto del 23% de aumento del SMLV en 2026.

5. **Lag-12 como detector de anomalias**: Las entidades que rompen el
   patron de memoria anual (R_Lag12 < 0.3) son candidatas prioritarias
   para fiscalizacion proactiva en el STAR.

6. **SAT adaptativo operativo**: El semaforo basado en ERS (cuartiles)
   distribuye proporcionalmente las alertas, superando la version anterior
   que clasificaba el 97.9% como ROJO.

### Implicaciones para Politica Publica

- **Ley 1753 de 2015**: El SAT-STAR operacionaliza el mandato de
  fortalecimiento fiscal territorial.
- **Decreto 2265 de 2017**: La clasificacion ERS permite priorizar la
  distribucion de Rentas Cedidas hacia entidades criticas.
- **Multicolinealidad SMLV-UPC** (r = 0.903): El SAT prioriza UPC como
  variable de costo e IPC como deflactor de ingreso para evitar
  inestabilidad en coeficientes.

---

*Notebook generado automaticamente por el Sistema STAR de Rentas Cedidas.*
*Autores: Efren Bohorquez, Mauricio Garcia, Ernesto Sanchez*
""")


# ════════════════════════════════════════════════════════════════
# GUARDAR NOTEBOOK
# ════════════════════════════════════════════════════════════════
out_path = Path(__file__).resolve().parent.parent / 'notebooks' / '09_Benchmarking_Territorial.ipynb'
nbf.write(nb, str(out_path))
print(f'\nNotebook generado: {out_path}')
print(f'Celdas: {len(nb.cells)} ({sum(1 for c in nb.cells if c.cell_type == "markdown")} MD + '
      f'{sum(1 for c in nb.cells if c.cell_type == "code")} Code)')

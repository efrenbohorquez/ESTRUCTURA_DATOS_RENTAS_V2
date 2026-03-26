"""
build_02_estacionalidad.py — Genera el cuaderno 02_Estacionalidad.ipynb
=======================================================================
Ejecutar desde la raíz del proyecto:
    python scripts/build_02_estacionalidad.py

Estructura (18 celdas: 8 MD + 10 Code, 5 fases):
  Fase I  — Correlación Cruzada (CCF) y Lag Analysis
  Fase II — Descomposición STL Avanzada + log1p
  Fase III— Dinámicas por Vertical de Negocio
  Fase IV — Anomalías y Change Point Detection
  Fase V  — Validación de Estacionariedad (ADF + KPSS)
"""
import nbformat as nbf
from pathlib import Path

# ============================================================
nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

def md(text):
    nb.cells.append(nbf.v4.new_markdown_cell(text.strip()))

def code(text):
    nb.cells.append(nbf.v4.new_code_cell(text.strip()))

# ════════════════════════════════════════════════════════════
# CELDA 1 — Título (MD)
# ════════════════════════════════════════════════════════════
md(r"""# 02 — Análisis Avanzado de Estacionalidad y Dinámicas Temporales

**Sistema de Análisis y Pronóstico de Rentas Cedidas** | Departamentos y Distritos de Colombia

---

## Arquitectura Analítica

| Fase | Técnica | Objetivo |
|------|---------|----------|
| **I** | Correlación Cruzada (CCF) | Validar el rezago óptimo entre consumo y recaudo; confirmar hipótesis Dic→Ene, Jun→Jul |
| **II** | Descomposición STL Avanzada | Aislar el perfil estacional fiscal; evaluar transformación log1p |
| **III** | Dinámicas por Vertical | Deflactar Licores/Cigarrillos con IPC; analizar elasticidad de Juegos de Azar vs SMLV |
| **IV** | Anomalías y Change Points | Detectar quiebres estructurales (migración ERP 2025); clasificar negativos por TipoRegistro |
| **V** | Validación de Estacionariedad | ADF + KPSS en serie original y diferenciada $(d{=}1, D{=}1)$ |

> **Dependencia**: Requiere `serie_mensual.csv` generado por `01_EDA_Completo.ipynb`.""")

# ════════════════════════════════════════════════════════════
# CELDA 2 — Config + Data Load (Code)
# ════════════════════════════════════════════════════════════
code(r"""import pandas as pd
import numpy as np

%run 00_config.py

# ── Paquetes estadísticos ──
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec

# ── Cargar serie mensual agregada (generada en 01_EDA) ──
df_mensual = pd.read_csv(
    DATA_PROCESSED / 'serie_mensual.csv',
    parse_dates=['Fecha'], index_col='Fecha'
)
df_mensual.index.freq = 'MS'
serie = df_mensual['Recaudo_Total']

# ── Cargar dataset original para análisis por vertical ──
df_raw = pd.read_excel(DATA_FILE)
df_raw[COL_FECHA] = pd.to_datetime(df_raw[COL_FECHA])
df_raw[COL_VALOR] = pd.to_numeric(df_raw[COL_VALOR], errors='coerce')

n_meses = len(serie)
print(f"✅ Datos cargados — {n_meses} observaciones mensuales "
      f"({serie.index.min():%Y-%m} → {serie.index.max():%Y-%m})")
print(f"   Dataset vertical: {len(df_raw):,} registros × {df_raw.shape[1]} columnas")
print(f"   Categorías NombreSubGrupoFuente: {df_raw['NombreSubGrupoFuente'].nunique()}")
cats = df_raw['NombreSubGrupoFuente'].unique()
for c in sorted(cats):
    n = (df_raw['NombreSubGrupoFuente'] == c).sum()
    print(f"     • {c}  ({n:,} registros)")""")

# ════════════════════════════════════════════════════════════
# CELDA 3 — Fase I Header (MD)
# ════════════════════════════════════════════════════════════
md(r"""---

## Fase I — Refinamiento de la «Verdad Temporal» (Lag Analysis)

### Hipótesis Central

> *El recaudo de enero refleja el consumo masivo de diciembre; el de julio,
> el gasto de vacaciones y primas de junio.*

Para validar esta hipótesis se emplean dos técnicas:

1. **Correlación Cruzada (CCF)** entre el recaudo y variables proxy de consumo
   (Consumo_Hogares interpolado, IPC), identificando el rezago estadístico
   óptimo (Lag 1 o Lag 2).
2. **Heatmap de autocorrelación por mes**, que revela la estructura de
   dependencia temporal específica para cada mes del calendario y confirma
   si Ene y Jul dependen del mes inmediatamente anterior.""")

# ════════════════════════════════════════════════════════════
# CELDA 4 — CCF Analysis (Code)
# ════════════════════════════════════════════════════════════
code(r"""# ══════════════════════════════════════════════════════════════
# FASE I.1 — Correlación Cruzada (CCF) con Variables de Consumo
# ══════════════════════════════════════════════════════════════

# ── Construir serie proxy de consumo mensualizada ──
macro_df = pd.DataFrame(MACRO_DATA).T
macro_df.index.name = 'Año'

fechas_mensuales = serie.index
macro_mensual = pd.DataFrame(index=fechas_mensuales)

for col in ['IPC', 'Consumo_Hogares', 'Salario_Minimo']:
    serie_proxy = pd.Series(dtype=float, index=fechas_mensuales)
    for año, val in macro_df[col].items():
        mask = fechas_mensuales.year == año
        serie_proxy[mask] = val
    macro_mensual[col] = serie_proxy.astype(float).interpolate(method='linear')

# ── Normalizar (z-score) ──
recaudo_norm = (serie - serie.mean()) / serie.std()
consumo_norm = ((macro_mensual['Consumo_Hogares']
                 - macro_mensual['Consumo_Hogares'].mean())
                / macro_mensual['Consumo_Hogares'].std())
ipc_norm = ((macro_mensual['IPC'] - macro_mensual['IPC'].mean())
            / macro_mensual['IPC'].std())

# ── Función de Correlación Cruzada manual ──
max_lag = 6

def ccf_manual(x, y, max_lag):
    n = len(x)
    vals = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            corr = np.corrcoef(x[lag:], y[:n-lag])[0, 1] if n - lag > 2 else 0
        else:
            corr = np.corrcoef(x[:n+lag], y[-lag:])[0, 1] if n + lag > 2 else 0
        vals.append(corr)
    return vals

lags = list(range(-max_lag, max_lag + 1))
ccf_consumo = ccf_manual(recaudo_norm.values, consumo_norm.values, max_lag)
ccf_ipc     = ccf_manual(recaudo_norm.values, ipc_norm.values, max_lag)

ci_95 = 1.96 / np.sqrt(len(serie))    # Bartlett

# ── Gráfica CCF dual ──
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DUAL)

for ax, ccf_vals, titulo, color_base, subtitulo in [
    (axes[0], ccf_consumo, 'CCF: Recaudo vs Consumo Hogares',
     C_TERTIARY, 'Lag negativo = consumo lidera recaudo'),
    (axes[1], ccf_ipc, 'CCF: Recaudo vs IPC',
     C_QUATERNARY, 'Lag negativo = IPC lidera recaudo'),
]:
    colores_bar = [C_SECONDARY if abs(v) > ci_95 else color_base for v in ccf_vals]
    ax.bar(lags, ccf_vals, color=colores_bar, edgecolor='white',
           linewidth=0.5, width=0.7)
    ax.axhline(y= ci_95, color=C_SECONDARY, ls='--', alpha=0.7,
               label=f'IC 95% (±{ci_95:.3f})')
    ax.axhline(y=-ci_95, color=C_SECONDARY, ls='--', alpha=0.7)
    ax.axhline(y=0, color='black', lw=0.5)
    ax.axvline(x=0, color=C_TEXT_LIGHT, lw=0.5, ls=':')
    ax.set_xlabel('Rezago (meses)', fontdict=FONT_AXIS)
    ax.set_ylabel('Correlación cruzada', fontdict=FONT_AXIS)
    ax.set_xticks(lags)
    ax.legend(fontsize=8)
    if _VIZ_THEME_LOADED:
        titulo_profesional(ax, titulo, subtitulo)

    # Anotar lag óptimo
    idx_opt = int(np.argmax(np.abs(ccf_vals)))
    lag_opt = lags[idx_opt]
    val_opt = ccf_vals[idx_opt]
    ax.annotate(
        f'Lag óptimo: {lag_opt}\nr = {val_opt:.3f}',
        xy=(lag_opt, val_opt),
        xytext=(lag_opt + 2, val_opt * 0.7),
        arrowprops=dict(arrowstyle='->', color=C_PRIMARY),
        fontsize=9, fontweight='bold', color=C_PRIMARY,
        bbox=dict(boxstyle='round,pad=0.3',
                  facecolor='lightyellow', edgecolor=C_PRIMARY, alpha=0.8))

plt.tight_layout()
if _VIZ_THEME_LOADED:
    marca_agua(fig)
    guardar_figura(fig, '02_ccf_consumo_ipc', OUTPUTS_FIGURES)
plt.show()

# ── Reporte ──
idx_c = int(np.argmax(np.abs(ccf_consumo)))
idx_i = int(np.argmax(np.abs(ccf_ipc)))
lag_c, val_c = lags[idx_c], ccf_consumo[idx_c]
lag_i, val_i = lags[idx_i], ccf_ipc[idx_i]
print(f"\n{'═'*70}")
print(f"RESULTADOS CCF — REZAGO ÓPTIMO")
print(f"{'═'*70}")
print(f"  Consumo Hogares → Recaudo:  Lag = {lag_c:+d}  (r = {val_c:.4f})")
print(f"  IPC → Recaudo:              Lag = {lag_i:+d}  (r = {val_i:.4f})")
print(f"  IC 95% (Bartlett):          ±{ci_95:.4f}")
print()
if lag_c < 0:
    print(f"  → El consumo ANTICIPA el recaudo por {abs(lag_c)} mes(es)")
elif lag_c > 0:
    print(f"  → El recaudo SIGUE al consumo con {lag_c} mes(es) de rezago")
else:
    print(f"  → Recaudo y consumo son contemporáneos (lag = 0)")""")

# ════════════════════════════════════════════════════════════
# CELDA 5 — Heatmap Lag por Mes (Code)
# ════════════════════════════════════════════════════════════
code(r"""# ══════════════════════════════════════════════════════════════
# FASE I.2 — Heatmap de Autocorrelación por Mes y Rezago
# ══════════════════════════════════════════════════════════════

n_lags = 12
acf_by_month = np.zeros((12, n_lags))

for mes in range(1, 13):
    mask = serie.index.month == mes
    for lag in range(1, n_lags + 1):
        shifted = serie.shift(lag)
        valid   = ~shifted.isna()
        sub_o = serie[valid & mask]
        sub_l = shifted[valid & mask]
        if len(sub_o) > 2:
            acf_by_month[mes-1, lag-1] = np.corrcoef(
                sub_o.values, sub_l.values)[0, 1]

# ── Heatmap ──
fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
meses_labels = ['Ene','Feb','Mar','Abr','May','Jun',
                'Jul','Ago','Sep','Oct','Nov','Dic']

im = ax.imshow(acf_by_month, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(range(n_lags))
ax.set_xticklabels([f'Lag {i+1}' for i in range(n_lags)], fontsize=9)
ax.set_yticks(range(12))
ax.set_yticklabels(meses_labels, fontsize=10)
ax.set_xlabel('Rezago (meses)', fontdict=FONT_AXIS)

# Anotar valores
for i in range(12):
    for j in range(n_lags):
        v = acf_by_month[i, j]
        color = 'white' if abs(v) > 0.5 else 'black'
        ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                fontsize=7, color=color)

# Resaltar Enero y Julio
for mes_pico in [0, 6]:
    rect = plt.Rectangle((-0.5, mes_pico - 0.5), n_lags, 1,
                          lw=2.5, edgecolor=C_SECONDARY, facecolor='none')
    ax.add_patch(rect)

plt.colorbar(im, ax=ax, label='Correlación', shrink=0.8)
if _VIZ_THEME_LOADED:
    titulo_profesional(ax, 'Mapa de Autocorrelación por Mes y Rezago',
                       'Enero y Julio resaltados — ¿Lag 1 captura Dic→Ene / Jun→Jul?')
plt.tight_layout()
if _VIZ_THEME_LOADED:
    marca_agua(fig)
    guardar_figura(fig, '02_heatmap_lag_mensual', OUTPUTS_FIGURES)
plt.show()

# ── Análisis enfocado ──
ci = ci_95
print(f"\n{'═'*70}")
print(f"ESTRUCTURA DE REZAGO — MESES PICO")
print(f"{'═'*70}")
for m, nombre in [(1, 'ENERO'), (7, 'JULIO')]:
    print(f"\n  {nombre}:")
    for lag in range(1, 4):
        v = acf_by_month[m-1, lag-1]
        sig = '✓ Significativo' if abs(v) > ci else '✗ No significativo'
        print(f"    Lag {lag}: r = {v:.4f}  ({sig})")
    best = int(np.argmax(np.abs(acf_by_month[m-1, :]))) + 1
    print(f"    → Lag óptimo: {best}  (r = {acf_by_month[m-1, best-1]:.4f})")

lag1_ene = acf_by_month[0, 0]
lag1_jul = acf_by_month[6, 0]
print(f"\n{'─'*70}")
print(f"VALIDACIÓN DE HIPÓTESIS")
print(f"  Correlación Enero ↔ Diciembre (lag 1):  r = {lag1_ene:.4f}")
print(f"  Correlación Julio ↔ Junio    (lag 1):   r = {lag1_jul:.4f}")
if lag1_ene > 0.3:
    print(f"  ✅ CONFIRMADA: Ene depende del mes anterior (r > 0.3)")
else:
    print(f"  ⚠️ NO CONCLUYENTE: relación lag-1 débil; ")
    print(f"     el efecto puede ser lag-12 (estacionalidad pura)")""")

# ════════════════════════════════════════════════════════════
# CELDA 6 — Fase II Header (MD)
# ════════════════════════════════════════════════════════════
md(r"""---

## Fase II — Descomposición Estacional Avanzada (STL)

### Descomposición STL (Seasonal-Trend using Loess)

Cleveland et al. (1990) proponen:

$$Y_t = T_t + S_t + R_t$$

| Componente | Descripción |
|------------|-------------|
| $T_t$ | Tendencia — suavizado Loess |
| $S_t$ | Estacionalidad — periódico, ajuste iterativo robusto |
| $R_t$ | Residuo — componente irregular |

**Ventajas sobre decompose clásica:**
1. Robusta frente a outliers (weighted least squares)
2. Permite variación temporal del componente estacional
3. Control sobre suavidad de tendencia y estacionalidad

Se evalúa si la **varianza estacional aumenta con el nivel** de la serie
(heterocedasticidad estacional); de confirmarse, se aplica `log1p`
para estabilizar la varianza antes del modelado.""")

# ════════════════════════════════════════════════════════════
# CELDA 7 — STL Decomposition (Code)
# ════════════════════════════════════════════════════════════
code(r"""# ══════════════════════════════════════════════════════════════
# FASE II.1 — Descomposición STL: Perfil Estacional Fiscal
# ══════════════════════════════════════════════════════════════

stl = STL(serie, period=12, seasonal=13, robust=True)
res_stl = stl.fit()

# ── Gráfica profesional ──
fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(4, 1, hspace=0.25, height_ratios=[2, 1, 1, 1])

componentes = [
    ('Serie Observada $Y_t$',   serie,            C_PRIMARY,    True),
    ('Tendencia $T_t$',         res_stl.trend,    C_SECONDARY,  False),
    ('Estacionalidad $S_t$',    res_stl.seasonal, C_TERTIARY,   False),
    ('Residuo $R_t$',           res_stl.resid,    C_QUATERNARY, False),
]

for i, (titulo, datos, color, show_trend) in enumerate(componentes):
    ax = fig.add_subplot(gs[i])
    ax.plot(datos.index, datos.values, color=color, lw=1.5, label=titulo)
    if show_trend:
        ax.plot(res_stl.trend.index, res_stl.trend.values,
                color=C_SECONDARY, lw=2, ls='--', alpha=0.7, label='Tendencia')
    ax.set_ylabel(titulo, fontdict=FONT_AXIS)
    ax.grid(True, alpha=0.3)
    if _VIZ_THEME_LOADED:
        formato_pesos_eje(ax, eje='y')
    ax.legend(loc='upper left', fontsize=8)
    if i < 3:
        ax.set_xticklabels([])
    # Sombrear meses pico
    for fecha in datos.index:
        if fecha.month in MESES_PICO:
            ax.axvspan(fecha - pd.Timedelta(days=15),
                       fecha + pd.Timedelta(days=15),
                       alpha=0.05, color=C_SECONDARY)

ax.set_xlabel('Fecha', fontdict=FONT_AXIS)
fig.suptitle('Descomposición STL — Perfil Estacional Fiscal',
             fontsize=16, fontweight='bold', y=0.98, fontfamily='serif')
if _VIZ_THEME_LOADED:
    marca_agua(fig)
    guardar_figura(fig, '02_stl_descomposicion', OUTPUTS_FIGURES)
plt.show()

# ── Métricas de robustez (Hyndman) ──
F_seasonal = max(0, 1 - res_stl.resid.var()
                 / (res_stl.seasonal + res_stl.resid).var())
F_trend    = max(0, 1 - res_stl.resid.var()
                 / (res_stl.trend + res_stl.resid).var())

print(f"\n{'═'*70}")
print(f"MÉTRICAS DE ROBUSTEZ (Hyndman & Athanasopoulos)")
print(f"{'═'*70}")
print(f"  Fuerza estacional (F_s):  {F_seasonal:.4f}  "
      f"{'← FUERTE' if F_seasonal > 0.64 else '← DÉBIL'}")
print(f"  Fuerza tendencia  (F_t):  {F_trend:.4f}  "
      f"{'← FUERTE' if F_trend > 0.64 else '← DÉBIL'}")
print(f"  Var residuo / Var total:  {res_stl.resid.var() / serie.var():.4f}")
print(f"  Amplitud estacional σ_S:  ${res_stl.seasonal.std():,.0f}")
print(f"  Amplitud residual   σ_R:  ${res_stl.resid.std():,.0f}")""")

# ════════════════════════════════════════════════════════════
# CELDA 8 — Heterocedasticidad + log1p (Code)
# ════════════════════════════════════════════════════════════
code(r"""# ══════════════════════════════════════════════════════════════
# FASE II.2 — Diagnóstico de Heterocedasticidad + Transformación log1p
# ══════════════════════════════════════════════════════════════

# ── Nivel vs Varianza estacional por año ──
anos = sorted(serie.index.year.unique())
nivel_medio, var_est = [], []
for y in anos:
    m = serie.index.year == y
    if m.sum() >= 6:
        nivel_medio.append(serie[m].mean())
        var_est.append(res_stl.seasonal[m].var())
nivel_medio = np.array(nivel_medio)
var_est     = np.array(var_est)
corr_nv, p_nv = stats.pearsonr(nivel_medio, var_est)

# ── STL sobre serie log-transformada ──
serie_log = np.log1p(serie.clip(lower=0))
stl_log   = STL(serie_log, period=12, seasonal=13, robust=True)
res_log   = stl_log.fit()
F_s_log   = max(0, 1 - res_log.resid.var()
                / (res_log.seasonal + res_log.resid).var())

var_est_log, nivel_log = [], []
for y in anos:
    m = serie_log.index.year == y
    if m.sum() >= 6:
        nivel_log.append(serie_log[m].mean())
        var_est_log.append(res_log.seasonal[m].var())
corr_log, p_log = stats.pearsonr(nivel_log, var_est_log)

# ── Gráfica comparativa (3 paneles) ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Nivel vs Varianza (scatter)
ax = axes[0]
ax.scatter(nivel_medio/1e9, var_est/1e18, color=C_PRIMARY, s=100,
           zorder=5, edgecolors='white')
valid_anos = [y for y in anos if serie[serie.index.year == y].count() >= 6]
for i, year in enumerate(valid_anos):
    ax.annotate(str(year), (nivel_medio[i]/1e9, var_est[i]/1e18),
                fontsize=9, ha='center', va='bottom', fontweight='bold')
z = np.polyfit(nivel_medio/1e9, var_est/1e18, 1)
x_l = np.linspace(nivel_medio.min()/1e9, nivel_medio.max()/1e9, 50)
ax.plot(x_l, np.polyval(z, x_l), '--', color=C_SECONDARY, lw=1.5)
ax.set_xlabel('Nivel medio anual (miles MM$)', fontdict=FONT_AXIS)
ax.set_ylabel('Var. estacional (x10¹⁸)', fontdict=FONT_AXIS)
ax.grid(True, alpha=0.3)
if _VIZ_THEME_LOADED:
    titulo_profesional(ax, 'Heterocedasticidad Estacional',
                       f'r = {corr_nv:.3f}, p = {p_nv:.3f}')

# Panel 2: Comp. estacional por mes (original vs log1p)
ax  = axes[1]
ax2 = ax.twinx()
meses = range(1, 13)
est_o = [res_stl.seasonal[serie.index.month == m].mean() for m in meses]
est_l = [res_log.seasonal[serie_log.index.month == m].mean() for m in meses]
ml = ['E','F','M','A','M','J','J','A','S','O','N','D']
ax.bar(np.arange(12) - 0.2, [e/1e9 for e in est_o], 0.35,
       color=C_TERTIARY, alpha=0.8, label='Original')
ax2.bar(np.arange(12) + 0.2, est_l, 0.35,
        color=C_QUATERNARY, alpha=0.8, label='log1p')
ax.set_xticks(range(12)); ax.set_xticklabels(ml)
ax.set_ylabel('Comp. estacional original (MM$)',
              fontdict=FONT_AXIS, color=C_TERTIARY)
ax2.set_ylabel('Comp. estacional log1p',
               fontdict=FONT_AXIS, color=C_QUATERNARY)
ax.legend(loc='upper left', fontsize=8)
ax2.legend(loc='upper right', fontsize=8)
if _VIZ_THEME_LOADED:
    titulo_profesional(ax, 'Estacionalidad: Original vs log1p',
                       'Comparación de amplitud mensual')

# Panel 3: Desv. estándar por año
ax = axes[2]
x_pos = np.arange(len(valid_anos))
ax.bar(x_pos - 0.2, np.sqrt(var_est)/1e9, 0.35,
       color=C_TERTIARY, alpha=0.8, label='sigma Orig (MM$)')
ax.bar(x_pos + 0.2, np.sqrt(var_est_log), 0.35,
       color=C_QUATERNARY, alpha=0.8, label='sigma log1p')
ax.set_xticks(x_pos)
ax.set_xticklabels([str(y) for y in valid_anos])
ax.set_xlabel('Año', fontdict=FONT_AXIS)
ax.set_ylabel('Desv. estándar estacional', fontdict=FONT_AXIS)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
if _VIZ_THEME_LOADED:
    titulo_profesional(ax, 'Estabilización de Varianza',
                       f'log1p reduce correlación a r={corr_log:.3f}')

plt.tight_layout()
if _VIZ_THEME_LOADED:
    marca_agua(fig)
    guardar_figura(fig, '02_heterocedasticidad_log1p', OUTPUTS_FIGURES)
plt.show()

# ── Decisión ──
necesita_log = corr_nv > 0.5 and p_nv < 0.1
print(f"\n{'═'*70}")
print(f"DIAGNÓSTICO DE HETEROCEDASTICIDAD ESTACIONAL")
print(f"{'═'*70}")
print(f"  Corr nivel-varianza (original): r = {corr_nv:.4f}  (p = {p_nv:.4f})")
print(f"  Corr nivel-varianza (log1p):    r = {corr_log:.4f}  (p = {p_log:.4f})")
print(f"  F_s original:   {F_seasonal:.4f}")
print(f"  F_s log1p:      {F_s_log:.4f}")
print()
if necesita_log:
    print(f"  ✅ APLICAR log1p — varianza creciente detectada")
    print(f"     Modelar con serie log-transformada para SARIMA/SARIMAX")
else:
    print(f"  ⚠️ log1p OPCIONAL — heterocedasticidad no conclusiva")
    print(f"     Serie original es viable; log1p no mejora la estabilidad de forma significativa")""")

# ════════════════════════════════════════════════════════════
# CELDA 9 — Fase III Header (MD)
# ════════════════════════════════════════════════════════════
md(r"""---

## Fase III — Modelación de Dinámicas por Vertical de Negocio

### 3.1 Licores y Cigarrillos
Se deflacta el recaudo nominal con el IPC para separar:
- **Crecimiento orgánico** (volumen real): ¿se consume más o menos?
- **Efecto precio** (inflación): ¿el recaudo crece solo porque suben los precios?

Referencia complementaria: la **ENCSPA** (DANE — Encuesta Nacional de Consumo de
Sustancias Psicoactivas) reporta cambios en los patrones de consumo de alcohol y tabaco.

### 3.2 Juegos de Azar y Apuestas
Se cruza el recaudo con el **Salario Mínimo (SMLV)** y se estima la
**elasticidad ingreso** ($\beta$ de regresión log-log) para determinar si el
consumo de azar es inelástico frente a la pérdida de poder adquisitivo.""")

# ════════════════════════════════════════════════════════════
# CELDA 10 — Licores y Cigarrillos (Code)
# ════════════════════════════════════════════════════════════
code(r"""# ══════════════════════════════════════════════════════════════
# FASE III.1 — Licores y Cigarrillos: Serie Deflactada
# ══════════════════════════════════════════════════════════════

# ── Identificar categorías ──
mask_lic = df_raw['NombreSubGrupoFuente'].str.contains(
    r'licor|cigarr|tabaco|cerveza|alcohol|bebida',
    case=False, na=False)
cats_lic = df_raw.loc[mask_lic, 'NombreSubGrupoFuente'].unique()

if len(cats_lic) == 0:
    # Si no hay coincidencia literal, mostrar todas y buscar ampliamente
    print("⚠️ No se encontraron categorías con keywords exactas.")
    print("   Categorías disponibles:")
    for c in df_raw['NombreSubGrupoFuente'].unique():
        print(f"     • {c}")
    # Intentar match más amplio (rentas típicas colombianas)
    mask_lic = df_raw['NombreSubGrupoFuente'].str.contains(
        r'impuesto.*consumo|IVA.*licor|sobretasa|participaci',
        case=False, na=False)
    cats_lic = df_raw.loc[mask_lic, 'NombreSubGrupoFuente'].unique()

print(f"Categorías identificadas como Licores/Cigarrillos/Cerveza:")
for c in cats_lic:
    n = (df_raw['NombreSubGrupoFuente'] == c).sum()
    v = df_raw.loc[df_raw['NombreSubGrupoFuente'] == c, COL_VALOR].sum()
    print(f"  • {c}  ({n:,} reg, ${v/1e9:.1f}MM)")

# ── Serie mensual por vertical ──
df_lic = df_raw[mask_lic].copy()
serie_lic = (df_lic.groupby(pd.Grouper(key=COL_FECHA, freq='MS'))
             [COL_VALOR].sum())
serie_lic.name = 'Licores_Cigarrillos'

# ── Deflactar con IPC (índice acumulado, base oct 2021 = 100) ──
ipc_idx = pd.Series(100.0, index=serie_lic.index)
for i in range(1, len(ipc_idx)):
    year = ipc_idx.index[i].year
    ipc_anual = MACRO_DATA.get(year, MACRO_DATA[max(MACRO_DATA)])['IPC']
    ipc_idx.iloc[i] = ipc_idx.iloc[i-1] * (1 + ipc_anual / 100 / 12)

serie_lic_real = serie_lic / ipc_idx * 100

# ── Crecimiento YoY nominal vs real ──
yoy_nom = serie_lic.pct_change(12) * 100
yoy_rea = serie_lic_real.pct_change(12) * 100
efecto_p = yoy_nom - yoy_rea

# ── Gráfica ──
fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

ax = axes[0]
ax.plot(serie_lic.index, serie_lic.values/1e9, color=C_PRIMARY,
        lw=2, label='Nominal')
ax.plot(serie_lic_real.index, serie_lic_real.values/1e9, color=C_QUATERNARY,
        lw=2, ls='--', label='Real (deflactado IPC)')
ax.fill_between(serie_lic.index,
                serie_lic_real.values/1e9, serie_lic.values/1e9,
                alpha=0.15, color=C_SECONDARY, label='Efecto inflación')
ax.set_ylabel('Recaudo (miles MM$)', fontdict=FONT_AXIS)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
if _VIZ_THEME_LOADED:
    titulo_profesional(ax, 'Licores y Cigarrillos: Nominal vs Real',
                       'Deflactado con IPC — Base oct 2021 = 100')
    formato_pesos_eje(ax, eje='y')

ax = axes[1]
vld = ~yoy_nom.isna()
ax.bar(yoy_nom.index[vld], efecto_p.values[vld], width=25,
       color=C_SECONDARY, alpha=0.6, label='Efecto precio (inflación)')
ax.bar(yoy_nom.index[vld], yoy_rea.values[vld], width=25,
       color=C_QUATERNARY, alpha=0.6, bottom=efecto_p.values[vld],
       label='Crecimiento orgánico (volumen)')
ax.axhline(y=0, color='black', lw=0.8)
ax.set_ylabel('Crecimiento YoY (%)', fontdict=FONT_AXIS)
ax.set_xlabel('Fecha', fontdict=FONT_AXIS)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
if _VIZ_THEME_LOADED:
    titulo_profesional(ax, 'Descomposición del Crecimiento Interanual',
                       'Orgánico (volumen) vs Inflacionario (efecto precio)')

plt.tight_layout()
if _VIZ_THEME_LOADED:
    marca_agua(fig)
    guardar_figura(fig, '02_licores_deflactado', OUTPUTS_FIGURES)
plt.show()

# ── Reporte ──
if vld.sum() > 0:
    cn = yoy_nom[vld].mean()
    cr = yoy_rea[vld].mean()
    ep = efecto_p[vld].mean()
    print(f"\n{'═'*70}")
    print(f"LICORES Y CIGARRILLOS — CRECIMIENTO")
    print(f"{'═'*70}")
    print(f"  Crec. nominal medio YoY:  {cn:+.2f}%")
    print(f"  Crec. real medio YoY:     {cr:+.2f}%")
    print(f"  Efecto precio medio:      {ep:+.2f}%")
    if cr > 0:
        print(f"\n  ✅ Crecimiento orgánico POSITIVO: mayor consumo real")
    else:
        print(f"\n  ⚠️ Crecimiento orgánico NEGATIVO: solo crece por efecto precio")
        print(f"     Implicación ENCSPA (DANE): posible contracción del consumo")""")

# ════════════════════════════════════════════════════════════
# CELDA 11 — Juegos de Azar (Code)
# ════════════════════════════════════════════════════════════
code(r"""# ══════════════════════════════════════════════════════════════
# FASE III.2 — Juegos de Azar y Apuestas: Elasticidad al Ingreso
# ══════════════════════════════════════════════════════════════

mask_azar = df_raw['NombreSubGrupoFuente'].str.contains(
    r'juego|azar|apuesta|loter|chance|suerte',
    case=False, na=False)
cats_azar = df_raw.loc[mask_azar, 'NombreSubGrupoFuente'].unique()

if len(cats_azar) == 0:
    print("⚠️ No se encontraron categorías de azar con keywords exactas.")
    print("   Se usará la serie total como proxy para el análisis de elasticidad.")
    serie_azar = serie.copy()
    serie_azar.name = 'Total_como_proxy'
else:
    print("Categorías identificadas como Juegos de Azar:")
    for c in cats_azar:
        n = (df_raw['NombreSubGrupoFuente'] == c).sum()
        print(f"  • {c}  ({n:,} registros)")
    df_az = df_raw[mask_azar].copy()
    serie_azar = (df_az.groupby(pd.Grouper(key=COL_FECHA, freq='MS'))
                  [COL_VALOR].sum())
    serie_azar.name = 'Juegos_Azar'

# ── SMLV mensual (valores nominales en COP) ──
smlv_vals = {2021: 908526, 2022: 1000000, 2023: 1160000,
             2024: 1300000, 2025: 1423500, 2026: 1750905}
smlv_m = pd.Series(index=serie_azar.index, dtype=float)
for f in smlv_m.index:
    smlv_m[f] = smlv_vals.get(f.year, smlv_vals[max(smlv_vals)])

# Recaudo en "unidades de SMLV"
serie_azar_smlv = serie_azar / smlv_m

# ── Elasticidad ingreso (regresión log-log) ──
#   log(Recaudo) = alpha + beta * log(SMLV) + eps
#   beta < 1: inelástico | beta > 1: elástico
pos = serie_azar > 0
elasticidad = np.nan
if pos.sum() > 12:
    log_r = np.log(serie_azar[pos])
    log_s = np.log(smlv_m[pos])
    slope, intercept, r_val, p_el, se = stats.linregress(
        log_s.values, log_r.values)
    elasticidad = slope

# ── Gráfica (3 paneles) ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Nominal vs SMLV normalizado
ax = axes[0]
ax.plot(serie_azar.index, serie_azar.values/1e9, color=C_SENARY,
        lw=2, label='Nominal (MM$)')
ax.set_ylabel('Recaudo nominal (miles MM$)', fontdict=FONT_AXIS,
              color=C_SENARY)
ax.legend(loc='upper left', fontsize=8)
ax_r = ax.twinx()
ax_r.plot(serie_azar_smlv.index, serie_azar_smlv.values/1e3,
          color=C_QUINARY, lw=2, ls='--', label='En SMLV (miles)')
ax_r.set_ylabel('Recaudo / SMLV (miles)', fontdict=FONT_AXIS,
                color=C_QUINARY)
ax_r.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)
if _VIZ_THEME_LOADED:
    titulo_profesional(ax, 'Juegos de Azar: Nominal vs SMLV',
                       'Recaudo en poder adquisitivo constante')

# Panel 2: Scatter elasticidad
ax = axes[1]
if pos.sum() > 12:
    ax.scatter(log_s.values, log_r.values, color=C_SENARY, s=40,
               alpha=0.7, edgecolors='white')
    x_fit = np.linspace(log_s.min(), log_s.max(), 50)
    lbl = 'inelástico' if abs(elasticidad) < 1 else 'elástico'
    ax.plot(x_fit, intercept + slope * x_fit, '--', color=C_SECONDARY,
            lw=2, label=f'beta = {elasticidad:.3f} ({lbl})')
    ax.set_xlabel('log(SMLV)', fontdict=FONT_AXIS)
    ax.set_ylabel('log(Recaudo Azar)', fontdict=FONT_AXIS)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    if _VIZ_THEME_LOADED:
        titulo_profesional(ax, 'Elasticidad Ingreso-Recaudo',
                           f'beta={elasticidad:.3f}, '
                           f'R2={r_val**2:.3f}, p={p_el:.4f}')

# Panel 3: YoY azar vs SMLV
ax = axes[2]
yoy_az = serie_azar.pct_change(12) * 100
yoy_sm = smlv_m.pct_change(12) * 100
vld2 = ~yoy_az.isna() & ~yoy_sm.isna()
if vld2.sum() > 0:
    ax.bar(yoy_az.index[vld2], yoy_az.values[vld2], width=25,
           color=C_SENARY, alpha=0.7, label='YoY Azar (%)')
    ax.plot(yoy_sm.index[vld2], yoy_sm.values[vld2], color=C_SECONDARY,
            lw=2.5, marker='o', ms=3, label='YoY SMLV (%)')
    ax.axhline(y=0, color='black', lw=0.8)
    ax.set_ylabel('Variación YoY (%)', fontdict=FONT_AXIS)
    ax.set_xlabel('Fecha', fontdict=FONT_AXIS)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
    if _VIZ_THEME_LOADED:
        titulo_profesional(ax, 'Crecimiento: Azar vs Salario Mínimo',
                           '¿El recaudo de azar acompaña al SMLV?')

plt.tight_layout()
if _VIZ_THEME_LOADED:
    marca_agua(fig)
    guardar_figura(fig, '02_juegos_azar_elasticidad', OUTPUTS_FIGURES)
plt.show()

# ── Reporte ──
print(f"\n{'═'*70}")
print(f"JUEGOS DE AZAR — ELASTICIDAD")
print(f"{'═'*70}")
if not np.isnan(elasticidad):
    print(f"  Elasticidad ingreso (beta): {elasticidad:.4f}")
    print(f"  R²:                         {r_val**2:.4f}")
    print(f"  p-valor:                    {p_el:.4f}")
    if abs(elasticidad) < 1:
        print(f"\n  ✅ INELÁSTICO: gasto en azar no responde al ingreso")
        print(f"     Los hogares mantienen su gasto incluso con pérdida adquisitiva")
        print(f"     → Ingreso fiscal relativamente estable en este segmento")
    else:
        print(f"\n  ⚠️ ELÁSTICO: gasto en azar RESPONDE al ingreso")
        print(f"     → Mayor sensibilidad a ciclos económicos")
else:
    print(f"  ⚠️ Elasticidad no calculable (datos insuficientes o negativos)")""")

# ════════════════════════════════════════════════════════════
# CELDA 12 — Fase IV Header (MD)
# ════════════════════════════════════════════════════════════
md(r"""---

## Fase IV — Tratamiento de Anomalías y Quiebres Estructurales

### 4.1 Change Point Detection
La migración del sistema ERP de **Dynamics a Oracle** durante 2025 puede generar
picos artificiales (doble registro, cierres anticipados, acumulaciones) que sesguen
la estacionalidad estimada.

Se aplica detección de puntos de cambio mediante:
1. **CUSUM** (Cumulative Sum Control Chart) sobre los residuos STL
2. **Rolling Window** — cambios bruscos en la media y volatilidad
3. **Test de Levene + Welch** — comparación formal pre-2025 vs 2025

### 4.2 Clasificación de Valores Negativos
Los registros negativos se clasifican por `TipoRegistro` para confirmar que son
**ajustes contables presupuestales** y no errores de captura.""")

# ════════════════════════════════════════════════════════════
# CELDA 13 — Change Point Detection (Code)
# ════════════════════════════════════════════════════════════
code(r"""# ══════════════════════════════════════════════════════════════
# FASE IV.1 — Detección de Puntos de Cambio
# ══════════════════════════════════════════════════════════════

# ── CUSUM sobre residuos STL ──
residuos = res_stl.resid
mu_r  = residuos.mean()
sig_r = residuos.std()
cusum = np.cumsum(residuos.values - mu_r)
h_umbral = 4 * sig_r          # umbral Hawkins

# ── Rolling window (cambios bruscos) ──
ventana = 6
roll_mu  = serie.rolling(ventana, center=True).mean()
roll_sig = serie.rolling(ventana, center=True).std()
diff_mu  = roll_mu.diff().abs()
umbral90 = diff_mu.quantile(0.90)
cambios  = serie.index[diff_mu > umbral90]

# ── Test formal pre-2025 vs 2025 ──
pre  = serie[serie.index.year < 2025]
post = serie[serie.index.year >= 2025]
if len(post) > 3 and len(pre) > 3:
    stat_lev, p_lev = stats.levene(pre.values, post.values)
    stat_wel, p_wel = stats.ttest_ind(pre.values, post.values, equal_var=False)
else:
    stat_lev = p_lev = stat_wel = p_wel = np.nan

# ── Gráfica (3 paneles) ──
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

# Panel 1: Serie + change points
ax = axes[0]
ax.plot(serie.index, serie.values/1e9, color=C_PRIMARY, lw=1.5, label='Recaudo')
ax.plot(roll_mu.index, roll_mu.values/1e9, color=C_SECONDARY, lw=2, ls='--',
        label=f'MA({ventana})')
erp_s = pd.Timestamp('2025-01-01')
erp_e = pd.Timestamp('2025-12-31')
ax.axvspan(erp_s, erp_e, alpha=0.10, color=C_HIGHLIGHT,
           label='Migración ERP (2025)')
for cp in cambios:
    ax.axvline(x=cp, color=C_SECONDARY, alpha=0.3, lw=0.8, ls=':')
ax.set_ylabel('Recaudo (miles MM$)', fontdict=FONT_AXIS)
ax.legend(fontsize=8, loc='upper left'); ax.grid(True, alpha=0.3)
if _VIZ_THEME_LOADED:
    titulo_profesional(ax, 'Detección de Puntos de Cambio',
                       'Líneas punteadas = cambios bruscos en media móvil')
    formato_pesos_eje(ax, eje='y')

# Panel 2: CUSUM
ax = axes[1]
ax.plot(serie.index, cusum/1e9, color=C_TERTIARY, lw=1.5, label='CUSUM')
ax.axhline(y= h_umbral/1e9, color=C_SECONDARY, ls='--', alpha=0.7,
           label=f'Umbral ±{h_umbral/1e9:.0f}MM$')
ax.axhline(y=-h_umbral/1e9, color=C_SECONDARY, ls='--', alpha=0.7)
ax.axhline(y=0, color='black', lw=0.5)
ax.fill_between(serie.index, -h_umbral/1e9, h_umbral/1e9,
                alpha=0.05, color=C_QUATERNARY)
ax.axvspan(erp_s, erp_e, alpha=0.10, color=C_HIGHLIGHT)
ax.set_ylabel('CUSUM (miles MM$)', fontdict=FONT_AXIS)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
if _VIZ_THEME_LOADED:
    titulo_profesional(ax, 'CUSUM de Residuos STL',
                       'Desviación acumulada del residuo respecto a su media')

# Panel 3: Volatilidad rolling
ax = axes[2]
ax.plot(roll_sig.index, roll_sig.values/1e9, color=C_SENARY, lw=1.5,
        label=f'sigma rolling ({ventana}M)')
ax.axvspan(erp_s, erp_e, alpha=0.10, color=C_HIGHLIGHT, label='Migración ERP')
ax.axhline(y=pre.std()/1e9, color=C_TERTIARY, ls='--',
           label=f'sigma pre-2025 ({pre.std()/1e9:.0f}MM$)')
if len(post) > 1:
    ax.axhline(y=post.std()/1e9, color=C_SECONDARY, ls='--',
               label=f'sigma 2025 ({post.std()/1e9:.0f}MM$)')
ax.set_ylabel('Volatilidad (miles MM$)', fontdict=FONT_AXIS)
ax.set_xlabel('Fecha', fontdict=FONT_AXIS)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
if _VIZ_THEME_LOADED:
    titulo_profesional(ax, 'Evolución de Volatilidad',
                       f'Levene: F={stat_lev:.3f}, p={p_lev:.4f}')

plt.tight_layout()
if _VIZ_THEME_LOADED:
    marca_agua(fig)
    guardar_figura(fig, '02_change_point_detection', OUTPUTS_FIGURES)
plt.show()

# ── Reporte ──
print(f"\n{'═'*70}")
print(f"DETECCIÓN DE PUNTOS DE CAMBIO")
print(f"{'═'*70}")
print(f"  Puntos de cambio brusco: {len(cambios)}")
if len(cambios) > 0:
    for cp in cambios[:10]:
        v = serie.get(cp, np.nan)
        print(f"    {cp:%Y-%m}  ${v/1e9:.1f}MM")
print(f"\n  Pre-2025 vs 2025:")
print(f"    Media pre-2025: ${pre.mean()/1e9:.1f}MM")
if len(post) > 0:
    print(f"    Media 2025:     ${post.mean()/1e9:.1f}MM")
print(f"    Levene (var):   F = {stat_lev:.3f}  p = {p_lev:.4f}")
print(f"    Welch  (media): t = {stat_wel:.3f}  p = {p_wel:.4f}")
if not np.isnan(p_lev) and p_lev < 0.05:
    print(f"\n  ⚠️ CAMBIO SIGNIFICATIVO en varianza post-ERP")
    print(f"     → Incluir dummy 'ERP_migration' en SARIMAX")
else:
    print(f"\n  ✅ Sin cambio significativo — migración ERP no distorsiona estacionalidad")""")

# ════════════════════════════════════════════════════════════
# CELDA 14 — Negativos classification (Code)
# ════════════════════════════════════════════════════════════
code(r"""# ══════════════════════════════════════════════════════════════
# FASE IV.2 — Clasificación de Valores Negativos por TipoRegistro
# ══════════════════════════════════════════════════════════════

negativos = df_raw[df_raw[COL_VALOR] < 0].copy()

print(f"{'═'*70}")
print(f"CLASIFICACIÓN DE VALORES NEGATIVOS")
print(f"{'═'*70}")
print(f"  Total negativos:     {len(negativos):,}")
print(f"  Suma:                ${negativos[COL_VALOR].sum():,.0f}")
print(f"  % del recaudo bruto: "
      f"{negativos[COL_VALOR].sum() / df_raw[COL_VALOR].sum() * 100:.4f}%")

# ── Distribución por TipoRegistro ──
print(f"\n{'─'*70}")
print(f"POR TipoRegistro:")
tipo_dist = negativos.groupby('TipoRegistro').agg(
    Registros=(COL_VALOR, 'count'),
    Suma=(COL_VALOR, 'sum'),
    Media=(COL_VALOR, 'mean'),
    Min=(COL_VALOR, 'min'),
    Max=(COL_VALOR, 'max'))
for tipo, row in tipo_dist.iterrows():
    pct = row['Registros'] / len(negativos) * 100
    print(f"\n  Tipo: {tipo}")
    print(f"    Registros: {row['Registros']:,}  ({pct:.1f}%)")
    print(f"    Suma:      ${row['Suma']:,.0f}")
    print(f"    Rango:     [${row['Min']:,.0f},  ${row['Max']:,.0f}]")

# ── Distribución por NombreSubGrupoFuente ──
print(f"\n{'─'*70}")
print(f"POR NombreSubGrupoFuente:")
fuente_d = (negativos.groupby('NombreSubGrupoFuente')[COL_VALOR]
            .agg(['count', 'sum']).sort_values('sum'))
for fuente, row in fuente_d.iterrows():
    print(f"  {fuente:<55} {row['count']:>5} reg  "
          f"${row['sum']:>15,.0f}")

# ── Distribución temporal ──
neg_m = (negativos.groupby(pd.Grouper(key=COL_FECHA, freq='MS'))
         [COL_VALOR].agg(['count', 'sum']))

fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DUAL)

ax = axes[0]
if len(neg_m) > 0:
    ax.bar(neg_m.index, neg_m['count'], width=25,
           color=C_NEGATIVE, alpha=0.7)
ax.set_ylabel('Nro. registros negativos', fontdict=FONT_AXIS)
ax.set_xlabel('Fecha', fontdict=FONT_AXIS)
ax.grid(True, alpha=0.3, axis='y')
if _VIZ_THEME_LOADED:
    titulo_profesional(ax, 'Frecuencia de Registros Negativos',
                       'Distribución temporal de ajustes contables')

ax = axes[1]
if len(tipo_dist) > 0:
    tipos  = tipo_dist.index.tolist()
    vals   = tipo_dist['Registros'].values
    colors = [C_SECONDARY, C_TERTIARY, C_QUATERNARY,
              C_QUINARY, C_SENARY][:len(tipos)]
    wedges, txt, atxt = ax.pie(vals, labels=tipos, autopct='%1.1f%%',
                               colors=colors, startangle=90)
    for t in txt:  t.set_fontsize(9)
    for a in atxt: a.set_fontsize(8); a.set_fontweight('bold')
if _VIZ_THEME_LOADED:
    titulo_profesional(ax, 'Negativos por TipoRegistro',
                       f'Total: {len(negativos):,} registros')

plt.tight_layout()
if _VIZ_THEME_LOADED:
    marca_agua(fig)
    guardar_figura(fig, '02_negativos_clasificacion', OUTPUTS_FIGURES)
plt.show()

# ── Conclusión ──
main_tipo = tipo_dist['Registros'].idxmax() if len(tipo_dist) > 0 else 'N/A'
pct_main  = (tipo_dist.loc[main_tipo, 'Registros'] / len(negativos) * 100
             if main_tipo != 'N/A' else 0)
print(f"\n{'═'*70}")
print(f"CONCLUSIÓN")
print(f"{'═'*70}")
if pct_main > 80:
    print(f"  ✅ {pct_main:.1f}% de los negativos son «{main_tipo}»")
    print(f"     → Ajustes contables legítimos, NO errores de captura")
    print(f"     → Se MANTIENEN para reflejar la dinámica fiscal neta")
else:
    print(f"  ⚠️ Tipo principal: «{main_tipo}» ({pct_main:.1f}%)")
    print(f"     Negativos distribuidos entre varios tipos")
    print(f"     → Se mantienen; revisar con la entidad si necesario")""")

# ════════════════════════════════════════════════════════════
# CELDA 15 — Fase V Header (MD)
# ════════════════════════════════════════════════════════════
md(r"""---

## Fase V — Validación de Estacionariedad (ADF + KPSS)

### Marco Teórico

| Test | $H_0$ | $H_1$ | Interpretación |
|------|--------|--------|----------------|
| **ADF** | Raíz unitaria (no estacionaria) | Estacionaria | Rechazo $H_0$ → estacionaria |
| **KPSS** | Estacionaria (o de tendencia) | Raíz unitaria | Rechazo $H_0$ → NO estacionaria |

**Estrategia de confirmación** (Hyndman, 2021):
- ADF rechaza **Y** KPSS no rechaza → **Estacionaria** ✅
- ADF no rechaza **Y** KPSS rechaza → **No estacionaria** → diferenciar
- Resultados contradictorios → más datos o transformación

Se evalúan 4 variantes:
1. Serie original $Y_t$
2. Primera diferencia $\Delta Y_t\;(d{=}1)$
3. Diferencia estacional $\Delta_{12} Y_t\;(D{=}1)$
4. Doble diferencia $\Delta\Delta_{12} Y_t\;(d{=}1, D{=}1)$""")

# ════════════════════════════════════════════════════════════
# CELDA 16 — ADF + KPSS (Code)
# ════════════════════════════════════════════════════════════
code(r"""# ══════════════════════════════════════════════════════════════
# FASE V — VALIDACIÓN DE ESTACIONARIEDAD (ADF + KPSS)
# ══════════════════════════════════════════════════════════════

def test_est(s, nombre):
    r = {'nombre': nombre}
    try:
        a_stat, a_p, a_lags, *_ = adfuller(s.dropna(), autolag='AIC')
        r['adf_stat'] = a_stat; r['adf_p'] = a_p
        r['adf_ok'] = a_p < 0.05
    except Exception as e:
        r['adf_stat'] = r['adf_p'] = np.nan; r['adf_ok'] = None
        print(f"  ⚠️ ADF falló para {nombre}: {e}")
    try:
        k_stat, k_p, *_ = kpss(s.dropna(), regression='c', nlags='auto')
        r['kpss_stat'] = k_stat; r['kpss_p'] = k_p
        r['kpss_ok'] = k_p > 0.05   # no rechazo = estacionaria
    except Exception as e:
        r['kpss_stat'] = r['kpss_p'] = np.nan; r['kpss_ok'] = None
        print(f"  ⚠️ KPSS falló para {nombre}: {e}")
    return r

# Variantes
variantes = [
    (serie,                            'Original Y_t'),
    (serie.diff().dropna(),            'Diff regular (d=1)'),
    (serie.diff(12).dropna(),          'Diff estacional (D=1)'),
    (serie.diff(12).diff().dropna(),   'Doble diff (d=1, D=1)'),
]
resultados = []
for s, n in variantes:
    if len(s.dropna()) >= 12:
        resultados.append(test_est(s, n))
    else:
        print(f"  ⚠️ {n}: insuficientes datos ({len(s.dropna())})")

# ── Tabla ──
print(f"\n{'═'*90}")
print(f"RESULTADOS DE ESTACIONARIEDAD")
print(f"{'═'*90}")
header = (f"{'Serie':<35} {'ADF stat':>10} {'ADF p':>8} {'ADF':>5} "
          f"{'KPSS stat':>10} {'KPSS p':>8} {'KPSS':>5} {'Veredicto':>14}")
print(header)
print('─' * 90)
for r in resultados:
    a_ico = '✅' if r.get('adf_ok') else '❌'
    k_ico = '✅' if r.get('kpss_ok') else '❌'
    if r.get('adf_ok') and r.get('kpss_ok'):
        vdct = 'ESTAC.  ✅'
    elif not r.get('adf_ok') and not r.get('kpss_ok'):
        vdct = 'NO EST. ❌'
    else:
        vdct = 'AMBIGUO ⚠️'
    print(f"{r['nombre']:<35} {r.get('adf_stat', np.nan):>10.4f} "
          f"{r.get('adf_p', np.nan):>8.4f} {a_ico:>5} "
          f"{r.get('kpss_stat', np.nan):>10.4f} "
          f"{r.get('kpss_p', np.nan):>8.4f} {k_ico:>5} {vdct:>14}")

# ── Gráfica 2×2 ──
fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_QUAD)
axes = axes.flatten()
cols_v = [C_PRIMARY, C_SECONDARY, C_TERTIARY, C_SENARY]
for i, ((s, n), col) in enumerate(zip(variantes, cols_v)):
    ax = axes[i]
    sc = s.dropna()
    if len(sc) > 0:
        ax.plot(sc.index, sc.values/1e9, color=col, lw=1.5)
        ax.axhline(y=sc.mean()/1e9, color='black', ls='--', alpha=0.5, lw=0.8)
        ax.fill_between(sc.index,
                        (sc.mean() - 2*sc.std())/1e9,
                        (sc.mean() + 2*sc.std())/1e9,
                        alpha=0.1, color=col)
    r = resultados[i] if i < len(resultados) else {}
    sub = (f"ADF p={r.get('adf_p', np.nan):.4f} | "
           f"KPSS p={r.get('kpss_p', np.nan):.4f}")
    if _VIZ_THEME_LOADED:
        titulo_profesional(ax, n, sub)
        formato_pesos_eje(ax, eje='y')
    else:
        ax.set_title(f'{n}\n{sub}', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Valor (miles MM$)', fontsize=9)

plt.suptitle('Validación de Estacionariedad — ADF + KPSS',
             fontsize=15, fontweight='bold', y=1.01, fontfamily='serif')
plt.tight_layout()
if _VIZ_THEME_LOADED:
    marca_agua(fig)
    guardar_figura(fig, '02_estacionariedad_adf_kpss', OUTPUTS_FIGURES)
plt.show()

# ── Recomendación SARIMAX ──
d_opt, D_opt = 0, 0
if len(resultados) >= 1 and resultados[0].get('adf_ok') and resultados[0].get('kpss_ok'):
    rec = 'Serie original estacionaria → d=0, D=0'
elif len(resultados) >= 2 and resultados[1].get('adf_ok') and resultados[1].get('kpss_ok'):
    d_opt = 1; rec = 'Diff regular estacionaria → d=1, D=0'
elif len(resultados) >= 3 and resultados[2].get('adf_ok') and resultados[2].get('kpss_ok'):
    D_opt = 1; rec = 'Diff estacional estacionaria → d=0, D=1'
elif len(resultados) >= 4 and resultados[3].get('adf_ok') and resultados[3].get('kpss_ok'):
    d_opt = 1; D_opt = 1; rec = 'Doble diff estacionaria → d=1, D=1'
else:
    d_opt = 1; D_opt = 1; rec = 'Ninguna variante es clara → d=1, D=1 (conservador)'

print(f"\n{'═'*70}")
print(f"RECOMENDACIÓN PARA SARIMAX")
print(f"{'═'*70}")
print(f"  {rec}")
print(f"  Orden sugerido: SARIMA(p, {d_opt}, q)(P, {D_opt}, Q)[{ESTACIONALIDAD}]")
print(f"  → p, q, P, Q se determinan en notebook 04 (ACF/PACF + AIC/BIC)")""")

# ════════════════════════════════════════════════════════════
# CELDA 17 — Conclusiones (MD)
# ════════════════════════════════════════════════════════════
md(r"""---

## Conclusiones del Análisis de Estacionalidad

### Hallazgos Principales

| Análisis | Resultado | Implicación para Modelado |
|----------|-----------|---------------------------|
| **CCF (Lag)** | Rezago óptimo identificado | Confirma/rechaza hipótesis Dic→Ene, Jun→Jul |
| **STL Avanzado** | F_s cuantificado | Perfil estacional fiscal con patrón robusto cada 12 meses |
| **Heterocedasticidad** | Evaluación log1p | Decisión sobre transformación para SARIMA |
| **Licores/Cigarrillos** | Crecimiento orgánico vs inflacionario | Separación efecto volumen vs precio |
| **Juegos de Azar** | Elasticidad ingreso cuantificada | Base para pronóstico con variables exógenas |
| **Change Points** | Quiebres 2025 evaluados | Dummy para migración ERP si significativo |
| **Valores Negativos** | Clasificados por TipoRegistro | Confirmación de ajustes contables legítimos |
| **ADF + KPSS** | Orden (d, D) determinado | Parámetros de diferenciación para SARIMAX |

### Parámetros para el Modelado (Notebooks 04–08)

```
Variables exógenas candidatas:
  - IPC (deflactado) con lag identificado por CCF
  - SMLV (si juegos de azar es inelástico, incluir como control)
  - Dummy ERP_migration (si cambio significativo en 2025)
  - Dummies is_festivity (Jun, Dic) e is_peak (Ene, Jul)
```

> **Siguiente paso**: `03_Correlacion_Macro.ipynb` — Análisis formal de correlación
> multivariada y selección de regresores exógenos para SARIMAX.""")

# ════════════════════════════════════════════════════════════
# GUARDAR NOTEBOOK
# ════════════════════════════════════════════════════════════
out = Path(__file__).resolve().parent.parent / 'notebooks' / '02_Estacionalidad.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

n_md   = sum(1 for c in nb.cells if c.cell_type == 'markdown')
n_code = sum(1 for c in nb.cells if c.cell_type == 'code')
print(f"✅ Notebook generado: {out}")
print(f"   Celdas: {len(nb.cells)} ({n_md} MD + {n_code} Code)")

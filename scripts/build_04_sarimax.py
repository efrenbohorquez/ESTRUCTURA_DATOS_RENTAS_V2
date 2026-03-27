"""
build_04_sarimax.py — Genera 04_SARIMAX.ipynb
==============================================
Ejecutar:  python scripts/build_04_sarimax.py

Estructura (12 celdas: 6 MD + 6 Code, 6 fases):
  Fase I   — Carga de datos, Split temporal, transformación log1p
  Fase II  — Identificación de orden óptimo (ACF/PACF + auto_arima)
  Fase III — Ajuste SARIMAX con variables exógenas (IPC)
  Fase IV  — Validación OOS Oct–Dic 2025 vs datos REALES
  Fase V   — Diagnóstico de residuos
  Fase VI  — Pronóstico de Producción 2026 + Exportación
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
# CELDA 1 — MARKDOWN: Encabezado y Arquitectura
# ════════════════════════════════════════════════════════════════
md(r"""# 04 — SARIMAX: Modelo Econométrico con Variables Exógenas

**Sistema de Análisis y Pronóstico de Rentas Cedidas** | ADRES — Colombia

---

## Arquitectura Analítica

| Fase | Contenido | Método |
|------|-----------|--------|
| **I** | Carga y preparación Ene 2022 – Dic 2025 | Split 45/3, transformación log1p |
| **II** | Identificación de orden óptimo | ACF/PACF + `auto_arima` AIC/BIC |
| **III** | Ajuste SARIMAX con exógenas | IPC como regresor, dummies estacionales |
| **IV** | Validación OOS Oct–Dic 2025 | Pronóstico vs datos REALES, MAPE/RMSE/MAE |
| **V** | Diagnóstico de residuos | Ljung-Box, Shapiro-Wilk, heteroscedasticidad |
| **VI** | Pronóstico de producción 2026 | Reentreno completo + 12 meses + IC 95% |

### Justificación Metodológica

**SARIMAX** (Seasonal AutoRegressive Integrated Moving Average with eXogenous
regressors) extiende el marco clásico de Box-Jenkins incorporando variables
macroeconómicas como regresores. Para Rentas Cedidas, el **IPC** es la
variable exógena más relevante: el recaudo depende de impuestos *ad valorem*
sobre licores, cigarrillos y juegos de azar, cuyo monto nominal se ajusta
con los precios al consumidor.

**Periodo de análisis:** Ene 2022 – Sep 2025 (45 meses) como entrenamiento.
Se excluye Oct-Dic 2021 por constituir un quiebre estructural (datos planos
post-pandemia que rompen la estacionalidad reproducible).

**Transformación log1p:** La serie presenta heterocedasticidad y asimetría
(CV ≈ 0.34). En escala logarítmica, la estacionalidad multiplicativa se
convierte en aditiva, cumpliendo mejor los supuestos de SARIMAX.

> **Perfil Estacional:** Los picos de Ene y Jul reflejan el
> mecanismo de recaudo mes vencido. SARIMAX captura esta estacionalidad
> mediante los componentes $(P, D, Q)_{12}$.
""")


# ════════════════════════════════════════════════════════════════
# CELDA 2 — CODE: Fase I — Setup y Carga
# ════════════════════════════════════════════════════════════════
code(r"""# ══════════════════════════════════════════════════════════════
# FASE I — Setup, Carga de Datos y Split Temporal
# ══════════════════════════════════════════════════════════════

%run 00_config.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from sklearn.metrics import (mean_absolute_percentage_error,
                             mean_squared_error, mean_absolute_error)

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pm
import warnings
warnings.filterwarnings('ignore')

# ── Carga de serie mensual procesada ──
csv_path = DATA_PROCESSED / 'serie_mensual.csv'
df_serie = pd.read_csv(csv_path, parse_dates=['Fecha'], index_col='Fecha')
df_serie.index.freq = 'MS'

# Cargar macro
csv_macro = DATA_PROCESSED / 'serie_mensual_macro.csv'
df_macro = pd.read_csv(csv_macro, parse_dates=['Fecha'], index_col='Fecha')
df_macro.index.freq = 'MS'

# Filtrar periodo completo (Oct 2021 – Dic 2025)
serie_full = df_serie['Recaudo_Total'].loc[FECHA_INICIO:FECHA_FIN].copy()
serie_full.name = 'Recaudo_Total'

exog_full = df_macro[['IPC_Idx']].loc[FECHA_INICIO:FECHA_FIN].copy()

# ── Split Train/Test ──
train = serie_full.loc[:TRAIN_END]
test  = serie_full.loc[TEST_START:VALIDATION_END]

exog_train = exog_full.loc[:TRAIN_END]
exog_test  = exog_full.loc[TEST_START:VALIDATION_END]

# ── Transformación log1p ──
train_log = np.log1p(train)
test_log  = np.log1p(test)

print(f"{'═'*70}")
print(f"PREPARACIÓN DE LA SERIE PARA MODELADO SARIMAX")
print(f"{'═'*70}")
print(f"  Serie completa: {len(serie_full)} meses ({serie_full.index.min().date()} → {serie_full.index.max().date()})")
print(f"  Entrenamiento:  {len(train)} meses ({train.index.min().date()} → {train.index.max().date()})")
print(f"  Prueba:         {len(test)} meses ({test.index.min().date()} → {test.index.max().date()})")
print(f"  Ratio:          {len(train)/len(serie_full)*100:.1f}% / {len(test)/len(serie_full)*100:.1f}%")

print(f"\n{'─'*70}")
print(f"ESTADÍSTICAS — Entrenamiento")
print(f"{'─'*70}")
print(f"  Media:     ${train.mean()/1e9:,.1f} MM COP")
print(f"  Mediana:   ${train.median()/1e9:,.1f} MM COP")
print(f"  Std:       ${train.std()/1e9:,.1f} MM COP")
print(f"  CV:        {train.std()/train.mean():.4f}")
print(f"  Asimetría: {stats.skew(train.values):.4f}")

print(f"\n  Variable exógena: IPC_Idx (Índice de precios al consumidor)")
print(f"  Rango IPC_Idx:    {exog_full['IPC_Idx'].min():.1f} → {exog_full['IPC_Idx'].max():.1f}")
print(f"\n  ✅ Datos cargados y división configurada")
""")


# ════════════════════════════════════════════════════════════════
# CELDA 3 — MARKDOWN: Fase II
# ════════════════════════════════════════════════════════════════
md(r"""---

## Fase II — Identificación de Orden Óptimo $(p,d,q)(P,D,Q)_{12}$

### Metodología Box-Jenkins (1970)

El proceso de identificación de orden sigue tres pasos clásicos:

1. **ACF (Autocorrelación)**: Muestra la correlación de la serie consigo misma
   a diferentes rezagos. Un corte abrupto en lag $q$ sugiere un componente MA($q$).
   Picos en lags 12, 24 confirman estacionalidad anual.
2. **PACF (Autocorrelación Parcial)**: Muestra la correlación *neta* de cada
   rezago, descontando los intermedios. Un corte abrupto en lag $p$ sugiere AR($p$).
3. **`auto_arima`**: Búsqueda automática que prueba combinaciones de $(p,d,q)$
   y $(P,D,Q)_{12}$, seleccionando por AIC mínimo.

### Reglas de Identificación

| Patrón en ACF | Patrón en PACF | Modelo sugerido |
|---------------|----------------|----------------|
| Decaimiento exponencial | Corte en lag $p$ | AR($p$) |
| Corte en lag $q$ | Decaimiento exponencial | MA($q$) |
| Decaimiento | Decaimiento | ARMA($p$,$q$) |
| Pico en lag 12 | Pico en lag 12 | SAR(1) o SMA(1) |

### Criterios de Selección

- **AIC** = $-2\ln L + 2k$: Criterio primario, balancea ajuste vs complejidad
- **BIC** = $-2\ln L + k\ln n$: Más conservador, penaliza más los parámetros
- **Principio de parsimonia**: Entre modelos con $\Delta AIC < 2$, elegir el más simple""")


# ════════════════════════════════════════════════════════════════
# CELDA 4 — CODE: Fase II — Identificación de orden + Fase III Ajuste
# ════════════════════════════════════════════════════════════════
code(r"""# ══════════════════════════════════════════════════════════════
# FASE II — ACF/PACF + auto_arima
# ══════════════════════════════════════════════════════════════

# ── Gráficas ACF / PACF ──
max_lags = min(24, len(train_log) // 2 - 1)
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_STANDARD if '_VIZ_THEME_LOADED' not in dir() or not _VIZ_THEME_LOADED else FIGSIZE_WIDE)

plot_acf(train_log, lags=max_lags, ax=axes[0], alpha=0.05)
axes[0].set_title('ACF — Serie log1p', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Lag (meses)')

plot_pacf(train_log, lags=max_lags, ax=axes[1], alpha=0.05, method='ywm')
axes[1].set_title('PACF — Serie log1p', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Lag (meses)')

plt.suptitle('Identificación de Orden — ACF y PACF',
             fontsize=14, fontweight='bold', y=1.02, fontfamily='serif')
plt.tight_layout()
if _VIZ_THEME_LOADED:
    marca_agua(fig)
    guardar_figura(fig, '04_acf_pacf', OUTPUTS_FIGURES)
plt.show()

# ── auto_arima para orden óptimo ──
print(f"{'═'*70}")
print(f"BÚSQUEDA AUTOMÁTICA DE ORDEN — auto_arima (AIC)")
print(f"{'═'*70}")

auto_model = pm.auto_arima(
    train_log,
    exogenous=exog_train.values,
    seasonal=True,
    m=ESTACIONALIDAD,
    d=None,         # auto_arima determina d
    D=None,         # auto_arima determina D
    max_p=3, max_q=3,
    max_P=2, max_Q=2,
    max_d=2, max_D=1,
    stepwise=True,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    information_criterion='aic',
    n_fits=50,
)

print(f"\n{'─'*70}")
print(f"MODELO ÓPTIMO SELECCIONADO")
print(f"{'─'*70}")
order = auto_model.order
seasonal_order = auto_model.seasonal_order
print(f"  Orden:       ARIMA{order}")
print(f"  Estacional:  {seasonal_order}")
print(f"  AIC:         {auto_model.aic():.2f}")
print(f"  BIC:         {auto_model.bic():.2f}")
print(f"\n  Modelo completo: SARIMAX{order}x{seasonal_order}")

# ══════════════════════════════════════════════════════════════
# FASE III — Ajuste SARIMAX con statsmodels
# ══════════════════════════════════════════════════════════════

print(f"\n{'═'*70}")
print(f"AJUSTE SARIMAX — statsmodels (MLE)")
print(f"{'═'*70}")

modelo_sarimax = SARIMAX(
    train_log,
    exog=exog_train,
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False,
)

resultado = modelo_sarimax.fit(disp=False, maxiter=500)
print(resultado.summary())

# ── Métricas de ajuste ──
print(f"\n{'─'*70}")
print(f"MÉTRICAS DE AJUSTE IN-SAMPLE")
print(f"{'─'*70}")
print(f"  AIC:         {resultado.aic:.2f}")
print(f"  BIC:         {resultado.bic:.2f}")
print(f"  HQIC:        {resultado.hqic:.2f}")
print(f"  Log-Lik:     {resultado.llf:.2f}")
print(f"  Parámetros:  {resultado.df_model}")
""")


# ════════════════════════════════════════════════════════════════
# CELDA 5 — MARKDOWN: Fase IV
# ════════════════════════════════════════════════════════════════
md(r"""---

## Fase IV — Validación Out-of-Sample: Oct–Dic 2025

### Qué significa validación OOS

La validación Out-of-Sample (*fuera de la muestra*) es la prueba ácida de
cualquier modelo predictivo. El modelo **nunca vio** los datos de Oct–Dic 2025
durante el entrenamiento. Comparar sus predicciones contra los valores reales
es equivalente a "viajar al futuro": el modelo hizo su apuesta y ahora
verificamos si acertó.

### Protocolo de Backtesting

El modelo se entrena **solo** con datos hasta Sep 2025 y genera pronósticos
para los 3 meses siguientes (Oct–Dic 2025). Estos se comparan con los valores
**reales observados** del recaudo, garantizando una evaluación sin *data leakage*.

### Métricas de Evaluación

| Métrica | Fórmula | Unidad | Utilidad ADRES |
|---------|---------|---------|--------------|
| **MAPE** | $\frac{1}{n}\sum\frac{|y_i - \hat{y}_i|}{y_i} \times 100$ | % | Comunicar precisión a gerencia |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | COP | Penalizar errores grandes (picos Ene/Jul) |
| **MAE** | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | COP | Desvío promedio en pesos colombianos |

**Por qué tres métricas y no una?** MAPE comunica bien pero falla con valores
cercanos a cero. RMSE penaliza errores grandes pero sufre con outliers.
MAE es robusto pero no distingue errores pequeños de grandes. Juntas dan
una visión completa.""")


# ════════════════════════════════════════════════════════════════
# CELDA 6 — CODE: Fase IV — Validación OOS
# ════════════════════════════════════════════════════════════════
code(r"""# ══════════════════════════════════════════════════════════════
# FASE IV — Validación OOS Oct–Dic 2025
# ══════════════════════════════════════════════════════════════

# ── Pronóstico OOS (3 meses) ──
forecast_log = resultado.get_forecast(
    steps=HORIZONTE_TEST,
    exog=exog_test,
)

pred_log = forecast_log.predicted_mean
ci_log   = forecast_log.conf_int(alpha=0.05)

# ── Invertir transformación log1p ──
pred_oos  = np.expm1(pred_log)
ci_inf    = np.expm1(ci_log.iloc[:, 0])
ci_sup    = np.expm1(ci_log.iloc[:, 1])
real_oos  = test.values

# ── Métricas OOS ──
mape = mean_absolute_percentage_error(real_oos, pred_oos) * 100
rmse = np.sqrt(mean_squared_error(real_oos, pred_oos))
mae  = mean_absolute_error(real_oos, pred_oos)

print(f"{'═'*70}")
print(f"VALIDACIÓN OOS — SARIMAX Oct–Dic 2025")
print(f"{'═'*70}")
print(f"  MAPE:  {mape:.2f}%")
print(f"  RMSE:  ${rmse/1e9:.1f} MM COP")
print(f"  MAE:   ${mae/1e9:.1f} MM COP")

# Error acumulado trimestral
error_trim = abs(real_oos.sum() - pred_oos.values.sum()) / real_oos.sum() * 100
print(f"\n  Error acumulado trimestral: {error_trim:.2f}%")
print(f"  Real acumulado:             ${real_oos.sum()/1e9:.1f} MM")
print(f"  Predicho acumulado:         ${pred_oos.values.sum()/1e9:.1f} MM")

# ── Tabla mes a mes ──
print(f"\n{'─'*70}")
print(f"{'Mes':<12} {'Real':>14} {'Predicho':>14} {'Error%':>10} {'IC 95%':>30}")
print(f"{'─'*70}")
for i, fecha in enumerate(test.index):
    err_pct = (pred_oos.iloc[i] - real_oos[i]) / real_oos[i] * 100
    print(f"  {fecha.strftime('%Y-%m'):<10} ${real_oos[i]/1e9:>11.1f}MM "
          f"${pred_oos.iloc[i]/1e9:>11.1f}MM {err_pct:>+8.1f}% "
          f"[${ci_inf.iloc[i]/1e9:.1f} – ${ci_sup.iloc[i]/1e9:.1f}]")

# ── Gráfica Real vs Predicho OOS ──
fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD if not _VIZ_THEME_LOADED else FIGSIZE_WIDE)

# Contexto: últimos 12 meses de entrenamiento
ctx = serie_full.loc['2025-01-01':TRAIN_END]
ax.plot(ctx.index, ctx.values/1e9, color=C_PRIMARY, alpha=0.4,
        linewidth=1.5, label='Entrenamiento (contexto)')

# Real OOS
ax.plot(test.index, real_oos/1e9, 'o-', color=C_PRIMARY,
        linewidth=2.5, markersize=8, label='Real OOS')

# Predicho SARIMAX
ax.plot(test.index, pred_oos.values/1e9, 's--', color=COLORES_MODELOS['sarimax'],
        linewidth=2, markersize=8, label=f'SARIMAX (MAPE={mape:.1f}%)')

# Intervalo de confianza
ax.fill_between(test.index, ci_inf.values/1e9, ci_sup.values/1e9,
                alpha=0.2, color=COLORES_MODELOS['sarimax'], label='IC 95%')

ax.axvline(pd.Timestamp(TEST_START), color='grey', linestyle=':', alpha=0.6, label='Inicio OOS')
ax.legend(loc='best', fontsize=9)
ax.set_ylabel('Recaudo (MM COP)', fontsize=11)
ax.grid(True, alpha=0.3)

if _VIZ_THEME_LOADED:
    titulo_profesional(ax, 'SARIMAX — Validación Out-of-Sample',
                       f'Oct–Dic 2025 | MAPE={mape:.2f}%')
    formato_pesos_eje(ax, eje='y')
    marca_agua(fig)
    guardar_figura(fig, '04_sarimax_real_vs_pred', OUTPUTS_FIGURES)
else:
    ax.set_title(f'SARIMAX — Real vs Predicho OOS (MAPE={mape:.2f}%)',
                 fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()

# ── Exportar pronóstico OOS ──
df_forecast_oos = pd.DataFrame({
    'Fecha': test.index,
    'Real': real_oos,
    'Pronostico_SARIMAX': pred_oos.values,
    'IC_Inferior': ci_inf.values,
    'IC_Superior': ci_sup.values,
    'Error_Abs': np.abs(pred_oos.values - real_oos),
    'Error_Pct': (pred_oos.values - real_oos) / real_oos * 100,
})
df_forecast_oos.to_csv(OUTPUTS_FORECASTS / 'sarimax_forecast.csv', index=False)
print(f"\n  ✅ Pronóstico OOS exportado: sarimax_forecast.csv")
""")


# ════════════════════════════════════════════════════════════════
# CELDA 7 — MARKDOWN: Fase V
# ════════════════════════════════════════════════════════════════
md(r"""---

## Fase V — Diagnóstico de Residuos

### Por qué importan los residuos

Los residuos = Real $-$ Predicho son la *información que el modelo no
capturó*. Si los residuos muestran patrones (autocorrelación, tendencia,
heteroscedasticidad), significa que hay estructura predecible que el modelo
está desperdiciando.

### Tests Estadísticos

| Test | Qué evalúa | Hipótesis Nula | Criterio |
|------|-----------|---------------|----------|
| **Ljung-Box** | Autocorrelación | Residuos son ruido blanco | $p > 0.05$: OK |
| **Shapiro-Wilk** | Normalidad | Residuos son normales | $p > 0.05$: OK |
| **Durbin-Watson** | Autocorrelación lag-1 | No hay autocorrelación | DW $\approx 2.0$: OK |
| **ARCH LM** | Heterocedasticidad | Varianza constante | $p > 0.05$: OK |

### Interpretación del QQ-Plot

El gráfico QQ compara los cuantiles de los residuos contra los cuantiles
teóricos de una distribución normal. Si los puntos siguen la línea diagonal,
los residuos son normales. Desviaciones en las colas indican valores extremos
más frecuentes de lo esperado (*colas pesadas*).""")


# ════════════════════════════════════════════════════════════════
# CELDA 8 — CODE: Fase V — Diagnóstico de residuos
# ════════════════════════════════════════════════════════════════
code(r"""# ══════════════════════════════════════════════════════════════
# FASE V — Diagnóstico de Residuos
# ══════════════════════════════════════════════════════════════

residuos = resultado.resid

# ── Diagnóstico visual (4 paneles) ──
fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_QUAD if _VIZ_THEME_LOADED else (14, 10))

# Panel 1: Serie de residuos
ax = axes[0, 0]
ax.plot(residuos.index, residuos.values, color=C_PRIMARY, linewidth=1)
ax.axhline(0, color='red', linewidth=0.8, linestyle='--')
ax.fill_between(residuos.index,
                -2*residuos.std(), 2*residuos.std(),
                alpha=0.1, color='red')
ax.set_title('Residuos del Modelo', fontsize=11, fontweight='bold')
ax.set_ylabel('Residuo (log1p)')
ax.grid(True, alpha=0.3)

# Panel 2: Histograma + QQ
ax2 = axes[0, 1]
ax2.hist(residuos, bins=15, density=True, alpha=0.7, color=C_PRIMARY, edgecolor='white')
x_vals = np.linspace(residuos.min(), residuos.max(), 100)
ax2.plot(x_vals, stats.norm.pdf(x_vals, residuos.mean(), residuos.std()),
         color=C_SECONDARY, linewidth=2, label='Normal teórica')
ax2.set_title('Distribución de Residuos', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Panel 3: ACF de residuos
max_lags_resid = min(24, len(residuos) // 2 - 1)
plot_acf(residuos, lags=max_lags_resid, ax=axes[1, 0], alpha=0.05)
axes[1, 0].set_title('ACF Residuos', fontsize=11, fontweight='bold')

# Panel 4: QQ-Plot
stats.probplot(residuos, dist='norm', plot=axes[1, 1])
axes[1, 1].set_title('QQ-Plot Residuos', fontsize=11, fontweight='bold')
axes[1, 1].get_lines()[1].set_color(C_SECONDARY)

plt.suptitle('Diagnóstico de Residuos — SARIMAX',
             fontsize=14, fontweight='bold', y=1.01, fontfamily='serif')
plt.tight_layout()
if _VIZ_THEME_LOADED:
    marca_agua(fig)
    guardar_figura(fig, '04_sarimax_diagnostico_residuos', OUTPUTS_FIGURES)
plt.show()

# ── Tests formales ──
print(f"{'═'*70}")
print(f"DIAGNÓSTICO ESTADÍSTICO DE RESIDUOS")
print(f"{'═'*70}")

# Ljung-Box
lb_lags = [l for l in [6, 12, 24] if l < len(residuos) // 2]
lb = acorr_ljungbox(residuos, lags=lb_lags, return_df=True)
print(f"\n  Ljung-Box (ruido blanco):")
for lag in lb_lags:
    if lag in lb.index:
        pval = lb.loc[lag, 'lb_pvalue']
        status = '✅ Aceptado' if pval > 0.05 else '⚠️ Rechazado'
        print(f"    Lag {lag:>2}: p={pval:.4f} {status}")

# Shapiro-Wilk
stat_sw, p_sw = stats.shapiro(residuos)
sw_status = '✅ Normal' if p_sw > 0.05 else '⚠️ No normal'
print(f"\n  Shapiro-Wilk: W={stat_sw:.4f}, p={p_sw:.4f} → {sw_status}")

# Durbin-Watson
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(residuos)
dw_status = '✅ OK' if 1.5 < dw < 2.5 else '⚠️ Posible autocorrelación'
print(f"  Durbin-Watson: {dw:.4f} → {dw_status}")

# Heterocedasticidad (Breusch-Pagan simplificado)
from statsmodels.stats.diagnostic import het_arch
if len(residuos) > 12:
    stat_arch, p_arch, _, _ = het_arch(residuos, nlags=6)
    arch_status = '✅ Homocedástico' if p_arch > 0.05 else '⚠️ Heterocedástico'
    print(f"  ARCH LM(6):    F={stat_arch:.4f}, p={p_arch:.4f} → {arch_status}")
""")


# ════════════════════════════════════════════════════════════════
# CELDA 9 — MARKDOWN: Fase VI
# ════════════════════════════════════════════════════════════════
md(r"""---

## Fase VI — Pronóstico de Producción 2026

### Estrategia de Reentreno

Para el pronóstico operativo, se **reentrena** el modelo con la serie
completa (incluyendo Oct–Dic 2025) y se proyecta 12 meses (Ene–Dic 2026).
Esto maximiza la información disponible para el modelo.

### Proyección de la Exógena IPC

Como el IPC de 2026 aún no se conoce, se utiliza la proyección del
Banco de la República (5.10% anual), interpolada mensualmente:

$$IPC\_Idx_{m+1} = IPC\_Idx_m \times (1 + \frac{0.051}{12})$$

Esta es una limitación inherente a los modelos con exógenas: la precisión
del pronóstico depende de la precisión de la proyección de las variables
externas.

### Intervalos de Confianza

Se reportan IC al 95%. Su ancho crece con el horizonte porque la
incertidumbre se acumula: predecir enero 2026 es más fácil que predecir
diciembre 2026. Esto es una propiedad natural de los modelos SARIMA,
no un defecto.""")


# ════════════════════════════════════════════════════════════════
# CELDA 10 — CODE: Fase VI — Pronóstico 2026
# ════════════════════════════════════════════════════════════════
code(r"""# ══════════════════════════════════════════════════════════════
# FASE VI — Pronóstico de Producción 2026
# ══════════════════════════════════════════════════════════════

# ── Reentreno con serie completa (Oct 2021 – Dic 2025) ──
full_log = np.log1p(serie_full)

modelo_full = SARIMAX(
    full_log,
    exog=exog_full,
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False,
)

resultado_full = modelo_full.fit(disp=False, maxiter=500)

# ── Construir exógenas 2026 ──
fechas_2026 = pd.date_range('2026-01-01', periods=HORIZONTE_PRONOSTICO, freq='MS')

# IPC_Idx 2026: continuar interpolación mensual con IPC anual 5.10%
ultimo_ipc_idx = exog_full['IPC_Idx'].iloc[-1]
ipc_mensual_2026 = ultimo_ipc_idx * (1 + MACRO_DATA[2026]['IPC']/100/12) ** np.arange(1, 13)
exog_2026 = pd.DataFrame({'IPC_Idx': ipc_mensual_2026}, index=fechas_2026)

# ── Pronóstico 12 meses ──
forecast_2026 = resultado_full.get_forecast(
    steps=HORIZONTE_PRONOSTICO,
    exog=exog_2026,
)

pred_2026_log = forecast_2026.predicted_mean
ci_2026_log   = forecast_2026.conf_int(alpha=0.05)

# Invertir transformación (proteger contra CI desbordados en log-space)
pred_2026 = np.expm1(pred_2026_log)
ci_2026_inf = np.expm1(ci_2026_log.iloc[:, 0]).clip(lower=0)
ci_2026_sup = np.expm1(ci_2026_log.iloc[:, 1])
ci_2026_sup = ci_2026_sup.where(~np.isinf(ci_2026_sup), pred_2026 * 2.5)

# ── Gráfica de producción 2026 ──
fig, ax = plt.subplots(figsize=FIGSIZE_WIDE if _VIZ_THEME_LOADED else (16, 7))

# Histórico completo
ax.plot(serie_full.index, serie_full.values/1e9, color=C_PRIMARY,
        linewidth=2, label='Real (Oct 2021 – Dic 2025)')

# Pronóstico 2026
ax.plot(fechas_2026, pred_2026.values/1e9, 'D-', color=COLORES_MODELOS['sarimax'],
        linewidth=2.5, markersize=7, label='SARIMAX Pronóstico 2026')

# IC 95%
ax.fill_between(fechas_2026, ci_2026_inf.values/1e9, ci_2026_sup.values/1e9,
                alpha=0.2, color=COLORES_MODELOS['sarimax'], label='IC 95%')

ax.axvline(pd.Timestamp('2026-01-01'), color='grey', linestyle=':', alpha=0.6)
ax.legend(loc='best', fontsize=9)
ax.set_ylabel('Recaudo (MM COP)', fontsize=11)
ax.grid(True, alpha=0.3)

if _VIZ_THEME_LOADED:
    titulo_profesional(ax, 'SARIMAX — Pronóstico de Producción 2026',
                       f'Orden {order}×{seasonal_order} | Exógena: IPC_Idx')
    formato_pesos_eje(ax, eje='y')
    marca_agua(fig)
    guardar_figura(fig, '04_sarimax_produccion_2026', OUTPUTS_FIGURES)
else:
    ax.set_title(f'SARIMAX — Pronóstico de Producción 2026\n'
                 f'Orden {order}×{seasonal_order} | Exógena: IPC_Idx',
                 fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()

# ── Tabla pronóstico mensual ──
print(f"{'═'*70}")
print(f"PRONÓSTICO MENSUAL 2026 — SARIMAX")
print(f"{'═'*70}")
total_anual = 0
for i, fecha in enumerate(fechas_2026):
    val = pred_2026.iloc[i]
    total_anual += val
    print(f"  {fecha.strftime('%Y-%m')}   ${val/1e9:>10.1f} MM   "
          f"[${ci_2026_inf.iloc[i]/1e9:.1f} – ${ci_2026_sup.iloc[i]/1e9:.1f}]")

print(f"{'─'*70}")
print(f"  TOTAL 2026:    ${total_anual/1e9:>10.1f} MM")
print(f"  Promedio mes:  ${total_anual/12/1e9:>10.1f} MM")

# ── Exportar pronóstico 2026 ──
df_forecast_2026 = pd.DataFrame({
    'Fecha': fechas_2026,
    'Pronostico': pred_2026.values,
    'Limite_Inferior': ci_2026_inf.values,
    'Limite_Superior': ci_2026_sup.values,
})
df_forecast_2026.to_csv(OUTPUTS_FORECASTS / 'sarimax_forecast_2026.csv', index=False)
print(f"\n  ✅ Pronóstico 2026 exportado: sarimax_forecast_2026.csv")
""")


# ════════════════════════════════════════════════════════════════
# CELDA 11 — MARKDOWN: Resumen Ejecutivo
# ════════════════════════════════════════════════════════════════
md(r"""---

## Resumen Ejecutivo — SARIMAX

### Fortalezas

| Aspecto | Descripción | Relevancia para ADRES |
|---------|-------------|----------------------|
| **Interpretabilidad** | Coeficientes directamente interpretables | La gerencia entiende qué impulsa el pronóstico |
| **Variables exógenas** | IPC captura efecto inflacionario | Impuestos *ad valorem* modelados explícitamente |
| **Marco teórico** | Box-Jenkins (1970) | Respaldo académico y regulatorio |
| **Intervalos analíticos** | IC derivados de la distribución del modelo | Cuantificación formal de incertidumbre |

### Limitaciones

| Aspecto | Descripción | Cómo se mitiga |
|---------|-------------|----------------|
| **Linealidad** | No captura relaciones no lineales | Prophet y XGBoost complementan |
| **Estacionariedad** | Requiere diferenciación | Transformación log1p reduce heterocedasticidad |
| **N limitada** | 45 meses puede ser insuficiente | Ensemble multi-modelo reduce riesgo |

### Siguiente Paso

$\rightarrow$ `05_Prophet.ipynb` — Modelo bayesiano aditivo con changepoints
adaptativos para capturar quiebres estructurales no lineales.
""")


# ════════════════════════════════════════════════════════════════
# CELDA 12 — CODE: Exportación final de métricas
# ════════════════════════════════════════════════════════════════
code(r"""# ══════════════════════════════════════════════════════════════
# EXPORTACIÓN FINAL DE MÉTRICAS
# ══════════════════════════════════════════════════════════════

metricas = {
    'Modelo': 'SARIMAX',
    'MAPE': round(mape, 2),
    'RMSE_MM': round(rmse / 1e9, 1),
    'MAE_MM': round(mae / 1e9, 1),
    'Error_Trimestral_%': round(error_trim, 2),
    'Orden': f'{order}x{seasonal_order}',
    'AIC': round(resultado.aic, 2),
    'BIC': round(resultado.bic, 2),
    'Exogenas': 'IPC_Idx',
    'N_Train': len(train),
    'N_Test': len(test),
}

# Exportar métricas
df_metricas = pd.DataFrame([metricas])
metricas_path = OUTPUTS_REPORTS / 'sarimax_metricas.csv'
df_metricas.to_csv(metricas_path, index=False)

print(f"{'═'*70}")
print(f"RESUMEN EJECUTIVO — MODELO SARIMAX")
print(f"{'═'*70}")
for k, v in metricas.items():
    print(f"  {k:<22}: {v}")
print(f"\n  ✅ Métricas exportadas:  {metricas_path.name}")
print(f"  ✅ Pronóstico OOS:      sarimax_forecast.csv")
print(f"  ✅ Pronóstico 2026:     sarimax_forecast_2026.csv")
print(f"\n  → Siguiente: 05_Prophet.ipynb")
""")


# ════════════════════════════════════════════════════════════════
# GUARDAR NOTEBOOK
# ════════════════════════════════════════════════════════════════
out_path = Path(__file__).resolve().parent.parent / 'notebooks' / '04_SARIMAX.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f'✅ 04_SARIMAX.ipynb generado ({len(nb.cells)} celdas) → {out_path}')

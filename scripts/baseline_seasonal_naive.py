"""
Baseline Seasonal Naive — Análisis exploratorio de referencia
=============================================================
Propósito: modelo de referencia mínimo (seasonal naive) para contextualizar
el desempeño de los modelos ML. NO forma parte del pipeline de modelado principal.
Se incluye como análisis exploratorio complementario conforme a la observación
C9 del sinodal (Bello Melo, 2026).

Seasonal naive: ŷ(t) = y(t-12)
Predice el recaudo de cada mes como el valor observado en el mismo mes del año anterior.
Representa el mejor resultado alcanzable con extrapolación histórica simple,
sin variables exógenas ni componentes estacionales ajustados.

Referencia: Hyndman & Athanasopoulos (2018). Forecasting: Principles and Practice
(3rd ed.). OTexts. https://otexts.com/fpp3/

Autor: Pipeline CRISP-DM — Rentas Cedidas
Fecha: Junio 2026
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Rutas del proyecto ────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS_FIGS   = PROJECT_ROOT / "outputs" / "figures"
OUTPUTS_REP    = PROJECT_ROOT / "outputs" / "reports"
OUTPUTS_FC     = PROJECT_ROOT / "outputs" / "forecasts"

for d in [OUTPUTS_FIGS, OUTPUTS_REP, OUTPUTS_FC]:
    d.mkdir(parents=True, exist_ok=True)

# ── Configuración del pipeline (consistente con 00_config.py) ─────────────────
FECHA_INICIO      = "2021-10-01"
FECHA_FIN         = "2025-12-31"
TEST_START        = "2025-10-01"
TEST_END          = "2025-12-31"

# ── 1. Carga de la serie ──────────────────────────────────────────────────────
print("=" * 65)
print("BASELINE SEASONAL NAIVE — ANÁLISIS EXPLORATORIO")
print("=" * 65)

serie = (
    pd.read_csv(DATA_PROCESSED / "serie_mensual.csv",
                parse_dates=["Fecha"], index_col="Fecha")
    ["Recaudo_Total"]
    .loc[FECHA_INICIO:FECHA_FIN]
    .rename("Recaudo_Total")
)
serie.index.freq = "MS"

# ── 2. Pronóstico naive (shift 12 meses) ──────────────────────────────────────
y_naive = serie.shift(12)          # y_hat(t) = y(t-12), definición sin fuga
y_test  = serie.loc[TEST_START:TEST_END]
y_naive_oos = y_naive.loc[TEST_START:TEST_END]

# ── 3. Métricas OOS ──────────────────────────────────────────────────────────
errores = y_test - y_naive_oos
mape    = float((errores.abs() / y_test).mean() * 100)
rmse    = float(np.sqrt((errores ** 2).mean()))
mae     = float(errores.abs().mean())
sesgo   = float(-errores.mean())                      # positivo = sobreestimación
err_trim = float(
    abs(y_test.sum() - y_naive_oos.sum()) / y_test.sum() * 100
)

print(f"\nPeríodo OOS: {TEST_START} → {TEST_END}  (n = {len(y_test)})")
print(f"\n{'Métrica':<30} {'Valor':>12}")
print("-" * 44)
print(f"{'MAPE mensual (%)':<30} {mape:>11.2f}%")
print(f"{'RMSE (MM COP)':<30} {rmse/1e9:>11.1f}")
print(f"{'MAE (MM COP)':<30} {mae/1e9:>11.1f}")
print(f"{'Error trimestral (%)':<30} {err_trim:>11.2f}%")
print(f"{'Sesgo (MM COP)':<30} {sesgo/1e9:>+11.1f}")

print(f"\n{'Mes':<12} {'Real (MM)':>10} {'Naive (MM)':>11} {'Error %':>9}")
print("-" * 44)
meses_es = {1:"Ene", 2:"Feb", 3:"Mar", 4:"Abr", 5:"May", 6:"Jun",
            7:"Jul", 8:"Ago", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dic"}
for fecha in y_test.index:
    real  = y_test[fecha] / 1e9
    pred  = y_naive_oos[fecha] / 1e9
    ep    = (pred - real) / real * 100
    label = f"{meses_es[fecha.month]} {fecha.year}"
    print(f"  {label:<10} {real:>10.1f} {pred:>11.1f} {ep:>+9.2f}%")

# ── 4. Backtesting con 4 cortes retrospectivos ────────────────────────────────
print("\nBacktesting (4 cortes OOS de 3 meses):")
cortes = ["2024-12-31", "2025-03-31", "2025-06-30", "2025-09-30"]
mapes_bt = []
rows_bt  = []

for c in cortes:
    ini = pd.Timestamp(c) + pd.offsets.MonthBegin(1)
    fin = ini + pd.offsets.MonthEnd(3)
    yt  = serie.loc[ini:fin]
    yn  = serie.shift(12).loc[ini:fin]
    if len(yt) == 3 and not yn.isna().any():
        m = float((abs(yt - yn) / yt).mean() * 100)
        mapes_bt.append(m)
        rows_bt.append({"OOS_inicio": ini.strftime("%b %Y"),
                        "OOS_fin":    fin.strftime("%b %Y"),
                        "MAPE_pct":   round(m, 2)})
        print(f"  {ini:%b %Y}–{fin:%b %Y}: MAPE = {m:.2f}%")

if mapes_bt:
    print(f"  Promedio: {np.mean(mapes_bt):.2f}% ± {np.std(mapes_bt):.2f}%")

# ── 5. Comparativa con los modelos ML (referencia del Día 1) ─────────────────
print("\n" + "=" * 65)
print("COMPARATIVA DE REFERENCIA (mismo período OOS oct–dic 2025)")
print("=" * 65)
comparativa = [
    ("Seasonal naive",      mape,   "Máxima", "Referencia — sin exógenas"),
    ("SARIMAX",             9.75,   "Alta",   "IPC_Idx como exógena"),
    ("Prophet",             6.30,   "Alta",   "UPC + festivos"),
    ("XGBoost (corregido)", 4.55,   "Media",  "Lags + MA + Optuna 200 trials"),
    ("LSTM",               23.52,   "Baja",   "n_train=38 vs 31.905 params"),
]
print(f"\n{'Modelo':<25} {'MAPE':>7} {'Parsim.':>9}  Notas")
print("-" * 75)
for nombre, mp, par, nota in comparativa:
    marca = " ✓" if nombre == "XGBoost (corregido)" else \
            " ←" if nombre == "Seasonal naive" else ""
    print(f"  {nombre:<23} {mp:>6.2f}%  {par:>8}  {nota}{marca}")
mejora = mape - 4.55
print(f"\n  Mejora XGBoost vs. naive: {mejora:.2f} pp")
print(f"  Reducción relativa del error: {mejora/mape*100:.1f}%")

# ── 6. Exportar resultados ────────────────────────────────────────────────────
# CSV de métricas
metricas = pd.DataFrame([{
    "Modelo":           "Seasonal naive",
    "MAPE_mensual_pct": round(mape, 2),
    "RMSE_MM_COP":      round(rmse / 1e9, 1),
    "MAE_MM_COP":       round(mae / 1e9, 1),
    "Error_trim_pct":   round(err_trim, 2),
    "Sesgo_MM_COP":     round(sesgo / 1e9, 1),
    "Parsimonia":       "Maxima",
    "Descripcion":      "y_hat(t)=y(t-12), sin variables exogenas",
}])
metricas.to_csv(OUTPUTS_REP / "baseline_naive_metricas_oos.csv", index=False)

# CSV del backtesting
pd.DataFrame(rows_bt).to_csv(
    OUTPUTS_REP / "baseline_naive_backtesting.csv", index=False
)

# CSV del forecast naive OOS (para agregar a la comparativa del notebook 08)
oos_df = pd.DataFrame({
    "Fecha":           y_test.index,
    "Real_MM":         (y_test.values / 1e9).round(1),
    "Pronostico_Naive_MM": (y_naive_oos.values / 1e9).round(1),
    "Error_MM":        ((y_naive_oos.values - y_test.values) / 1e9).round(1),
    "Error_Pct":       ((y_naive_oos.values - y_test.values) / y_test.values * 100).round(2),
})
oos_df.to_csv(OUTPUTS_FC / "naive_forecast_oos.csv", index=False)

# ── 7. Figura para la presentación ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Baseline Seasonal Naive — Análisis Exploratorio de Referencia\n"
    "(oct. 2021 – dic. 2025)",
    fontsize=12, fontweight="bold"
)

# 7a. Serie completa: real vs naive
ax = axes[0]
serie_mm = serie / 1e9
naive_mm = y_naive / 1e9
ax.plot(serie_mm.index, serie_mm.values,
        color="#1C4E80", lw=1.8, label="Recaudo real")
ax.plot(naive_mm.dropna().index, naive_mm.dropna().values,
        color="#888780", lw=1.2, ls="--", alpha=0.8, label="Seasonal naive")
# Destacar período OOS
oos_mask = (serie_mm.index >= TEST_START)
ax.fill_between(serie_mm.index, 0, serie_mm.values,
                where=oos_mask, alpha=0.08, color="#3B6D11", label="OOS (oct–dic 2025)")
ax.set_xlabel("Fecha", fontsize=10)
ax.set_ylabel("MM COP (miles de millones)", fontsize=10)
ax.set_title("Serie histórica: real vs. naive", fontsize=11)
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
ax.grid(axis="y", alpha=0.3)

# 7b. MAPE comparativo en barras
ax2 = axes[1]
modelos_plot  = ["Seasonal\nnaive", "SARIMAX", "Prophet", "XGBoost\n(corregido)", "LSTM"]
mapes_plot    = [mape, 9.75, 6.30, 4.55, 23.52]
colores_plot  = ["#A32D2D", "#E8A838", "#1C4E80", "#1B7A4A", "#888780"]
bars = ax2.bar(modelos_plot, mapes_plot, color=colores_plot,
               edgecolor="white", linewidth=0.8)
ax2.axhline(12, color="#B5280F", lw=1.2, ls="--", label="Umbral 12%")
for bar, val in zip(bars, mapes_plot):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.3,
             f"{val:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.set_ylabel("MAPE OOS (%)", fontsize=10)
ax2.set_title("MAPE comparativo — período OOS (oct–dic 2025)", fontsize=11)
ax2.legend(fontsize=9)
ax2.set_ylim(0, max(mapes_plot) * 1.18)
ax2.grid(axis="y", alpha=0.3)
ax2.set_axisbelow(True)

plt.tight_layout()
out_fig = OUTPUTS_FIGS / "00_baseline_naive_comparativa.png"
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close()

print(f"\nArchivos exportados:")
print(f"  {OUTPUTS_REP / 'baseline_naive_metricas_oos.csv'}")
print(f"  {OUTPUTS_REP / 'baseline_naive_backtesting.csv'}")
print(f"  {OUTPUTS_FC  / 'naive_forecast_oos.csv'}")
print(f"  {out_fig}")
print("\n✅ LISTO — Seasonal naive calculado.")
print(f"\nResultado clave para la sustentación:")
print(f"  MAPE naive  = {mape:.2f}%  (extrapolación simple mes-a-mes)")
print(f"  MAPE XGBoost = 4.55%  (modelo seleccionado)")
print(f"  Mejora      = {mape - 4.55:.2f} pp  ({(mape-4.55)/mape*100:.1f}% de reducción)")
print(f"\n  Este valor reemplaza el '25% sin fuente' del documento.")

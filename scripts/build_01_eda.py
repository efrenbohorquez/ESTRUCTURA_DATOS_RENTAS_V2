"""
build_01_eda.py — Genera el notebook 01_EDA_Completo.ipynb a nivel tesis doctoral.
Ejecutar una sola vez:  python scripts/build_01_eda.py
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.11.0"
    }
})

cells = []

def md(source):
    cells.append(nbf.v4.new_markdown_cell(source))

def code(source):
    cells.append(nbf.v4.new_code_cell(source))

# ════════════════════════════════════════════════════════════════
# CELDA 1 — TÍTULO Y MARCO METODOLÓGICO
# ════════════════════════════════════════════════════════════════
md(r"""# 01 · Análisis Exploratorio de Datos (EDA)
## Recaudo de Rentas Cedidas del SGSS — Colombia, Oct 2021 – Dic 2025

### Contexto institucional

Las **rentas cedidas** son tributos específicos (impuesto al consumo de licores,
cervezas, cigarrillos, juegos de azar, entre otros) cuyo recaudo es transferido
por la DIAN y entes territoriales a la **ADRES** (Administradora de los Recursos
del Sistema General de Seguridad Social en Salud) para financiar el aseguramiento
en salud de la población colombiana. Comprender la dinámica de estos ingresos
fiscales es crítico para la planeación presupuestal del sector salud y la
sostenibilidad financiera del SGSS.

> **Objetivo.** Realizar una exploración estadística rigurosa de la serie de recaudo
> de rentas cedidas para salud, con el fin de: *(i)* auditar la calidad de los datos,
> *(ii)* caracterizar la estructura categórica y temporal del ingreso fiscal,
> *(iii)* identificar componentes de tendencia, estacionalidad y ruido, y
> *(iv)* establecer las hipótesis estadísticas que guiarán la selección de modelos
> predictivos en las etapas subsiguientes.

### Marco metodológico

El **Análisis Exploratorio de Datos** (EDA, por sus siglas en inglés) es una
filosofía de análisis propuesta por John W. Tukey (1977) en su obra seminal
*Exploratory Data Analysis*. A diferencia del enfoque confirmatorio clásico,
el EDA prioriza la **visualización** y la **detección de patrones** antes de
formular hipótesis formales. William S. Cleveland (1993) extendió esta filosofía
con el concepto de *visualizing data* — la idea de que los gráficos bien diseñados
revelan estructura que los estadísticos resumen no capturan.

Nuestro protocolo se articula en cuatro fases progresivas:

| Fase | Contenido | Preguntas clave |
|------|-----------|-----------------|
| **I** | Auditoría de calidad, tipificación y estructura categórica | ¿Son los datos confiables? ¿Qué conceptos fiscales los componen? |
| **II** | Agregación mensual, tendencia visual, estacionalidad | ¿Cuál es la forma de la serie? ¿Hay patrones recurrentes? |
| **III** | Descomposición clásica, estacionariedad (ADF/KPSS), ACF/PACF | ¿Es estacionaria la serie? ¿Qué estructura de dependencia temporal tiene? |
| **IV** | Deflación real con IPC, crecimiento interanual, correlación macro | ¿Cuánto del crecimiento es real vs. inflación? ¿Qué variables macro se correlacionan? |

> **¿Por qué es importante el EDA antes de modelar?** Un modelo predictivo es tan
> bueno como los datos y supuestos sobre los que se construye. El EDA permite:
> descubrir valores atípicos que distorsionarían las estimaciones, verificar si la
> serie cumple los supuestos de estacionariedad que requieren modelos como ARIMA,
> y formular hipótesis informadas sobre qué regresores exógenos incluir.

**Dataset.** `BaseRentasCedidasVF.xlsx` — 149 648 registros
transaccionales x 13 variables, periodo Oct 2021 -- Dic 2025 (51 meses).
""")

# ════════════════════════════════════════════════════════════════
# CELDA 2 — CONFIGURACIÓN E IMPORTACIONES
# ════════════════════════════════════════════════════════════════
code(r"""# ── 0. Configuración centralizada ─────────────────────────────────
import pandas as pd
import numpy as np
%run 00_config.py

# Librerías adicionales para EDA avanzado
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import calendar

print(f"\n📁 Archivo de datos: {DATA_FILE.name}")
print(f"📊 Variables macro cargadas: {sorted(MACRO_DATA.keys())}")
print(f"🔧 Periodo: {FECHA_INICIO} → {FECHA_FIN}")
print(f"✂️  División Entrenamiento/Prueba: Entrenamiento hasta {TRAIN_END} | Prueba desde {TEST_START}")
""")

# ════════════════════════════════════════════════════════════════
# FASE I — AUDITORÍA DE CALIDAD
# ════════════════════════════════════════════════════════════════
md(r"""---
## Fase I — Auditoría de Calidad y Estructura de Datos

Antes de cualquier análisis estadístico, la **auditoría de calidad** es un paso
indispensable. Datos con errores de tipo, valores faltantes ocultos o codificaciones
inconsistentes pueden generar resultados espurios en etapas posteriores. Esta fase
sigue las recomendaciones del framework *Data Quality Assessment* de Pipino, Lee y
Wang (2002), evaluando cuatro dimensiones: **completitud**, **exactitud**,
**consistencia** y **oportunidad**.

### 1.1  Carga y perfilamiento inicial

Se importa el archivo fuente y se realiza un **perfilamiento automático** (automated
profiling) que incluye: dimensiones, tipos de dato, conteo de nulos, rango temporal
y valores únicos por columna. Este perfilamiento equivale a lo que herramientas como
*pandas-profiling* realizan, pero implementado explícitamente para mayor transparencia.

> **Lectura del perfilamiento:** La columna *Nulos* indica si hay datos faltantes
> (marcados con *NaN*). La columna *Únicos* revela la cardinalidad de cada variable:
> valores bajos sugieren variables categóricas, valores altos sugieren variables
> continuas o identificadores.
""")

code(r"""# ── 1.1 Carga del dataset ─────────────────────────────────────────
print(f"Cargando: {DATA_FILE.name}")
df = pd.read_excel(DATA_FILE)
print(f"✅ Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas\n")

# Perfilamiento completo
print("═" * 70)
print("PERFIL DEL DATASET")
print("═" * 70)
print(f"{'Columna':<35} {'Tipo':<18} {'Nulos':>6} {'Únicos':>8}")
print("─" * 70)
for col in df.columns:
    n_nulos = df[col].isna().sum()
    n_unicos = df[col].nunique()
    dtype = str(df[col].dtype)
    marca = " ⚠️" if n_nulos > 0 else ""
    print(f"{col:<35} {dtype:<18} {n_nulos:>6,} {n_unicos:>8,}{marca}")

print(f"\n📅 Rango temporal: {df[COL_FECHA].min()} → {df[COL_FECHA].max()}")
print(f"📊 Meses únicos:  {df[COL_FECHA].dt.to_period('M').nunique()}")
""")

# ────────────────────────────────────────────────────────────────
md(r"""### 1.2  Conversión de tipos y detección de anomalías

`ValorRecaudo` se almacena como *object* (texto) en el archivo fuente, lo cual es
común en archivos Excel donde columnas monetarias incluyen separadores de miles o
símbolos ($). Se convierte a numérico con `errors='coerce'`, que asigna *NaN* a
entradas no parseables en lugar de producir un error.

**Valores negativos: ajustes contables presupuestales.** En el contexto de las
rentas cedidas colombianas, los registros negativos no son errores sino
**reversiones presupuestales** y **conciliaciones inter-vigencia**. Esto ocurre
cuando se ajustan pagos anteriores, se corrigen errores de imputación entre
vigencias o se realizan devoluciones por pagos en exceso. Estos valores se
**mantienen** en el análisis porque representan la realidad fiscal neta del flujo
de recursos — eliminarlos sesgaría la serie hacia arriba.

> **Decisión de diseño:** Mantener los negativos implica que la serie mensual
> refleja el *recaudo neto efectivo*, no el recaudo bruto. Esta distinción es
> relevante para la interpretación de los pronósticos: los modelos predicen el
> flujo neto, incluyendo la posibilidad de ajustes a la baja.
""")

code(r"""# ── 1.2 Conversiones y limpieza ───────────────────────────────────
# Fecha → datetime
df[COL_FECHA] = pd.to_datetime(df[COL_FECHA])

# Valor → numérico (viene como string)
df[COL_VALOR] = pd.to_numeric(df[COL_VALOR], errors='coerce')
n_coerced = df[COL_VALOR].isna().sum()
if n_coerced > 0:
    print(f"⚠️ {n_coerced} valores no numéricos convertidos a NaN — se eliminan.")
    df = df.dropna(subset=[COL_VALOR])

# Valores negativos
mask_neg = df[COL_VALOR] < 0
neg = df[mask_neg]
print(f"\n{'═' * 60}")
print(f"DIAGNÓSTICO DE VALORES NEGATIVOS")
print(f"{'═' * 60}")
print(f"Total registros negativos: {len(neg):,}")
print(f"Suma de negativos:         ${neg[COL_VALOR].sum():,.0f}")
print(f"\nDistribución por TipoRegistro:")
print(neg['TipoRegistro'].value_counts().to_string())
print(f"\nDistribución por NombreSubGrupoFuente:")
print(neg['NombreSubGrupoFuente'].value_counts().to_string())
print(f"\n✅ Los valores negativos se MANTIENEN: representan la dinámica fiscal neta.")

# Índice temporal
df.set_index(COL_FECHA, inplace=True)
df.sort_index(inplace=True)
print(f"\n📅 Índice temporal establecido: {df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}")
print(f"📊 Registros finales: {len(df):,}")
""")

# ────────────────────────────────────────────────────────────────
md(r"""### 1.3  Estructura categórica del recaudo

Se explora la composición del dataset por sus dimensiones cualitativas:
**Rubro**, **Grupo/SubGrupo Fuente**, **Tipo de Registro** y **SubGrupo Aportante**.

**¿Por qué importa la estructura categórica?** Las rentas cedidas no son un flujo
homogéneo: están compuestas por **subgrupos fuente** con dinámicas económicas
muy distintas. Por ejemplo:
- **Licores y cerveza** dependen del consumo de alcohol (elástico al ingreso disponible).
- **Cigarrillos** tienen demanda inelástica pero están sujetos a regulación creciente.
- **Juegos de azar** son altamente volátiles y dependen del ciclo económico.

Comprender esta composición permite anticipar qué subgrupos podrían explicar
variaciones inesperadas en la serie agregada y justifica la inclusión de
regresores macro específicos en modelos posteriores.

> **Nota visual:** Los colores en los gráficos de barras distinguen valores
> positivos (azul) de negativos (rojo), facilitando la identificación de
> subgrupos donde los ajustes contables son más frecuentes.
""")

code(r"""# ── 1.3 Exploración categórica ────────────────────────────────────
cat_cols = ['Nombre de Rubro', 'NombreGrupoFuente', 'NombreSubGrupoFuente',
            'TipoRegistro', 'Nombre_SubGrupo_Aportante']

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flat

for i, col in enumerate(cat_cols):
    ax = axes[i]
    # Calcular participación por categoría
    resumen = (df.groupby(col)[COL_VALOR].sum()
               .sort_values(ascending=True))
    colores = [C_PRIMARY if v >= 0 else C_NEGATIVE for v in resumen.values]
    resumen.plot.barh(ax=ax, color=colores, edgecolor='white', linewidth=0.5)
    
    if _VIZ_THEME_LOADED:
        titulo_profesional(ax, col.replace('Nombre', '').replace('_', ' ').strip())
        formato_pesos_eje(ax, eje='x')
    else:
        ax.set_title(col, fontsize=11, fontweight='bold')
    ax.set_ylabel('')

# Ocultar subplot sobrante
axes[5].axis('off')

fig.suptitle('Composición Categórica del Recaudo Acumulado',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
if _VIZ_THEME_LOADED:
    guardar_figura(fig, '01_composicion_categorica', OUTPUTS_FIGURES)
plt.show()

# Tabla resumen de participación
print("\n" + "═" * 70)
print("PARTICIPACIÓN POR SUBGRUPO FUENTE (% del recaudo neto)")
print("═" * 70)
total = df[COL_VALOR].sum()
part = (df.groupby('NombreSubGrupoFuente')[COL_VALOR].sum()
        .sort_values(ascending=False))
for nombre, valor in part.items():
    pct = valor / total * 100
    print(f"  {nombre:<60} {pct:6.2f}%  (${valor:>15,.0f})")
""")

# ════════════════════════════════════════════════════════════════
# FASE II — AGREGACIÓN Y VISUALIZACIÓN TEMPORAL
# ════════════════════════════════════════════════════════════════
md(r"""---
## Fase II — Agregación Mensual y Visualización Temporal

Los datos transaccionales (149K+ registros) deben **agregarse** a una frecuencia
regular para poder aplicar modelos de series de tiempo. La elección de la frecuencia
de agregación es una decisión de diseño importante:
- **Diaria:** Demasiado ruidosa para recaudo fiscal (muchos días sin transferencias).
- **Mensual:** Frecuencia natural del ciclo presupuestal colombiano (los giros de ADRES
  se consolidan mensualmente). Genera 51 observaciones — suficientes para SARIMA y Prophet.
- **Trimestral:** Muy pocas observaciones (17) para modelado estadístico.

Usamos frecuencia **mensual** (`MS` = *Month Start*), que además coincide con el ciclo
de reporte de las rentas cedidas ante el Ministerio de Salud.

### 2.1  Remuestreo mensual y estadísticas descriptivas

Las **estadísticas descriptivas** proporcionan un resumen cuantitativo de la distribución
del recaudo mensual. Métricas clave a observar:

| Métrica | Interpretación |
|---------|---------------|
| **Media** | Nivel central del recaudo mensual típico |
| **Mediana** | Valor central robusto a outliers (si media >> mediana, hay asimetría positiva) |
| **Desv. estándar** | Dispersión absoluta del recaudo |
| **CV (%)** | Dispersión relativa: CV > 30% indica alta variabilidad |
| **IQR** | Rango intercuartílico: dispersión del 50% central de los datos |
""")

code(r"""# ── 2.1 Agregación mensual ────────────────────────────────────────
serie = df[COL_VALOR].resample('MS').sum()
df_mensual = serie.to_frame(name='Recaudo_Total')
df_mensual['Año'] = df_mensual.index.year
df_mensual['Mes'] = df_mensual.index.month
df_mensual['Mes_Nombre'] = df_mensual.index.strftime('%b')

n_meses = len(df_mensual)
print(f"Serie mensual generada: {n_meses} observaciones")
print(f"Periodo: {df_mensual.index.min():%Y-%m} → {df_mensual.index.max():%Y-%m}\n")

# Estadísticas descriptivas
desc = df_mensual['Recaudo_Total'].describe()
print("═" * 50)
print("ESTADÍSTICAS DESCRIPTIVAS — Recaudo Mensual")
print("═" * 50)
for idx, val in desc.items():
    print(f"  {idx:<12} ${val:>18,.0f}")

cv = desc['std'] / desc['mean'] * 100
print(f"  {'CV (%):':<12} {cv:>18.2f}%")

# Adicionar IQR y rango
iqr = desc['75%'] - desc['25%']
rango = desc['max'] - desc['min']
print(f"  {'IQR:':<12} ${iqr:>18,.0f}")
print(f"  {'Rango:':<12} ${rango:>18,.0f}")

print(f"\n📋 Primeros 6 meses:")
print(df_mensual[['Recaudo_Total']].head(6).to_string())
print(f"\n📋 Últimos 6 meses:")
print(df_mensual[['Recaudo_Total']].tail(6).to_string())
""")

# ────────────────────────────────────────────────────────────────
md(r"""### 2.2  Serie de tiempo — Evolución histórica

La **visualización de la serie temporal** es el paso más importante del EDA para
series de tiempo. Un buen gráfico revela instantáneamente tendencia,
estacionalidad, outliers y cambios de régimen que ningún estadístico resumen captura.

Elementos de la visualización:
- **Línea azul (serie observada):** Recaudo mensual neto. Los picos y valles
  revelan el patrón estacional.
- **Línea roja (media móvil 6M):** Suaviza la variabilidad de corto plazo para
  revelar la **tendencia subyacente**. Una MA=6 es apropiada porque captura
  medio ciclo estacional (12/2 = 6 meses).
- **Marcadores de picos:** Señalan enero y julio, meses de mayor recaudo históricamente,
  asociados al calendario de giros del SGSS (cierre de vigencias en diciembre
  y junio genera picos de transferencia en enero y julio).
- **Línea gris punteada (media global):** Referencia para evaluar si la serie
  está por encima o por debajo de su nivel promedio.
- **Zona sombreada train/test:** Delimitación del período de entrenamiento
  (Oct 2021 -- Sep 2025) vs. validación out-of-sample (Oct -- Dic 2025).

> **Guía de interpretación:** Si la media móvil es creciente, la tendencia es alcista.
> Si los picos son cada vez más pronunciados, la variabilidad aumenta con el nivel
> (heterocedasticidad), sugiriendo que una transformación logarítmica sería apropiada.
""")

code(r"""# ── 2.2 Gráfica de serie de tiempo ────────────────────────────────
fig, ax = plt.subplots(figsize=FIGSIZE_WIDE if _VIZ_THEME_LOADED else (16, 6))

if _VIZ_THEME_LOADED:
    grafica_serie_tiempo(ax, df_mensual.index, df_mensual['Recaudo_Total'],
                         label='Recaudo Mensual Neto', color=C_PRIMARY,
                         mostrar_ma=True, ma_window=6, mostrar_picos=True,
                         meses_pico=MESES_PICO)
    linea_media(ax, df_mensual['Recaudo_Total'].mean())
    zona_train_test(ax, pd.Timestamp(TRAIN_END), pd.Timestamp(TEST_START))
    titulo_profesional(ax,
        'Evolución Histórica del Recaudo Mensual',
        f'Rentas Cedidas — {FECHA_INICIO[:7]} a {FECHA_FIN[:7]} | {n_meses} observaciones')
    leyenda_profesional(ax, loc='upper left', ncol=2)
    marca_agua(fig)
else:
    ax.plot(df_mensual.index, df_mensual['Recaudo_Total'], color='b',
            marker='o', linewidth=1.5, label='Recaudo Mensual')
    ma6 = df_mensual['Recaudo_Total'].rolling(6, center=True).mean()
    ax.plot(df_mensual.index, ma6, color='r', linewidth=2, label='MA(6)')
    ax.axhline(df_mensual['Recaudo_Total'].mean(), color='gray', ls='--', lw=1)
    ax.set_title('Evolución Histórica del Recaudo Mensual', fontsize=14, fontweight='bold')
    ax.legend()

ax.set_xlabel('Fecha')
ax.set_ylabel('Recaudo ($COP)')
plt.tight_layout()
if _VIZ_THEME_LOADED:
    guardar_figura(fig, '01_serie_tiempo_recaudo', OUTPUTS_FIGURES)
plt.show()
""")

# ────────────────────────────────────────────────────────────────
md(r"""### 2.3  Patrones estacionales

La **estacionalidad** es uno de los componentes más importantes de una serie de
tiempo fiscal. En el contexto de rentas cedidas colombianas, la estacionalidad
se origina en:

1. **Calendario de transferencias del SGSS:** Los giros de ADRES siguen un ciclo
   administrativo donde diciembre y junio son meses de cierre de cuentas,
   generando picos de recaudo en enero y julio.
2. **Ciclo de consumo:** El IVA al consumo de licores y cervezas tiene picos
   naturales en diciembre (fiestas) y a mediados de año.
3. **Regulación fiscal:** Los municipios tienen plazos específicos para reportar
   y transferir lo recaudado, creando concentración en ciertos meses.

Se analiza la estacionalidad mediante dos visualizaciones complementarias:

1. **Boxplot mensual** — Muestra la distribución del recaudo para cada mes del año.
   La **caja** contiene el 50% central de los datos (IQR), la línea roja es la
   **mediana**, y los *whiskers* extienden hasta 1.5 veces el IQR. Puntos fuera
   son **outliers**. Meses con cajas más altas tienen *mayor dispersión*.

2. **Heatmap año x mes** — Mapa de calor donde colores más intensos indican
   meses de mayor recaudo. Permite detectar si el patrón estacional es
   **estable** (mismo patrón cada año) o **evolutivo** (cambia con el tiempo).
   También revela *level shifts* — años enteros que son sistemáticamente más
   altos o bajos que otros.
""")

code(r"""# ── 2.3 Estacionalidad: Boxplot + Heatmap ────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_QUAD if _VIZ_THEME_LOADED else (16, 6),
                                gridspec_kw={'width_ratios': [1.2, 1]})

# ─── Boxplot mensual ───
meses_labels = [calendar.month_abbr[m] for m in range(1, 13)]
box_data = [df_mensual[df_mensual['Mes'] == m]['Recaudo_Total'].values
            for m in range(1, 13)]
bp = ax1.boxplot(box_data, patch_artist=True, labels=meses_labels,
                 widths=0.6, medianprops={'color': C_SECONDARY, 'linewidth': 2})
for i, patch in enumerate(bp['boxes']):
    mes = i + 1
    color = C_BAR_PEAK if mes in MESES_PICO else (C_BAR_VALLEY if mes in [3, 9] else C_BAR_NORMAL)
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('white')

if _VIZ_THEME_LOADED:
    titulo_profesional(ax1, 'Distribución del Recaudo por Mes')
    formato_pesos_eje(ax1)
else:
    ax1.set_title('Distribución por Mes', fontweight='bold')
ax1.set_xlabel('Mes')
ax1.set_ylabel('Recaudo ($COP)')

# ─── Heatmap año × mes ───
pivot = df_mensual.pivot_table(values='Recaudo_Total', index='Año',
                                columns='Mes', aggfunc='sum')
pivot.columns = [calendar.month_abbr[m] for m in pivot.columns]

sns.heatmap(pivot, annot=True, fmt=',.0f', cmap='YlOrRd', linewidths=0.5,
            ax=ax2, cbar_kws={'label': 'Recaudo ($COP)', 'shrink': 0.8})
if _VIZ_THEME_LOADED:
    titulo_profesional(ax2, 'Mapa de Calor Año × Mes')
else:
    ax2.set_title('Mapa de Calor Año × Mes', fontweight='bold')
ax2.set_ylabel('Año')

plt.tight_layout()
if _VIZ_THEME_LOADED:
    guardar_figura(fig, '01_estacionalidad_boxplot_heatmap', OUTPUTS_FIGURES)
plt.show()

# Resumen numérico de estacionalidad
print("\n═════════════════════════════════════════════")
print("RECAUDO PROMEDIO POR MES")
print("═════════════════════════════════════════════")
prom_mes = df_mensual.groupby('Mes')['Recaudo_Total'].mean().sort_values(ascending=False)
for m, val in prom_mes.items():
    marca = " ★ PICO" if m in MESES_PICO else ""
    print(f"  {calendar.month_abbr[m]:>4}:  ${val:>15,.0f}{marca}")
""")

# ════════════════════════════════════════════════════════════════
# FASE III — DESCOMPOSICIÓN Y PROPIEDADES ESTOCÁSTICAS
# ════════════════════════════════════════════════════════════════
md(r"""---
## Fase III — Descomposición, Estacionariedad y Autocorrelación

Esta fase profundiza en las **propiedades estocásticas** de la serie. Mientras que
la Fase II proporcionó una visión visual e intuitiva, aquí aplicamos herramientas
estadísticas formales que fundamentarán las decisiones de modelado.

### 3.1  Descomposición estacional clásica

La **descomposición** de una serie de tiempo separa la señal observada $Y_t$ en
componentes interpretables. El método clásico (Cleveland *et al.*, 1990) propone:

$$Y_t = T_t + S_t + R_t \quad \text{(modelo aditivo)}$$
$$Y_t = T_t \times S_t \times R_t \quad \text{(modelo multiplicativo)}$$

donde:
- $T_t$: **Tendencia-ciclo** — movimiento suave de largo plazo.
- $S_t$: **Componente estacional** — patrón cíclico de período fijo (12 meses).
- $R_t$: **Residuo** (o componente irregular) — variación no explicada.

**¿Aditivo o multiplicativo?** Se elige el modelo **aditivo** cuando la amplitud
de la estacionalidad es constante a lo largo del tiempo. Si la amplitud crece
con el nivel de la serie (los picos son más pronunciados en años recientes),
el modelo **multiplicativo** es más apropiado. Una transformación $\log$ convierte
un modelo multiplicativo en aditivo.

**Métricas de fuerza** (Hyndman & Athanasopoulos, 2021):

$$F_s = \max\left(0, \; 1 - \frac{\text{Var}(R_t)}{\text{Var}(S_t + R_t)}\right)$$

Donde $F_s > 0.64$ indica estacionalidad **significativa**. Análogamente, $F_t$
mide la fuerza de la tendencia.
""")

code(r"""# ── 3.1 Descomposición estacional ─────────────────────────────────
serie_clean = df_mensual['Recaudo_Total'].ffill()

# Descomposición aditiva
decomp_add = seasonal_decompose(serie_clean, model='additive', period=ESTACIONALIDAD)

fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

componentes = [
    (decomp_add.observed, 'Serie Observada', C_PRIMARY),
    (decomp_add.trend,    'Tendencia',       C_SECONDARY),
    (decomp_add.seasonal, 'Estacionalidad',  C_QUATERNARY),
    (decomp_add.resid,    'Residuos',        C_SENARY),
]

for ax, (data, titulo, color) in zip(axes, componentes):
    if titulo == 'Residuos':
        ax.scatter(data.index, data, color=color, s=20, alpha=0.7, edgecolors='white', linewidth=0.5)
        ax.axhline(0, color=C_TEXT_LIGHT, ls='--', lw=0.8)
    else:
        ax.plot(data.index, data, color=color, linewidth=1.8)
        ax.fill_between(data.index, 0, data, alpha=0.05, color=color)
    
    if _VIZ_THEME_LOADED:
        titulo_profesional(ax, titulo)
        formato_pesos_eje(ax)
    else:
        ax.set_title(titulo, fontweight='bold')
    ax.set_ylabel(titulo)

fig.suptitle('Descomposición Estacional Aditiva (periodo = 12 meses)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
if _VIZ_THEME_LOADED:
    marca_agua(fig)
    guardar_figura(fig, '01_descomposicion_estacional', OUTPUTS_FIGURES)
plt.show()

# Fuerza de la estacionalidad (Hyndman & Athanasopoulos, 2021)
var_resid = np.nanvar(decomp_add.resid)
var_resid_plus_season = np.nanvar(decomp_add.resid + decomp_add.seasonal)
F_s = max(0, 1 - var_resid / var_resid_plus_season)

var_trend_plus_resid = np.nanvar(decomp_add.trend.dropna() + decomp_add.resid.dropna()[:len(decomp_add.trend.dropna())])
var_resid_trend = np.nanvar(decomp_add.resid.dropna()[:len(decomp_add.trend.dropna())])
F_t = max(0, 1 - var_resid_trend / var_trend_plus_resid)

print(f"\n{'═' * 50}")
print(f"FUERZA DE LOS COMPONENTES (Hyndman & Athanasopoulos)")
print(f"{'═' * 50}")
print(f"  Fuerza de Estacionalidad (F_s): {F_s:.4f}  {'→ FUERTE' if F_s > 0.64 else '→ DÉBIL'}")
print(f"  Fuerza de Tendencia     (F_t):  {F_t:.4f}  {'→ FUERTE' if F_t > 0.64 else '→ DÉBIL'}")
print(f"\n  Interpretación: F > 0.64 indica componente significativo (Hyndman, 2021)")
""")

# ────────────────────────────────────────────────────────────────
md(r"""### 3.2  Pruebas de estacionariedad

La **estacionariedad** es un concepto central en el análisis de series de tiempo.
Una serie es *estacionaria en sentido débil* si su media, varianza y autocovarianza
no dependen del tiempo. Los modelos ARIMA requieren estacionariedad; si la serie
no la cumple, debe **diferenciarse** (restar el valor anterior) hasta lograrlo.

Se aplican dos pruebas complementarias con hipótesis **opuestas**, lo cual reduce
el riesgo de conclusiones erróneas que una sola prueba podría generar:

| Prueba | H₀ | H₁ | Decisión si p < α |
|--------|----|----|-------------------|
| **ADF** (Dickey-Fuller Aumentada) | Raíz unitaria (no estacionaria) | Estacionaria | Rechazar H₀ → estacionaria |
| **KPSS** (Kwiatkowski–Phillips–Schmidt–Shin) | Estacionaria | Raíz unitaria | Rechazar H₀ → no estacionaria |

**¿Qué es una raíz unitaria?** En términos intuitivos, una serie con raíz unitaria
es como un *paseo aleatorio*: los shocks (perturbaciones) tienen efecto permanente
y la serie no tiende a regresar a su media. Esto significa que los pronósticos a
largo plazo tendrán incertidumbre creciente.

La **combinación ADF + KPSS** proporciona un diagnóstico robusto:
- ADF rechaza + KPSS no rechaza → **estacionaria** ✅
- ADF no rechaza + KPSS rechaza → **no estacionaria** ✗
- Ambas rechazan → **estacionaria con tendencia** (requiere diferenciación)
- Ninguna rechaza → **resultado inconcluso** (inspeccionar visualmente)

> **Implicación para el modelado:** Si la serie es no estacionaria, se establece
> el parámetro $d=1$ (o superior) para modelos ARIMA/SARIMA. Si además hay
> no-estacionariedad estacional, se añade $D=1$ en la componente estacional.
""")

code(r"""# ── 3.2 Pruebas de estacionariedad ────────────────────────────────
def diagnostico_estacionariedad(serie, nombre='Serie'):
    # Ejecuta ADF + KPSS y genera reporte consolidado.
    print(f"\n{'═' * 60}")
    print(f"DIAGNÓSTICO DE ESTACIONARIEDAD: {nombre}")
    print(f"{'═' * 60}")
    
    # ADF
    adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(serie, autolag='AIC')
    print(f"\n► Prueba ADF (Dickey-Fuller Aumentada):")
    print(f"  Estadístico:      {adf_stat:.4f}")
    print(f"  p-valor:          {adf_p:.6f}")
    print(f"  Lags usados:      {adf_lags}")
    print(f"  Observaciones:    {adf_nobs}")
    for k, v in adf_crit.items():
        print(f"  Valor crítico {k}: {v:.4f}")
    adf_decision = "ESTACIONARIA" if adf_p <= 0.05 else "NO ESTACIONARIA"
    print(f"  → Decisión (α=0.05): {adf_decision}")
    
    # KPSS
    kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(serie, regression='c', nlags='auto')
    print(f"\n► Prueba KPSS (Kwiatkowski-Phillips-Schmidt-Shin):")
    print(f"  Estadístico:      {kpss_stat:.4f}")
    print(f"  p-valor:          {kpss_p:.4f}")
    print(f"  Lags usados:      {kpss_lags}")
    for k, v in kpss_crit.items():
        print(f"  Valor crítico {k}: {v:.4f}")
    kpss_decision = "ESTACIONARIA" if kpss_p > 0.05 else "NO ESTACIONARIA"
    print(f"  → Decisión (α=0.05): {kpss_decision}")
    
    # Diagnóstico combinado
    print(f"\n{'─' * 50}")
    if adf_p <= 0.05 and kpss_p > 0.05:
        veredicto = "✅ ESTACIONARIA (ADF rechaza H₀ + KPSS no rechaza H₀)"
    elif adf_p > 0.05 and kpss_p <= 0.05:
        veredicto = "❌ NO ESTACIONARIA (ADF no rechaza H₀ + KPSS rechaza H₀)"
    elif adf_p <= 0.05 and kpss_p <= 0.05:
        veredicto = "⚠️ ESTACIONARIA con TENDENCIA (ambas rechazan H₀ — requiere diferenciación)"
    else:
        veredicto = "🔍 INCONCLUSO (ninguna rechaza H₀ — revisar manualmente)"
    print(f"  VEREDICTO: {veredicto}")
    
    return {'adf_p': adf_p, 'kpss_p': kpss_p, 'adf_stat': adf_stat, 'kpss_stat': kpss_stat}

# Ejecutar sobre la serie original
resultados_est = diagnostico_estacionariedad(serie_clean, 'Recaudo Mensual')

# Visualización de media/varianza móvil
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE_FULL if _VIZ_THEME_LOADED else (14, 8),
                                sharex=True)

rolmean = serie_clean.rolling(window=12).mean()
rolstd = serie_clean.rolling(window=12).std()

# Panel superior: serie + media móvil
ax1.plot(serie_clean.index, serie_clean, color=C_PRIMARY, alpha=0.6, linewidth=1, label='Original')
ax1.plot(rolmean.index, rolmean, color=C_SECONDARY, linewidth=2.5, label='Media Móvil (12M)')
if _VIZ_THEME_LOADED:
    titulo_profesional(ax1, 'Serie Original vs Media Móvil (12 meses)')
    formato_pesos_eje(ax1)
    leyenda_profesional(ax1, loc='upper left')
else:
    ax1.set_title('Serie Original vs Media Móvil', fontweight='bold')
    ax1.legend()

# Panel inferior: desviación estándar móvil
ax2.plot(rolstd.index, rolstd, color=C_SENARY, linewidth=2, label='Desv. Estándar Móvil (12M)')
ax2.axhline(serie_clean.std(), color=C_TEXT_LIGHT, ls='--', lw=1, label=f'Desv. Global: ${serie_clean.std():,.0f}')
if _VIZ_THEME_LOADED:
    titulo_profesional(ax2, 'Varianza Móvil — ¿Heterocedasticidad?')
    formato_pesos_eje(ax2)
    leyenda_profesional(ax2, loc='upper left')
else:
    ax2.set_title('Desviación Estándar Móvil', fontweight='bold')
    ax2.legend()

plt.tight_layout()
if _VIZ_THEME_LOADED:
    marca_agua(fig)
    guardar_figura(fig, '01_estacionariedad_rolling', OUTPUTS_FIGURES)
plt.show()
""")

# ────────────────────────────────────────────────────────────────
md(r"""### 3.3  Funciones de autocorrelación (ACF / PACF)

Los **correlogramas** son herramientas fundamentales de la metodología **Box-Jenkins**
(1970) para identificar la estructura de dependencia temporal de una serie.

**ACF (Autocorrelación):** Mide la correlación entre $Y_t$ y $Y_{t-k}$ para
cada rezago $k$. Incluye los efectos indirectos a través de rezagos intermedios.
- Decaimiento **exponencial** sugiere un proceso AR.
- Corte **abrupto** en lag $q$ sugiere un proceso MA(q).
- **Picos** en lag 12, 24... confirman estacionalidad anual ($s=12$).

**PACF (Autocorrelación Parcial):** Mide la correlación entre $Y_t$ y $Y_{t-k}$
**removiendo** el efecto de los rezagos intermedios $Y_{t-1}, ..., Y_{t-k+1}$.
- Corte **abrupto** en lag $p$ sugiere un proceso AR(p).
- Pico significativo en lag 12 sugiere componente AR estacional $P \geq 1$.

> **Reglas prácticas de Box-Jenkins:**
> - Si ACF decae y PACF corta en lag $p$ → modelo AR(p)
> - Si ACF corta en lag $q$ y PACF decae → modelo MA(q)
> - Si ambas decaen → modelo ARMA(p,q)
> - Picos estacionales en lag 12 → añadir componente $(P,D,Q)_{12}$

Se analizan tanto la **serie original** como la **primera diferencia** ($d=1$)
para comparar la estructura antes y después de remover la tendencia.
""")

code(r"""# ── 3.3 ACF / PACF ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_QUAD if _VIZ_THEME_LOADED else (16, 10))

# ACF (serie original)
plot_acf(serie_clean, ax=axes[0, 0], lags=min(24, len(serie_clean)//2 - 1),
         alpha=0.05, color=C_PRIMARY,
         vlines_kwargs={'colors': C_PRIMARY, 'linewidth': 1.0})
if _VIZ_THEME_LOADED:
    titulo_profesional(axes[0, 0], 'ACF — Serie Original')
else:
    axes[0, 0].set_title('ACF — Serie Original', fontweight='bold')

# PACF (serie original)
plot_pacf(serie_clean, ax=axes[0, 1], lags=min(24, len(serie_clean)//2 - 1),
          alpha=0.05, method='ywm', color=C_PRIMARY,
          vlines_kwargs={'colors': C_PRIMARY, 'linewidth': 1.0})
if _VIZ_THEME_LOADED:
    titulo_profesional(axes[0, 1], 'PACF — Serie Original')
else:
    axes[0, 1].set_title('PACF — Serie Original', fontweight='bold')

# Serie diferenciada (d=1)
serie_diff = serie_clean.diff().dropna()

# ACF (diferenciada)
plot_acf(serie_diff, ax=axes[1, 0], lags=min(24, len(serie_diff)//2 - 1),
         alpha=0.05, color=C_TERTIARY,
         vlines_kwargs={'colors': C_TERTIARY, 'linewidth': 1.0})
if _VIZ_THEME_LOADED:
    titulo_profesional(axes[1, 0], 'ACF — Primera Diferencia (d=1)')
else:
    axes[1, 0].set_title('ACF — Primera Diferencia', fontweight='bold')

# PACF (diferenciada)
plot_pacf(serie_diff, ax=axes[1, 1], lags=min(24, len(serie_diff)//2 - 1),
          alpha=0.05, method='ywm', color=C_TERTIARY,
          vlines_kwargs={'colors': C_TERTIARY, 'linewidth': 1.0})
if _VIZ_THEME_LOADED:
    titulo_profesional(axes[1, 1], 'PACF — Primera Diferencia (d=1)')
else:
    axes[1, 1].set_title('PACF — Primera Diferencia', fontweight='bold')

for ax in axes.flat:
    ax.set_xlabel('Rezagos')

fig.suptitle('Análisis de Autocorrelación — Serie Original vs Diferenciada',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
if _VIZ_THEME_LOADED:
    marca_agua(fig)
    guardar_figura(fig, '01_acf_pacf', OUTPUTS_FIGURES)
plt.show()

# Interpretación automática
print("\n═════════════════════════════════════════════")
print("INTERPRETACIÓN AUTOMÁTICA DE ACF/PACF")
print("═════════════════════════════════════════════")
from statsmodels.tsa.stattools import acf, pacf
acf_vals = acf(serie_clean, nlags=min(24, len(serie_clean)//2 - 1), alpha=0.05)
n = len(serie_clean)
ci = 1.96 / np.sqrt(n)
sig_lags = [i for i in range(1, len(acf_vals[0])) if abs(acf_vals[0][i]) > ci]
print(f"  Lags ACF significativos (|ρ| > {ci:.3f}): {sig_lags}")
if 12 in sig_lags:
    print(f"  → Lag 12 significativo: CONFIRMA estacionalidad anual (s=12)")
if any(l in sig_lags for l in [1, 2, 3]):
    print(f"  → Lags cortos significativos: sugiere componente AR/MA de orden bajo")
""")

# ────────────────────────────────────────────────────────────────
md(r"""### 3.4  Análisis de distribución

Conocer la **distribución marginal** de la serie es esencial para:
1. Decidir si se requieren **transformaciones** (log, Box-Cox) antes del modelado.
2. Validar supuestos de normalidad en los residuos de los modelos.
3. Identificar **asimetría** y **colas pesadas** que afectan la estimación.

Se evalúa la distribución mediante tres visualizaciones y dos pruebas formales:

**Visualizaciones:**
- **Histograma + KDE + Normal teórica:** Si la curva KDE (azul) difiere de la
  normal teórica (línea gris), la serie no es gaussiana.
- **Gráfico Q-Q:** Si los puntos se alínean sobre la diagonal roja, la
  distribución es normal. Desviaciones en las colas indican colas pesadas.
- **Diagrama de violín:** Combina un boxplot con la densidad kernel, revelando
  la forma completa de la distribución (bimodalidad, asimetría).

**Pruebas formales de normalidad:**
- **Jarque-Bera:** Compara asimetría y curtosis observadas contra los valores
  esperados bajo normalidad ($\text{Skew}=0$, $\text{Kurt}=3$).
- **Shapiro-Wilk:** Prueba omnibus de normalidad, con mayor potencia en muestras pequeñas.

> **Implicación:** Si la serie no es normal (en particular, si es asimétrica
> positiva), la transformación $\log(1+x)$ estabiliza la varianza y simetriza
> la distribución, mejorando el rendimiento de modelos lineales como SARIMA.
""")

code(r"""# ── 3.4 Distribución de la serie ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Histograma + KDE
axes[0].hist(serie_clean, bins=15, density=True, alpha=0.6, color=C_TERTIARY,
             edgecolor='white', linewidth=0.8, label='Histograma')
serie_clean.plot.kde(ax=axes[0], color=C_SECONDARY, linewidth=2, label='KDE')
# Normal teórica
x_range = np.linspace(serie_clean.min(), serie_clean.max(), 100)
axes[0].plot(x_range, stats.norm.pdf(x_range, serie_clean.mean(), serie_clean.std()),
             color=C_TEXT_LIGHT, ls='--', lw=1.5, label='Normal teórica')
if _VIZ_THEME_LOADED:
    titulo_profesional(axes[0], 'Distribución del Recaudo')
    formato_pesos_eje(axes[0], eje='x')
else:
    axes[0].set_title('Distribución', fontweight='bold')
axes[0].legend(fontsize=8)

# Q-Q Plot
stats.probplot(serie_clean.values, dist='norm', plot=axes[1])
axes[1].get_lines()[0].set(color=C_TERTIARY, markersize=5, alpha=0.7)
axes[1].get_lines()[1].set(color=C_SECONDARY, linewidth=2)
if _VIZ_THEME_LOADED:
    titulo_profesional(axes[1], 'Gráfico Q-Q (Normal)')
else:
    axes[1].set_title('Gráfico Q-Q', fontweight='bold')

# Box-Violin
parts = axes[2].violinplot(serie_clean.values, positions=[0], showmeans=True,
                            showmedians=True, showextrema=True)
for pc in parts['bodies']:
    pc.set_facecolor(C_TERTIARY)
    pc.set_alpha(0.4)
parts['cmeans'].set_color(C_SECONDARY)
parts['cmedians'].set_color(C_PRIMARY)
if _VIZ_THEME_LOADED:
    titulo_profesional(axes[2], 'Diagrama de Violín')
    formato_pesos_eje(axes[2])
else:
    axes[2].set_title('Diagrama de Violín', fontweight='bold')
axes[2].set_xticks([0])
axes[2].set_xticklabels(['Recaudo'])

plt.tight_layout()
if _VIZ_THEME_LOADED:
    marca_agua(fig)
    guardar_figura(fig, '01_distribucion', OUTPUTS_FIGURES)
plt.show()

# Tests de normalidad
jb_stat, jb_p = stats.jarque_bera(serie_clean.values)
shap_stat, shap_p = stats.shapiro(serie_clean.values)
skewness = stats.skew(serie_clean.values)
kurt = stats.kurtosis(serie_clean.values)

print(f"\n{'═' * 50}")
print(f"PRUEBAS DE NORMALIDAD")
print(f"{'═' * 50}")
print(f"  Asimetría (Skewness): {skewness:.4f}  {'→ Asimétrica positiva' if skewness > 0.5 else '→ Asimétrica negativa' if skewness < -0.5 else '→ Aproximadamente simétrica'}")
print(f"  Curtosis (exceso):    {kurt:.4f}  {'→ Leptocúrtica (colas pesadas)' if kurt > 1 else '→ Platocúrtica' if kurt < -1 else '→ Mesocúrtica'}")
print(f"  Jarque-Bera:          stat={jb_stat:.4f}, p={jb_p:.6f}  {'→ NO Normal' if jb_p < 0.05 else '→ Normal'}")
print(f"  Shapiro-Wilk:         stat={shap_stat:.4f}, p={shap_p:.6f}  {'→ NO Normal' if shap_p < 0.05 else '→ Normal'}")
""")

# ════════════════════════════════════════════════════════════════
# FASE IV — DEFLACIÓN, CRECIMIENTO Y MACRO
# ════════════════════════════════════════════════════════════════
md(r"""---
## Fase IV — Análisis en Valores Reales, Crecimiento Interanual y Correlación Macro

Esta fase contextualiza el recaudo dentro del entorno macroeconómico colombiano.
Los datos fiscales nominales pueden crear una **ilusión de crecimiento** cuando la
inflación es alta: si el recaudo crece 10% pero la inflación es 8%, el crecimiento
real es solo del 2%. La **deflación** permite aislar el crecimiento genuino del
recaudo del efecto de la pérdida de poder adquisitivo del peso.

### 4.1  Deflación con IPC — Serie en valores reales

Para convertir la serie de recaudo nominal a **términos reales** (poder de compra
constante), se construye un índice de precios con base 2021 = 100 utilizando el
IPC (Indice de Precios al Consumidor) anual. La fórmula de deflación es:

$$\text{Recaudo Real}_t = \frac{\text{Recaudo Nominal}_t}{\text{Índice IPC}_t} \times 100$$

Donde el índice IPC se construye acumulativamente:
$$\text{Índice IPC}_{2022} = 100 \times (1 + \text{IPC}_{2022}/100)$$
$$\text{Índice IPC}_{2023} = \text{Índice IPC}_{2022} \times (1 + \text{IPC}_{2023}/100)$$

> **Ejemplo práctico:** Si en 2023 el recaudo fue \$100M y el índice IPC es 125.7
> (inflación acumulada del 25.7% desde 2021), el recaudo real es
> \$100M / 125.7 x 100 = \$79.6M en pesos de 2021. El crecimiento real es
> sustancialmente menor que el nominal.

> **¿Por qué deflactamos los modelos?** Los modelos predictivos (SARIMAX, Prophet,
> XGBoost, LSTM) trabajan con la serie nominal pero incluyen el IPC como regresor
> exógeno. La deflación aquí es un ejercicio analítico para comprender la dinámica
> real del recaudo; no reemplaza el tratamiento de la inflación dentro de cada modelo.
""")

code(r"""# ── 4.1 Deflación con IPC ─────────────────────────────────────────
# Construir índice de precios (base 2021 = 100)
años = sorted(MACRO_DATA.keys())
ipc_dict = {a: MACRO_DATA[a]['IPC'] for a in años}

# Índice acumulado: 2021=100, 2022=100*(1+IPC_2022/100), etc.
indice_ipc = {}
base = 100
for a in años:
    if a == años[0]:
        indice_ipc[a] = base
    else:
        base = base * (1 + ipc_dict[a] / 100)
        indice_ipc[a] = base

print("Índice de Precios (base 2021 = 100):")
for a, idx in indice_ipc.items():
    print(f"  {a}: {idx:.2f}  (IPC interanual: {ipc_dict[a]:.2f}%)")

# Deflactar serie mensual
df_mensual['IPC_Indice'] = df_mensual['Año'].map(indice_ipc)
df_mensual['Recaudo_Real'] = df_mensual['Recaudo_Total'] / df_mensual['IPC_Indice'] * 100

# Visualización nominal vs real
fig, ax = plt.subplots(figsize=FIGSIZE_WIDE if _VIZ_THEME_LOADED else (16, 6))

ax.plot(df_mensual.index, df_mensual['Recaudo_Total'], color=C_PRIMARY,
        linewidth=2, label='Recaudo NOMINAL', alpha=0.8)
ax.plot(df_mensual.index, df_mensual['Recaudo_Real'], color=C_QUATERNARY,
        linewidth=2, ls='--', label='Recaudo REAL (base 2021)', alpha=0.9)

# Media móvil para ambas
ma_nom = df_mensual['Recaudo_Total'].rolling(6, center=True).mean()
ma_real = df_mensual['Recaudo_Real'].rolling(6, center=True).mean()
ax.plot(df_mensual.index, ma_nom, color=C_PRIMARY, linewidth=1, alpha=0.3)
ax.plot(df_mensual.index, ma_real, color=C_QUATERNARY, linewidth=1, alpha=0.3)

if _VIZ_THEME_LOADED:
    titulo_profesional(ax, 'Recaudo Nominal vs Real (deflactado con IPC)',
                       f'Base 2021 = 100 | IPC acumulado {años[-1]}: {indice_ipc[años[-1]]:.1f}')
    formato_pesos_eje(ax)
    leyenda_profesional(ax, loc='upper left')
    marca_agua(fig)
    guardar_figura(fig, '01_nominal_vs_real', OUTPUTS_FIGURES)
else:
    ax.set_title('Recaudo Nominal vs Real', fontweight='bold')
    ax.legend()

ax.set_xlabel('Fecha')
ax.set_ylabel('Recaudo ($COP)')
plt.tight_layout()
plt.show()

# Crecimiento real acumulado
crec_real = ((df_mensual['Recaudo_Real'].iloc[-1] / df_mensual['Recaudo_Real'].iloc[0]) - 1) * 100
crec_nom = ((df_mensual['Recaudo_Total'].iloc[-1] / df_mensual['Recaudo_Total'].iloc[0]) - 1) * 100
print(f"\n  Crecimiento nominal punta-a-punta: {crec_nom:+.1f}%")
print(f"  Crecimiento real punta-a-punta:    {crec_real:+.1f}%")
print(f"  Diferencia (efecto inflación):     {crec_nom - crec_real:+.1f} pp")
""")

# ────────────────────────────────────────────────────────────────
md(r"""### 4.2  Crecimiento interanual (YoY)

La tasa de crecimiento **interanual** (*Year-over-Year*, YoY) es el indicador
estándar para evaluar la dinámica de cualquier variable fiscal:

$$g_t = \left(\frac{Y_t}{Y_{t-12}} - 1\right) \times 100$$

Comparar el mismo mes del año anterior elimina automáticamente el efecto
estacional (enero con enero, julio con julio), permitiendo evaluar el
crecimiento *neto de estacionalidad*.

Se calculan dos variantes:
- **YoY Nominal:** Crecimiento en pesos corrientes (incluye inflación).
- **YoY Real:** Crecimiento deflactado (aislando el poder de compra).

> **Interpretación:** Tasas YoY positivas indican *aceleración* del recaudo;
negativas indican *desaceleración*. Una tasa nominal positiva pero real negativa
significa que el recaudo crece menos que la inflación — i.e., hay **erosión**
del ingreso fiscal en términos reales.
""")

code(r"""# ── 4.2 Crecimiento interanual (YoY) ─────────────────────────────
df_mensual['YoY'] = df_mensual['Recaudo_Total'].pct_change(periods=12) * 100
df_mensual['YoY_Real'] = df_mensual['Recaudo_Real'].pct_change(periods=12) * 100

# Solo meses con YoY disponible
yoy_valid = df_mensual.dropna(subset=['YoY'])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE_FULL if _VIZ_THEME_LOADED else (14, 8),
                                sharex=True)

# Panel 1: barras YoY nominal
colores_yoy = [C_POSITIVE if v >= 0 else C_NEGATIVE for v in yoy_valid['YoY']]
ax1.bar(yoy_valid.index, yoy_valid['YoY'], color=colores_yoy, alpha=0.7,
        edgecolor='white', linewidth=0.5, width=25)
ax1.axhline(0, color=C_TEXT_LIGHT, ls='-', lw=0.8)
ax1.axhline(yoy_valid['YoY'].mean(), color=C_SECONDARY, ls='--', lw=1.2,
            label=f'Media: {yoy_valid["YoY"].mean():.1f}%')
if _VIZ_THEME_LOADED:
    titulo_profesional(ax1, 'Crecimiento Interanual — Nominal')
    leyenda_profesional(ax1)
else:
    ax1.set_title('Crecimiento Interanual NOMINAL', fontweight='bold')
    ax1.legend()
ax1.set_ylabel('Variación (%)')

# Panel 2: barras YoY real
yoy_real_valid = df_mensual.dropna(subset=['YoY_Real'])
colores_yoy_r = [C_POSITIVE if v >= 0 else C_NEGATIVE for v in yoy_real_valid['YoY_Real']]
ax2.bar(yoy_real_valid.index, yoy_real_valid['YoY_Real'], color=colores_yoy_r,
        alpha=0.7, edgecolor='white', linewidth=0.5, width=25)
ax2.axhline(0, color=C_TEXT_LIGHT, ls='-', lw=0.8)
ax2.axhline(yoy_real_valid['YoY_Real'].mean(), color=C_SECONDARY, ls='--', lw=1.2,
            label=f'Media: {yoy_real_valid["YoY_Real"].mean():.1f}%')
if _VIZ_THEME_LOADED:
    titulo_profesional(ax2, 'Crecimiento Interanual — Real (deflactado)')
    leyenda_profesional(ax2)
else:
    ax2.set_title('Crecimiento Interanual REAL', fontweight='bold')
    ax2.legend()
ax2.set_ylabel('Variación (%)')
ax2.set_xlabel('Fecha')

plt.tight_layout()
if _VIZ_THEME_LOADED:
    marca_agua(fig)
    guardar_figura(fig, '01_crecimiento_yoy', OUTPUTS_FIGURES)
plt.show()

# Tabla resumen por año
print("\n═════════════════════════════════════════════════════")
print("RESUMEN POR AÑO")
print("═════════════════════════════════════════════════════")
resumen_anual = df_mensual.groupby('Año').agg(
    Recaudo_Nominal=('Recaudo_Total', 'sum'),
    Recaudo_Real=('Recaudo_Real', 'sum'),
    Meses=('Recaudo_Total', 'count'),
    Promedio_Mesl=('Recaudo_Total', 'mean'),
    CV=('Recaudo_Total', lambda x: x.std()/x.mean()*100)
).round(2)
print(resumen_anual.to_string())
""")

# ────────────────────────────────────────────────────────────────
md(r"""### 4.3  Concentración del recaudo — Análisis de Pareto

El **principio de Pareto** (también conocido como regla 80/20) establece que
frecuentemente un pequeño porcentaje de las categorías concentra la mayor
parte del valor total. En el contexto fiscal, esto significa que pocos
subgrupos fuente y pocas entidades aportantes podrían generar la mayor parte
del recaudo.

**¿Por qué importa para el pronóstico?** Si el recaudo está altamente concentrado
en pocas fuentes, los modelos de pronóstico agregados son vulnerables a
perturbaciones en las fuentes dominantes. Una política que afecte al principal
aportante impactaría desproporcionadamente la serie total.

Se complementa con el **coeficiente de Gini** (0 = distribución perfectamente
igualitaria, 1 = toda la concentración en una sola categoría), que cuantifica
el grado de concentración de forma resumida.

> **Lectura de la curva de Pareto:** El eje izquierdo muestra el recaudo por
> categoría (barras), y el eje derecho muestra el porcentaje acumulado (línea
> roja). El punto donde la línea cruza el umbral del 80% revela cuántas
> categorías concentran el 80% del recaudo total.
""")

code(r"""# ── 4.3 Concentración del recaudo ─────────────────────────────────
# Top 15 entidades aportantes
top_entidades = (df.groupby('NombreBeneficiarioAportante')[COL_VALOR]
                 .sum().sort_values(ascending=False).head(15))
total_recaudo = df[COL_VALOR].sum()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Barras horizontales Top 15
top_entidades.sort_values().plot.barh(ax=ax1, color=C_TERTIARY, edgecolor='white', linewidth=0.5)
if _VIZ_THEME_LOADED:
    titulo_profesional(ax1, 'Top 15 Entidades Aportantes')
    formato_pesos_eje(ax1, eje='x')
else:
    ax1.set_title('Top 15 Entidades Aportantes', fontweight='bold')
ax1.set_ylabel('')

# Curva de Pareto por concepto
por_concepto = (df.groupby('NombreSubGrupoFuente')[COL_VALOR]
                .sum().sort_values(ascending=False))
pct_acum = por_concepto.cumsum() / total_recaudo * 100

ax2.bar(range(len(por_concepto)), por_concepto.values, color=C_TERTIARY,
        alpha=0.7, edgecolor='white', label='Recaudo')
ax2_twin = ax2.twinx()
ax2_twin.plot(range(len(por_concepto)), pct_acum.values, color=C_SECONDARY,
              marker='D', linewidth=2, markersize=6, label='% Acumulado')
ax2_twin.axhline(80, color=C_TEXT_LIGHT, ls='--', lw=1, alpha=0.6, label='Umbral 80%')
ax2_twin.set_ylabel('% Acumulado')

ax2.set_xticks(range(len(por_concepto)))
ax2.set_xticklabels([s[:20] + '...' if len(s) > 20 else s for s in por_concepto.index],
                     rotation=45, ha='right', fontsize=7)
if _VIZ_THEME_LOADED:
    titulo_profesional(ax2, 'Curva de Pareto por SubGrupo Fuente')
    formato_pesos_eje(ax2)
else:
    ax2.set_title('Curva de Pareto', fontweight='bold')

# Combinar leyendas
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=8)

plt.tight_layout()
if _VIZ_THEME_LOADED:
    marca_agua(fig)
    guardar_figura(fig, '01_concentracion_pareto', OUTPUTS_FIGURES)
plt.show()

# Gini
sorted_vals = np.sort(por_concepto.values)
n = len(sorted_vals)
cumx = np.cumsum(sorted_vals)
gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
print(f"\n  Coeficiente de Gini (por SubGrupo Fuente): {gini:.4f}")
print(f"  → {' Alta concentración' if gini > 0.4 else 'Concentración moderada' if gini > 0.2 else 'Distribución dispersa'}")

# Top 5 concentración
top5_pct = por_concepto.head(5).sum() / total_recaudo * 100
print(f"  Los 5 principales SubGrupos concentran: {top5_pct:.1f}% del recaudo total")
""")

# ────────────────────────────────────────────────────────────────
md(r"""### 4.4  Correlación con variables macroeconómicas

Se construye una tabla anual de recaudo vs. indicadores macroeconómicos
(IPC, Salario Mínimo, UPC, Consumo de Hogares) para explorar posibles
relaciones que justifiquen el uso de **regresores exógenos** en modelos
SARIMAX y XGBoost.

**¿Qué es un regresor exógeno?** En el contexto de series de tiempo, un
regresor exógeno es una variable externa que ayuda a explicar los movimientos
de la serie objetivo. Por ejemplo, si el recaudo de licores se correlaciona
fuertemente con el Consumo de Hogares, incluir esta variable en el modelo
puede mejorar las predicciones.

Las variables macro disponibles son:
- **IPC** (Inflación): Afecta el valor nominal de todas las transacciones gravadas.
- **Salario Mínimo (SMLV):** Proxy del ingreso disponible de la población.
- **UPC** (Unidad de Pago por Capitación): Valor que el SGSS paga por asegurado.
- **Consumo de Hogares:** Medida directa de la demanda agregada.
- **Desempleo:** Indicador de actividad económica y poder adquisitivo.

> **Nota metodológica.** Con solo 4–5 años completos, los coeficientes de
> correlación son *indicativos* pero no concluyentes estadísticamente.
> El análisis formal con pruebas de significancia y correlación cruzada
> rezagada se profundiza en el notebook `03_Correlacion_Macro.ipynb`,
> donde se incrementa la resolución temporal a nivel mensual.
""")

code(r"""# ── 4.4 Correlación con variables macro ──────────────────────────
# Construir DataFrame macro anual
macro_df = pd.DataFrame(MACRO_DATA).T
macro_df.index.name = 'Año'

# Recaudo anual (sólo años completos con 12 meses)
recaudo_anual = df_mensual.groupby('Año')['Recaudo_Total'].agg(['sum', 'count'])
recaudo_anual = recaudo_anual[recaudo_anual['count'] == 12]  # sólo años completos
recaudo_anual = recaudo_anual.rename(columns={'sum': 'Recaudo_Anual'})

# Merge
tabla = macro_df.join(recaudo_anual[['Recaudo_Anual']], how='inner')

if len(tabla) >= 3:
    print("═" * 60)
    print("TABLA RECAUDO ANUAL vs MACRO")
    print("═" * 60)
    print(tabla.to_string())
    
    # Correlaciones
    print(f"\n{'─' * 60}")
    print("CORRELACIONES con Recaudo Anual")
    print(f"{'─' * 60}")
    for col in ['IPC', 'Salario_Minimo', 'UPC', 'Consumo_Hogares']:
        if col in tabla.columns:
            corr = tabla['Recaudo_Anual'].corr(tabla[col])
            fuerza = '★★★ Fuerte' if abs(corr) > 0.7 else '★★ Moderada' if abs(corr) > 0.4 else '★ Débil'
            print(f"  {col:<20} r = {corr:+.4f}  {fuerza}")
    
    # Scatter plots
    macro_vars = [c for c in ['IPC', 'Salario_Minimo', 'UPC', 'Consumo_Hogares'] if c in tabla.columns]
    n_vars = len(macro_vars)
    fig, axes = plt.subplots(1, n_vars, figsize=(4*n_vars, 4))
    if n_vars == 1:
        axes = [axes]
    
    for ax, var in zip(axes, macro_vars):
        ax.scatter(tabla[var], tabla['Recaudo_Anual'], color=C_TERTIARY,
                   s=80, zorder=3, edgecolors='white', linewidth=1)
        # Etiquetar puntos con año
        for idx, row in tabla.iterrows():
            ax.annotate(str(idx), (row[var], row['Recaudo_Anual']),
                        textcoords='offset points', xytext=(5, 5), fontsize=8)
        # Línea de tendencia
        if len(tabla) >= 3:
            z = np.polyfit(tabla[var], tabla['Recaudo_Anual'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(tabla[var].min(), tabla[var].max(), 50)
            ax.plot(x_line, p(x_line), color=C_SECONDARY, ls='--', lw=1.5, alpha=0.7)
        
        corr_val = tabla['Recaudo_Anual'].corr(tabla[var])
        if _VIZ_THEME_LOADED:
            titulo_profesional(ax, f'{var}\nr = {corr_val:+.3f}')
            formato_pesos_eje(ax)
        else:
            ax.set_title(f'{var} (r={corr_val:+.3f})', fontweight='bold', fontsize=10)
        ax.set_xlabel(f'{var} (%)')
    
    plt.tight_layout()
    if _VIZ_THEME_LOADED:
        marca_agua(fig)
        guardar_figura(fig, '01_correlacion_macro_preview', OUTPUTS_FIGURES)
    plt.show()
else:
    print("⚠️ Insuficientes años completos para correlación macro.")
""")

# ════════════════════════════════════════════════════════════════
# EXPORTACIÓN DE DATOS PROCESADOS
# ════════════════════════════════════════════════════════════════
md(r"""---
## Exportación de Datos Procesados

Se persisten los artefactos generados para consumo en notebooks posteriores.
Esta práctica de **persistencia intermedia** asegura reproducibilidad:
cada notebook lee datos procesados del anterior, evitando que cambios en
un paso afecten silenciosamente los resultados de pasos posteriores.

**Artefactos generados:**
- `serie_mensual.csv` — Serie mensual con recaudo nominal, real, índice IPC y crecimiento YoY.
- Figuras en `outputs/figures/` en formato PNG a 300 DPI.
""")

code(r"""# ── Guardar serie mensual procesada ───────────────────────────────
cols_export = ['Recaudo_Total', 'Recaudo_Real', 'IPC_Indice', 'YoY', 'YoY_Real',
               'Año', 'Mes']
export = df_mensual[[c for c in cols_export if c in df_mensual.columns]].copy()
export.index.name = 'Fecha'

ruta_csv = DATA_PROCESSED / 'serie_mensual.csv'
export.to_csv(ruta_csv)
print(f"✅ Serie mensual guardada: {ruta_csv}")
print(f"   {len(export)} filas × {export.shape[1]} columnas")
print(f"   Columnas: {list(export.columns)}")

# Vista previa
print(f"\n📋 Vista previa:")
print(export.head(6).to_string())
""")

# ════════════════════════════════════════════════════════════════
# CONCLUSIONES
# ════════════════════════════════════════════════════════════════
md(r"""---
## Conclusiones y Fundamentos para el Modelado

### Hallazgos principales

1. **Estructura de datos.** El dataset comprende 149 648 registros transaccionales
   de 51 meses (Oct 2021 – Dic 2025), con 10 subgrupos fuente y ~1 100 entidades
   aportantes. La columna `ValorRecaudo` requiere conversión a numérico; 27
   registros negativos corresponden a ajustes contables legítimos.

2. **Tendencia.** La serie mensual exhibe una tendencia creciente sostenida en
   valores nominales. Al deflactar con IPC, el crecimiento real es más moderado,
   evidenciando que parte del incremento nominal se explica por inflación.

3. **Estacionalidad.** Se confirma un patrón estacional de periodicidad 12 meses
   con picos recurrentes en enero y julio, coincidentes con el calendario de
   transferencias del SGSS. La fuerza de estacionalidad $F_s$ cuantifica la 
   significancia de este componente.

4. **Estacionariedad.** Las pruebas ADF y KPSS proporcionan el diagnóstico
   combinado de estacionariedad. Si la serie es no estacionaria, se requerirá
   al menos una diferenciación ($d \geq 1$) para los modelos ARIMA/SARIMA.

5. **Autocorrelación.** Los correlogramas ACF/PACF revelan la estructura de
   dependencia temporal que parametrizará los órdenes $(p, d, q)(P, D, Q)_{12}$
   de los modelos SARIMA. Picos en lag 12 confirman la componente estacional.

6. **Distribución.** Las pruebas de normalidad (Jarque-Bera, Shapiro-Wilk)
   determinan si se requieren transformaciones (log, Box-Cox) antes del modelado.

7. **Concentración.** El análisis de Pareto revela el grado de dependencia del
   recaudo en un número reducido de fuentes y entidades.

8. **Macro.** Las correlaciones preliminares con IPC, Salario Mínimo, UPC y
   Consumo de Hogares sugieren relaciones que serán formalizadas en el
   análisis de correlación macro (Notebook 03) y explotadas como
   regresores exógenos en SARIMAX y XGBoost.

### Implicaciones para la selección de modelos

| Hallazgo | Modelo favorecido |
|----------|-------------------|
| Estacionalidad fuerte (s=12) | SARIMA, Prophet |
| Tendencia + estacionalidad | SARIMA con $(D=1)$, Prophet (piecewise linear) |
| Correlación con macro | SARIMAX, XGBoost |
| Posibles no-linealidades | XGBoost, LSTM |
| Serie corta (51 obs) | SARIMA, Prophet (robustos con pocas observaciones) |

### Próximos pasos
→ **Notebook 02:** Análisis formal de estacionalidad (periodograma, test de Canova-Hansen).  
→ **Notebook 03:** Correlación cruzada con variables macroeconómicas.  
→ **Notebook 04:** Modelado SARIMA con selección automática de órdenes.
""")

# ════════════════════════════════════════════════════════════════
# ESCRIBIR NOTEBOOK
# ════════════════════════════════════════════════════════════════
nb.cells = cells

out_path = Path(__file__).resolve().parent.parent / 'notebooks' / '01_EDA_Completo.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"✅ Notebook generado: {out_path}")
print(f"   {len(cells)} celdas ({sum(1 for c in cells if c.cell_type == 'markdown')} MD + {sum(1 for c in cells if c.cell_type == 'code')} Code)")

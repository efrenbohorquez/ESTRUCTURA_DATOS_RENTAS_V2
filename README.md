# Sistema de Análisis y Pronóstico de Rentas Cedidas

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)](notebooks/)

**Modelo predictivo de series de tiempo para el recaudo tributario de Rentas
Cedidas** (licores, tabaco, juegos de azar) — fuente de financiamiento del
Régimen Subsidiado de Salud en Colombia.

> Desarrollado para la **ADRES** (Administradora de los Recursos del Sistema
> General de Seguridad Social en Salud) · Octubre 2021 – Diciembre 2025

---

## Problema

Las Rentas Cedidas financian la atención de ~24 millones de personas del
régimen subsidiado, pero presentan una **volatilidad del 42%** y variaciones
interanuales de hasta el 40%. Los métodos de proyección lineal actuales
generan errores superiores al 25%, impidiendo una planeación financiera
confiable para la ADRES.

## Solución

Pipeline reproducible de nueve notebooks que construye, evalúa y compara
cinco modelos de pronóstico mensual:

| # | Modelo | Paradigma | MAPE OOS |
|---|--------|-----------|----------|
| 1 | **XGBoost** | Gradient Boosting + SHAP | **5.05%** |
| 2 | Prophet | Bayesiano aditivo + festivos CO | 6.30% |
| 3 | SARIMAX | SARIMA + variables exógenas (IPC) | 9.75% |
| 4 | LSTM | Red neuronal recurrente | 23.52% |

**Modelo Ganador:** XGBoost con MAPE mensual del 5.05% y error trimestral
acumulado del 4.99% sobre el set de prueba Oct–Dic 2025.

---

## Estructura del Proyecto

```
ESTRUCTURA_DATOS_RENTAS_V2/
│
├── data/
│   ├── raw/                          # Archivos fuente (no versionados)
│   └── processed/                    # Series mensuales listas para modelar
│       ├── serie_mensual.csv         #   51 meses × 8 columnas
│       ├── serie_mensual_macro.csv   #   51 meses × 8 var. macroeconómicas
│       └── base_lstm_optimizada.csv  #   Dataset para la red LSTM
│
├── notebooks/                        # Pipeline reproducible (ejecutar 01→09)
│   ├── 00_config.py                  #   Configuración centralizada
│   ├── 01_EDA_Completo.ipynb         #   Análisis exploratorio de datos
│   ├── 02_Estacionalidad.ipynb       #   STL, ADF, KPSS, change-point
│   ├── 03_Correlacion_Macro.ipynb    #   Correlación con IPC, Salario, UPC
│   ├── 04_SARIMAX.ipynb              #   SARIMAX: SARIMA + exógenas (IPC)
│   ├── 05_Prophet.ipynb              #   Facebook Prophet + festivos CO
│   ├── 06_XGBoost.ipynb              #   Gradient Boosting + SHAP
│   ├── 07_LSTM.ipynb                 #   Red LSTM con Keras/TensorFlow
│   └── 08_Comparacion_Modelos.ipynb  #   Evaluación comparativa doctoral
│
├── scripts/                          # Automatización y utilidades
│   ├── build_01_eda.py … build_09_*  #   Generadores de notebooks
│   ├── viz_theme.py                  #   Tema visual profesional unificado
│   ├── model_helpers.py              #   Funciones auxiliares de modelos
│   └── utils.py                      #   Utilidades generales
│
├── outputs/
│   ├── figures/                      # ~105 gráficos de alta resolución
│   ├── forecasts/                    # CSVs de pronósticos por modelo
│   └── reports/                      # Métricas, PDFs y recomendaciones
│
├── docs/                             # Documentación técnica
│   ├── contexto_rentas_cedidas.md    #   Contexto del dominio
│   ├── propuesta_sistema_star.md     #   Propuesta de sistema de alertas
│   └── evidencia_referencias_soporte.md
│
├── legacy/                           # Notebooks exploratorios iniciales
├── requirements.txt                  # Dependencias Python
├── LICENSE                           # MIT License
└── .gitignore
```

---

## Instalación

### Requisitos

- Python 3.11 (requerido por TensorFlow)
- Git

### Configuración

```bash
# 1. Clonar el repositorio
git clone https://github.com/efrenbohorquez/ESTRUCTURA_DATOS_RENTAS_V2.git
cd ESTRUCTURA_DATOS_RENTAS_V2

# 2. Crear entorno virtual
python -m venv .venv

# 3. Activar entorno (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# En Linux/macOS: source .venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt
```

### Datos

Colocar el archivo Excel fuente en `data/raw/` y ejecutar el notebook 01 para
generar automáticamente los CSV procesados en `data/processed/`.

---

## Ejecución

### Pipeline completo (notebooks 01 → 09)

Abrir los notebooks en orden numérico en **Jupyter** o **VS Code**:

```bash
cd notebooks
jupyter lab
```

Cada notebook importa la configuración centralizada desde `00_config.py`.

### Regeneración automatizada

Los build scripts permiten regenerar cualquier notebook:

```bash
python scripts/build_01_eda.py            # Genera 01_EDA_Completo.ipynb
python scripts/build_08_comparacion.py    # Genera 08_Comparacion_Modelos.ipynb
```

---

## Datos

| Atributo | Valor |
|----------|-------|
| **Fuente** | BaseRentasCedidasVF (ADRES) |
| **Registros** | 149,648 transacciones |
| **Periodo** | Octubre 2021 – Diciembre 2025 |
| **Frecuencia** | Mensual (51 observaciones) |
| **Train** | Oct 2021 – Sep 2025 (48 meses) |
| **Test OOS** | Oct – Dic 2025 (3 meses) |

### Variables Macroeconómicas Incluidas

- **IPC** (Índice de Precios al Consumidor) — deflactor obligatorio
- **Salario Mínimo** (SMLV)
- **UPC** (Unidad de Pago por Capitación)
- **Consumo de Hogares**
- **Tasa de Desempleo**

---

## Resultados

### Ranking de Modelos — Validación OOS Oct–Dic 2025

| Modelo | MAPE (%) | RMSE (MM COP) | MAE (MM COP) | Error Trim. (%) |
|--------|----------|---------------|---------------|-----------------|
| **XGBoost** | **5.05** | **15.4** | **13.8** | **4.99** |
| Prophet | 6.30 | 28.7 | 19.3 | 5.03 |
| SARIMA | 13.99 | 42.5 | 39.6 | 2.69 |
| SARIMAX | 13.99 | 42.5 | 39.6 | 2.69 |
| LSTM | 23.52 | 73.5 | 59.6 | 21.54 |

### Hallazgos Clave

1. **XGBoost** logra el menor MAPE mensual (5.05%), confirmando que las no
   linealidades del recaudo fiscal son mejor capturadas por gradient boosting.
2. **Prophet** alcanza el segundo lugar (6.30%), validando su utilidad en
   contextos de datos escasos con estacionalidad fuerte.
3. **LSTM** confirma la hipótesis de escasez de datos (n < 48) — las redes
   neuronales requieren volúmenes superiores para superar modelos clásicos.
4. El **"Gráfico Comparativo de Curvas de Pronóstico"** demuestra que el recaudo es un proceso
   altamente predecible: los cinco paradigmas convergen en la forma de la curva.

---

## Documentación Técnica

| Documento | Descripción |
|-----------|-------------|
| [Contexto de Rentas Cedidas](docs/contexto_rentas_cedidas.md) | Marco fiscal y epidemiológico |
| [Propuesta STAR](docs/propuesta_sistema_star.md) | Sistema de Alerta y Recomendación Territorial |
| [Evidencia y Referencias](docs/evidencia_referencias_soporte.md) | Soporte bibliográfico |
| [Notebook 09 — Comparación](notebooks/08_Comparacion_Modelos.ipynb) | Evaluación doctoral de modelos |

---

## Stack Tecnológico

| Categoría | Herramienta |
|-----------|-------------|
| Lenguaje | Python 3.11 |
| Econometría | statsmodels, pmdarima |
| ML Bayesiano | Prophet 1.1 |
| Gradient Boosting | XGBoost 2.0 + SHAP |
| Deep Learning | TensorFlow / Keras 2.20 |
| Optimización | Optuna 3.4 |
| Visualización | Matplotlib, Seaborn |
| Notebooks | Jupyter, nbconvert |

---

## Autores

- **Efrén Bohorquez** — [@efrenbohorquez](https://github.com/efrenbohorquez)
- **Mauricio García**
- **Ernesto Sánchez**

---

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Ver [LICENSE](LICENSE) para
más detalles.

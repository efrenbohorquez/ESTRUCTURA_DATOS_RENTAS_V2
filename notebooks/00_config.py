"""
00_config.py — Configuración Centralizada del Sistema de Análisis de Rentas Cedidas
====================================================================================
Importar en cada notebook con:
    %run 00_config.py
"""

from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

warnings.filterwarnings('ignore')

# ============================================================
# 1. RUTAS DEL PROYECTO
# ============================================================
# Resolver PROJECT_ROOT de forma robusta (funciona con %run, exec, import)
try:
    _THIS_DIR = Path(__file__).resolve().parent
except NameError:
    _THIS_DIR = Path(os.getcwd())

PROJECT_ROOT = _THIS_DIR.parent if _THIS_DIR.name == 'notebooks' else _THIS_DIR
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUTS_FORECASTS = PROJECT_ROOT / "outputs" / "forecasts"
OUTPUTS_REPORTS = PROJECT_ROOT / "outputs" / "reports"

# Crear directorios si no existen
for _d in [DATA_RAW, DATA_PROCESSED, OUTPUTS_FIGURES, OUTPUTS_FORECASTS, OUTPUTS_REPORTS]:
    _d.mkdir(parents=True, exist_ok=True)

# Archivo fuente de datos principal (ruta definitiva confirmada)
DATA_FILE = PROJECT_ROOT / "BaseRentasCedidasVF.xlsx"

# Años a incluir en el análisis
ANOS_ANALISIS = [2021, 2022, 2023, 2024, 2025]

# ============================================================
# 2. PARÁMETROS DEL ANÁLISIS
# ============================================================
COL_FECHA = 'FechaRecaudo'
COL_VALOR = 'ValorRecaudo'
COL_RECAUDO_NETO = 'Recaudo_Neto'

# Periodo de análisis completo (Oct 2021 – Dic 2025, 51 meses)
FECHA_INICIO = '2021-10-01'
FECHA_FIN = '2025-12-31'

# Split Train/Test
# Train: Oct 2021 → Sep 2025 (48 meses)
# Test:  Oct 2025 → Dic 2025 (3 meses — comparar con datos reales)
TRAIN_END = '2025-09-30'
TEST_START = '2025-10-01'

# Ventana de validación = Test (Oct–Dic 2025)
VALIDATION_START = '2025-10-01'
VALIDATION_END = '2025-12-31'

# Horizonte de pronóstico de producción: 12 meses (Ene–Dic 2026)
HORIZONTE_PRONOSTICO = 12

# Horizonte de validación (3 meses Oct–Dic 2025)
HORIZONTE_TEST = 3

# Frecuencia estacional
ESTACIONALIDAD = 12

# ============================================================
# 3. VARIABLES MACROECONÓMICAS
# ============================================================
# Fuentes oficiales verificadas (Feb 2026):
# ── IPC (variación anual dic-dic):
#    DANE — Boletín Técnico IPC (dane.gov.co/index.php/estadisticas-por-tema/precios-y-costos/indice-de-precios-al-consumidor-ipc)
#    2021–2023: Serie histórica DANE confirmada.
#    2024: 5.20% — DANE Boletín Técnico IPC dic-2024.
#    2025: 5.10% — DANE Boletín Técnico IPC dic-2025; Wikipedia "Anexo:Indicadores económicos de Colombia".
#    2026: 5.10% — Carry-forward del último dato real (dic-2025). Año en curso, dato final desconocido.
#
# ── Salario Mínimo (% incremento anual SMLMV):
#    Decretos presidenciales publicados en Diario Oficial. Ref: Wikipedia "Salario mínimo en Colombia".
#    2021: $908,526  → aumento 3.50%  (Decreto dic-2020).
#    2022: $1,000,000 → aumento 10.07% (Decreto dic-2021).
#    2023: $1,160,000 → aumento 16.00% (Decreto dic-2022).
#    2024: $1,300,000 → aumento 12.07% (Decreto dic-2023).
#    2025: $1,423,500 → aumento 9.54%  (Decreto 24-dic-2024).
#    2026: $1,750,905 → aumento 23.00% (Decreto 30-dic-2025, Presidencia; "salario vital" = $2,000,000 con auxilio).
#
# ── UPC (% incremento anual — Unidad de Pago por Capitación):
#    MinSalud — Resoluciones anuales. Valores 2021–2024 confirmados con resoluciones publicadas.
#    2025–2026: Estimación pendiente verificación contra Res. MinSalud 2024/2025.
#    Nota: consultorsalud.com (fuente especializada) tiene datos detrás de paywall.
#
# ── Consumo_Hogares (% variación anual real — Gasto de Consumo Final de los Hogares):
#    Banco Mundial — Indicador NE.CON.PRVT.KD.ZG (Households and NPISHs Final consumption expenditure, annual % growth).
#    API: https://api.worldbank.org/v2/country/COL/indicator/NE.CON.PRVT.KD.ZG?format=json
#    Fuente primaria: DANE — Cuentas Nacionales Trimestrales, PIB por enfoque del gasto.
#    2021: 14.72% — Rebote post-COVID; fuerte recuperación de la demanda interna.
#    2022: 10.79% — Expansión impulsada por consumo e inversión; PIB +7.3%.
#    2023:  0.38% — Desaceleración marcada; PIB +0.6%; política monetaria restrictiva.
#    2024:  1.60% — Recuperación lenta; World Bank dato confirmado (lastupdated 2026-02-24).
#    2025:  2.60% — Estimación igual al crecimiento PIB 2025pr (DANE, 16-feb-2026). WB aún no publica.
#    2026:  2.50% — Promedio de proyecciones. Año en curso, dato final desconocido.
#
# ── Desempleo (tasa anual promedio, %) — DANE Gran Encuesta Integrada de Hogares (GEIH):
#    dane.gov.co/index.php/estadisticas-por-tema/mercado-laboral/empleo-y-desempleo
#    2021: 13.7% — Recuperación post-COVID, aún elevado por efectos de pandemia.
#    2022: 11.2% — Reactivación económica generalizada.
#    2023: 10.2% — Tendencia descendente continua; mercado laboral en estabilización.
#    2024:  9.8% — DANE GEIH dic-2024. Promedio anual confirmado.
#    2025:  9.5% — DANE GEIH dic-2025. Promedio anual preliminar.
#    2026:  9.3% — Proyección basada en tendencia descendente del mercado laboral.
#
MACRO_DATA = {
    2021: {'IPC': 5.62,  'Salario_Minimo': 3.50,  'UPC': 5.00,  'Consumo_Hogares': 14.72, 'Desempleo': 13.7},  # Fuentes verificadas
    2022: {'IPC': 13.12, 'Salario_Minimo': 10.07, 'UPC': 5.42,  'Consumo_Hogares': 10.79, 'Desempleo': 11.2},  # Fuentes verificadas
    2023: {'IPC': 9.28,  'Salario_Minimo': 16.00, 'UPC': 16.23, 'Consumo_Hogares': 0.38,  'Desempleo': 10.2},  # Fuentes verificadas
    2024: {'IPC': 5.20,  'Salario_Minimo': 12.07, 'UPC': 12.01, 'Consumo_Hogares': 1.60,  'Desempleo': 9.8},   # DANE GEIH confirmado
    2025: {'IPC': 5.10,  'Salario_Minimo': 9.54,  'UPC': 8.00,  'Consumo_Hogares': 2.60,  'Desempleo': 9.5},   # DANE GEIH preliminar
    2026: {'IPC': 5.10,  'Salario_Minimo': 23.00, 'UPC': 7.00,  'Consumo_Hogares': 2.50,  'Desempleo': 9.3},   # Proyección tendencial
}

MESES_PICO = [1, 7]        # Enero y Julio
MESES_FESTIVIDAD = [6, 12]  # Junio y Diciembre

# ============================================================
# 4. SISTEMA DE VISUALIZACIÓN PROFESIONAL
# ============================================================
_scripts_dir = str(PROJECT_ROOT / 'scripts')
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

_VIZ_THEME_LOADED = False
try:
    from viz_theme import (
        # Constantes de tipografía
        FONT_FAMILY, FONT_TITLE, FONT_SUBTITLE, FONT_AXIS, FONT_TICK,
        FONT_LEGEND, FONT_ANNOTATION, FONT_WATERMARK,
        # Colores principales
        C_PRIMARY, C_SECONDARY, C_TERTIARY, C_QUATERNARY,
        C_QUINARY, C_SENARY, C_SEPTENARY,
        # Colores de soporte
        C_GRID, C_BACKGROUND, C_TEXT, C_TEXT_LIGHT, C_HIGHLIGHT,
        C_CI_FILL, C_CI_BORDER, C_POSITIVE, C_NEGATIVE,
        C_TRAIN, C_TEST, C_BAR_PEAK, C_BAR_NORMAL, C_BAR_VALLEY,
        # Diccionario de colores por modelo
        COLORES_MODELOS, PALETTE_SEQUENTIAL, PALETTE_DIVERGING,
        # Dimensiones
        FIGSIZE_FULL, FIGSIZE_WIDE, FIGSIZE_STANDARD, FIGSIZE_DUAL,
        FIGSIZE_QUAD, FIGSIZE_SMALL, FIGSIZE_SQUARE,
        # Funciones
        aplicar_tema_profesional, formato_pesos, formato_pesos_eje,
        formato_porcentaje, titulo_profesional, marca_agua,
        anotar_pico, linea_media, zona_train_test, leyenda_profesional,
        guardar_figura,
        # Gráficas prediseñadas
        grafica_serie_tiempo, grafica_barras_estacional, grafica_residuos,
        grafica_pronostico, grafica_comparacion_modelos, tabla_metricas,
        grafica_radar,
    )
    _VIZ_THEME_LOADED = True
except ImportError as e:
    print(f'  ⚠️ viz_theme.py no cargado: {e} — usando tema básico')
    # --- Fallback mínimo ---
    C_PRIMARY = '#1B2A4A'; C_SECONDARY = '#C0392B'; C_TERTIARY = '#2980B9'
    C_QUATERNARY = '#27AE60'; C_QUINARY = '#E67E22'; C_SENARY = '#8E44AD'
    C_SEPTENARY = '#16A085'; C_CI_FILL = '#D5E8F0'
    FIGSIZE_STANDARD = (14, 6); FIGSIZE_WIDE = (16, 6)
    FIGSIZE_FULL = (14, 7); FIGSIZE_QUAD = (16, 12); FIGSIZE_SMALL = (8, 5)
    COLORES_MODELOS = {
        'real': C_PRIMARY, 'sarima': C_SECONDARY, 'sarimax': C_TERTIARY,
        'prophet': C_QUATERNARY, 'xgboost': C_QUINARY, 'lstm': C_SENARY,
        'ensemble': C_SEPTENARY, 'ci': C_CI_FILL,
    }
    def formato_pesos(valor, pos=None):
        if abs(valor) >= 1e9: return f'${valor/1e9:,.0f}MM'
        elif abs(valor) >= 1e6: return f'${valor/1e6:,.0f}M'
        return f'${valor:,.0f}'
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'figure.figsize': FIGSIZE_STANDARD, 'figure.dpi': 150,
                         'font.size': 11, 'axes.titlesize': 14, 'axes.titleweight': 'bold'})
    sns.set_palette('husl')

# Alias de compatibilidad (funciona con o sin viz_theme)
COLORES = {
    'real': C_PRIMARY, 'sarima': C_SECONDARY, 'sarimax': C_TERTIARY,
    'prophet': C_QUATERNARY, 'xgboost': C_QUINARY, 'lstm': C_SENARY,
    'ensemble': C_SEPTENARY, 'ci': C_CI_FILL,
}
FIGSIZE_LARGE = FIGSIZE_QUAD if _VIZ_THEME_LOADED else (16, 10)

# ============================================================
# 5. FUNCIÓN CENTRALIZADA DE CARGA DE DATOS
# ============================================================
import pandas as pd
import numpy as np

def cargar_datos(filtrar_anos=True, verbose=True):
    """
    Carga el dataset principal, convierte tipos y filtra años.
    Returns: DataFrame listo para análisis.
    """
    if verbose:
        print(f"Cargando: {DATA_FILE.name}")
    
    df = pd.read_excel(DATA_FILE)
    
    # Asegurar que FechaRecaudo sea datetime
    df[COL_FECHA] = pd.to_datetime(df[COL_FECHA], errors='coerce')
    
    # Convertir ValorRecaudo a numérico (viene como object/string en el Excel)
    df[COL_VALOR] = pd.to_numeric(
        df[COL_VALOR].astype(str).str.replace(',', '.', regex=False),
        errors='coerce'
    )
    
    # Filtrar solo años de análisis (excluir 2021)
    if filtrar_anos:
        df = df[df[COL_FECHA].dt.year.isin(ANOS_ANALISIS)].copy()
    
    # Ordenar por fecha
    df = df.sort_values(COL_FECHA).reset_index(drop=True)
    
    if verbose:
        print(f"  Dimensiones: {df.shape[0]:,} filas x {df.shape[1]} columnas")
        print(f"  Periodo: {df[COL_FECHA].min().strftime('%Y-%m-%d')} a {df[COL_FECHA].max().strftime('%Y-%m-%d')}")
        print(f"  Anos: {sorted(df[COL_FECHA].dt.year.unique())}")
    
    return df

# ============================================================
# 6. INFORMACIÓN DEL PROYECTO
# ============================================================
PROYECTO_NOMBRE = "Sistema de Análisis y Pronóstico de Rentas Cedidas"
PROYECTO_ENTIDAD = "Departamentos y Distritos de Colombia"
PROYECTO_PERIODO = f"{FECHA_INICIO} a {FECHA_FIN}"

print(f"Config cargada -- Datos: {DATA_FILE.name} | Periodo: {PROYECTO_PERIODO}")
if _VIZ_THEME_LOADED:
    print(f"  Tema profesional activo -- DPI 300, tipografia serif, paleta academica")

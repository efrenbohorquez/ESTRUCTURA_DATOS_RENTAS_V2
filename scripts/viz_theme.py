"""
viz_theme.py — Sistema de Visualización Profesional para Tesis de Maestría
===========================================================================
Tema centralizado de clase mundial para todas las gráficas del sistema.
Aplica tipografía académica, paleta de colores armónica, formateo monetario,
anotaciones automáticas y funciones helper reutilizables.

Uso:
    from viz_theme import *
    # Se aplica automáticamente al importar
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
from pathlib import Path

# ============================================================
# 1. TIPOGRAFÍA PROFESIONAL
# ============================================================
# Jerarquía tipográfica académica
FONT_FAMILY = 'serif'  # Estilo académico clásico
FONT_TITLE = {'family': FONT_FAMILY, 'weight': 'bold', 'size': 16}
FONT_SUBTITLE = {'family': FONT_FAMILY, 'weight': 'normal', 'size': 13, 'style': 'italic'}
FONT_AXIS = {'family': FONT_FAMILY, 'size': 12}
FONT_TICK = {'family': FONT_FAMILY, 'size': 11}
FONT_LEGEND = {'family': FONT_FAMILY, 'size': 10}
FONT_ANNOTATION = {'family': FONT_FAMILY, 'size': 10}
FONT_WATERMARK = {'family': FONT_FAMILY, 'size': 7, 'alpha': 0.4}

# ============================================================
# 2. PALETA DE COLORES — ACADEMIC DARK
# ============================================================
# Paleta armónica inspirada en publicaciones científicas de alto impacto

# Colores principales (saturados pero elegantes)
C_PRIMARY = '#1B2A4A'       # Azul marino profundo (datos reales)
C_SECONDARY = '#C0392B'     # Rojo académico (SARIMA)
C_TERTIARY = '#2980B9'      # Azul cielo (SARIMAX)
C_QUATERNARY = '#27AE60'    # Verde esmeralda (Prophet)
C_QUINARY = '#E67E22'       # Naranja cálido (XGBoost)
C_SENARY = '#8E44AD'        # Púrpura (LSTM)
C_SEPTENARY = '#16A085'     # Teal (Ensemble)

# Colores de soporte
C_GRID = '#E8ECF1'          # Gris muy claro para rejilla
C_BACKGROUND = '#FAFBFC'    # Fondo casi blanco
C_TEXT = '#2C3E50'          # Texto principal
C_TEXT_LIGHT = '#7F8C8D'    # Texto secundario
C_HIGHLIGHT = '#F39C12'     # Amarillo para destacar
C_CI_FILL = '#C8DDF0'       # Relleno intervalos de confianza (más visible)
C_CI_BORDER = '#7BAED4'     # Borde intervalos de confianza (más contraste)
C_POSITIVE = '#27AE60'      # Verde para valores positivos
C_NEGATIVE = '#E74C3C'      # Rojo para valores negativos
C_TRAIN = '#2C3E50'         # Zona de entrenamiento
C_TEST = '#E74C3C'          # Zona de prueba

# Paleta por modelo (para comparaciones)
COLORES_MODELOS = {
    'real': C_PRIMARY,
    'sarima': C_SECONDARY,
    'sarimax': C_TERTIARY,
    'prophet': C_QUATERNARY,
    'xgboost': C_QUINARY,
    'lstm': C_SENARY,
    'ensemble': C_SEPTENARY,
    'ci': C_CI_FILL,
}

# Paleta secuencial para heatmaps
PALETTE_SEQUENTIAL = ['#F7FBFF', '#D0E1F2', '#94C4DF', '#4A98C9', '#2171B5', '#08306B']
PALETTE_DIVERGING = ['#C0392B', '#E8D5D1', '#FAFBFC', '#D1DDE8', '#2980B9']

# Colores para barras de estacionalidad
C_BAR_PEAK = '#C0392B'      # Barras pico (Ene/Jul)
C_BAR_NORMAL = '#2980B9'    # Barras normales
C_BAR_VALLEY = '#85C1E9'    # Barras valle

# ============================================================
# 3. DIMENSIONES DE FIGURA
# ============================================================
FIGSIZE_FULL = (14, 7)       # Gráficas de página completa
FIGSIZE_WIDE = (16, 6)       # Gráficas panorámicas (series de tiempo)
FIGSIZE_STANDARD = (12, 6)   # Gráficas estándar
FIGSIZE_DUAL = (16, 7)       # Panel doble lado a lado
FIGSIZE_QUAD = (16, 12)      # Panel cuádruple (2x2)
FIGSIZE_SMALL = (8, 5)       # Gráficas compactas
FIGSIZE_SQUARE = (8, 8)      # Gráficas cuadradas (scatter, radar)

# ============================================================
# 4. CONFIGURACIÓN GLOBAL DE MATPLOTLIB
# ============================================================
def aplicar_tema_profesional():
    """
    Aplica el tema profesional globalmente a todas las gráficas matplotlib.
    Se ejecuta automáticamente al importar este módulo.
    """
    # Base style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Override completo
    plt.rcParams.update({
        # --- Figura ---
        'figure.figsize': FIGSIZE_STANDARD,
        'figure.dpi': 150,
        'figure.facecolor': C_BACKGROUND,
        'figure.edgecolor': 'none',
        'figure.autolayout': False,
        'figure.titlesize': 16,
        'figure.titleweight': 'bold',
        
        # --- Fuentes ---
        'font.family': FONT_FAMILY,
        'font.size': 11,
        
        # --- Ejes ---
        'axes.facecolor': 'white',
        'axes.edgecolor': '#D5D8DC',
        'axes.linewidth': 0.8,
        'axes.titlesize': 15,
        'axes.titleweight': 'bold',
        'axes.titlepad': 18,
        'axes.labelsize': 12,
        'axes.labelweight': 'normal',
        'axes.labelpad': 10,
        'axes.labelcolor': C_TEXT,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.prop_cycle': plt.cycler('color', [
            C_PRIMARY, C_SECONDARY, C_TERTIARY, C_QUATERNARY,
            C_QUINARY, C_SENARY, C_SEPTENARY
        ]),
        
        # --- Rejilla ---
        'axes.grid': True,
        'grid.color': C_GRID,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.7,
        'grid.linestyle': '-',
        
        # --- Ticks ---
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.color': C_TEXT,
        'ytick.color': C_TEXT,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.visible': False,
        
        # --- Leyenda ---
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '#D5D8DC',
        'legend.fancybox': True,
        'legend.shadow': False,
        'legend.borderpad': 0.6,
        'legend.labelspacing': 0.5,
        
        # --- Líneas ---
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'lines.markeredgewidth': 0.8,
        
        # --- Guardado ---
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
        'savefig.pad_inches': 0.15,
    })
    
    # Paleta de Seaborn
    sns.set_palette([C_PRIMARY, C_SECONDARY, C_TERTIARY, C_QUATERNARY,
                     C_QUINARY, C_SENARY, C_SEPTENARY])


# ============================================================
# 5. FORMATEADORES DE EJES
# ============================================================
def formato_pesos(valor, pos=None):
    """Formatea valores en pesos colombianos: $1.234M o $1.2B"""
    if abs(valor) >= 1e12:
        return f'${valor/1e12:,.1f}B'
    elif abs(valor) >= 1e9:
        return f'${valor/1e9:,.0f}MM'
    elif abs(valor) >= 1e6:
        return f'${valor/1e6:,.0f}M'
    elif abs(valor) >= 1e3:
        return f'${valor/1e3:,.0f}K'
    return f'${valor:,.0f}'


def formato_pesos_eje(ax, eje='y'):
    """Aplica formato de pesos colombianos al eje Y o X."""
    formatter = mticker.FuncFormatter(formato_pesos)
    if eje == 'y':
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


def formato_porcentaje(ax, eje='y', decimales=1):
    """Aplica formato de porcentaje al eje."""
    fmt = f'%.{decimales}f%%'
    formatter = mticker.FormatStrFormatter(fmt)
    if eje == 'y':
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


# ============================================================
# 6. FUNCIONES DE DECORACIÓN PROFESIONAL
# ============================================================
def titulo_profesional(ax, titulo, subtitulo=None):
    """
    Aplica título con tipografía profesional y subtítulo opcional en itálica.
    El subtítulo se coloca DEBAJO del título principal (dentro del área de pad)
    para evitar solapamiento.
    
    Ejemplo:
        titulo_profesional(ax, 
            'Serie de Tiempo: Recaudo Mensual',
            'Rentas Cedidas — Oct 2021 a Dic 2025')
    """
    if subtitulo:
        # Título principal arriba con pad grande para dejar espacio al subtítulo
        ax.set_title(titulo, fontdict=FONT_TITLE, loc='left', pad=32)
        # Subtítulo justo debajo del título (dentro del espacio de pad)
        ax.text(0.0, 1.005, subtitulo, transform=ax.transAxes,
                fontdict=FONT_SUBTITLE, color=C_TEXT_LIGHT, ha='left', va='bottom')
    else:
        ax.set_title(titulo, fontdict=FONT_TITLE, loc='left', pad=12)


def marca_agua(fig, texto='Tesis de Maestría — Rentas Cedidas'):
    """Agrega marca de agua sutil en esquina inferior derecha."""
    fig.text(0.98, 0.02, texto, ha='right', va='bottom',
             fontsize=7, color='#BDC3C7', fontstyle='italic',
             fontfamily=FONT_FAMILY, alpha=0.5)


def anotar_pico(ax, x, y, texto=None, offset=(0, 15)):
    """Anota un punto pico con flecha elegante."""
    if texto is None:
        texto = formato_pesos(y)
    ax.annotate(texto, xy=(x, y), xytext=offset,
                textcoords='offset points',
                fontsize=8, fontweight='bold', color=C_TEXT,
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=C_GRID, alpha=0.9),
                arrowprops=dict(arrowstyle='->', color=C_TEXT_LIGHT,
                              connectionstyle='arc3,rad=0.2', lw=0.8))


def linea_media(ax, valor, label=None, color=C_TEXT_LIGHT, estilo='--'):
    """Agrega línea horizontal de referencia (media, umbral, etc.)."""
    if label is None:
        label = f'Media: {formato_pesos(valor)}'
    ax.axhline(y=valor, color=color, linestyle=estilo, linewidth=1.0,
               alpha=0.6, label=label, zorder=1)


def zona_train_test(ax, train_end, test_start, ymin=None, ymax=None):
    """Agrega sombreado sutil para zonas train/test."""
    if ymin is None:
        ymin = ax.get_ylim()[0]
    if ymax is None:
        ymax = ax.get_ylim()[1]
    
    # Zona prueba (sombreado suave)
    ax.axvspan(test_start, ax.get_xlim()[1], alpha=0.08, color=C_TEST,
               label='Zona Prueba', zorder=0)
    ax.axvline(x=test_start, color=C_TEST, linestyle='--', linewidth=1.2,
               alpha=0.7, zorder=2)
    
    # Etiqueta del corte
    ax.text(test_start, ymax * 0.95, ' Prueba →', fontsize=8, color=C_TEST,
            fontweight='bold', va='top', ha='left', alpha=0.8)


def leyenda_profesional(ax, loc='best', ncol=1):
    """Crea leyenda con estilo profesional."""
    legend = ax.legend(loc=loc, ncol=ncol, frameon=True, framealpha=0.95,
                       edgecolor='#D5D8DC', fancybox=True, shadow=False,
                       borderpad=0.6, labelspacing=0.4,
                       prop={'family': FONT_FAMILY, 'size': 9})
    legend.get_frame().set_linewidth(0.5)
    return legend


def guardar_figura(fig, nombre, carpeta_figuras=None):
    """Guarda figura en alta resolución con fondo limpio."""
    if carpeta_figuras is None:
        carpeta_figuras = Path('.').resolve().parent / 'outputs' / 'figures'
    carpeta_figuras = Path(carpeta_figuras)
    carpeta_figuras.mkdir(parents=True, exist_ok=True)
    
    ruta = carpeta_figuras / f'{nombre}.png'
    fig.savefig(ruta, dpi=300, bbox_inches='tight', facecolor='white',
                edgecolor='none', pad_inches=0.15)
    print(f'  📊 Figura guardada: {ruta.name}')
    return ruta


# ============================================================
# 7. FUNCIONES DE GRÁFICAS PREDISEÑADAS
# ============================================================
def grafica_serie_tiempo(ax, fechas, valores, label='Recaudo Neto',
                         color=C_PRIMARY, mostrar_ma=True, ma_window=6,
                         mostrar_picos=True, meses_pico=[1, 7]):
    """
    Gráfica de serie de tiempo profesional con media móvil y picos.
    """
    import pandas as pd
    
    # Línea principal
    ax.plot(fechas, valores, color=color, linewidth=1.8, label=label,
            alpha=0.9, zorder=3)
    
    # Área sombreada bajo la curva
    ax.fill_between(fechas, 0, valores, alpha=0.06, color=color, zorder=1)
    
    # Media móvil
    if mostrar_ma and len(valores) >= ma_window:
        if isinstance(valores, pd.Series):
            ma = valores.rolling(window=ma_window, center=True).mean()
        else:
            ma = pd.Series(valores).rolling(window=ma_window, center=True).mean()
        ax.plot(fechas, ma, color=C_SECONDARY, linewidth=2.0, linestyle='-',
                alpha=0.7, label=f'Media Móvil ({ma_window}M)', zorder=4)
    
    # Marcadores de picos
    if mostrar_picos and hasattr(fechas, 'month'):
        mask_picos = fechas.month.isin(meses_pico)
        if isinstance(valores, pd.Series):
            picos_y = valores[mask_picos]
        else:
            picos_y = pd.Series(valores)[mask_picos]
        ax.scatter(fechas[mask_picos], picos_y, color=C_SECONDARY,
                   s=50, zorder=5, marker='D', edgecolors='white',
                   linewidth=1.0, label='Picos (Ene/Jul)')
    
    formato_pesos_eje(ax)
    return ax


def grafica_barras_estacional(ax, meses, valores, meses_pico=[1, 7]):
    """Barras de estacionalidad con colores diferenciados para picos."""
    colores = [C_BAR_PEAK if m in meses_pico else C_BAR_NORMAL for m in meses]
    
    bars = ax.bar(range(len(meses)), valores, color=colores, edgecolor='white',
                  linewidth=0.8, alpha=0.85, zorder=3)
    
    # Valor sobre cada barra
    for bar_obj, val in zip(bars, valores):
        ax.text(bar_obj.get_x() + bar_obj.get_width()/2, bar_obj.get_height(),
                formato_pesos(val), ha='center', va='bottom',
                fontsize=7, fontweight='bold', color=C_TEXT)
    
    # Media como línea
    media = np.mean(valores)
    ax.axhline(y=media, color=C_TEXT_LIGHT, linestyle='--', linewidth=1.0,
               alpha=0.6, label=f'Media: {formato_pesos(media)}')
    
    nombres_meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                     'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    ax.set_xticks(range(len(meses)))
    ax.set_xticklabels([nombres_meses[m-1] for m in meses])
    formato_pesos_eje(ax)
    return bars


def grafica_residuos(axes, residuos, titulo_prefix=''):
    """
    Panel 2x2 de diagnóstico de residuos: histograma, Q-Q, ACF, serie.
    axes debe ser un array 2x2 de subplots.
    """
    from scipy import stats
    from statsmodels.graphics.tsaplots import plot_acf
    
    ax1, ax2, ax3, ax4 = axes.flat
    
    # 1. Serie de residuos
    ax1.plot(residuos, color=C_PRIMARY, linewidth=1.0, alpha=0.7)
    ax1.axhline(y=0, color=C_SECONDARY, linestyle='-', linewidth=0.8)
    ax1.axhline(y=residuos.std()*2, color=C_TEXT_LIGHT, linestyle='--', linewidth=0.6, alpha=0.5)
    ax1.axhline(y=-residuos.std()*2, color=C_TEXT_LIGHT, linestyle='--', linewidth=0.6, alpha=0.5)
    ax1.fill_between(range(len(residuos)), -residuos.std()*2, residuos.std()*2,
                     alpha=0.05, color=C_TERTIARY)
    titulo_profesional(ax1, f'{titulo_prefix}Residuos vs Tiempo')
    
    # 2. Histograma
    ax2.hist(residuos, bins=20, color=C_TERTIARY, edgecolor='white',
             linewidth=0.8, alpha=0.7, density=True)
    x_range = np.linspace(residuos.min(), residuos.max(), 100)
    ax2.plot(x_range, stats.norm.pdf(x_range, residuos.mean(), residuos.std()),
             color=C_SECONDARY, linewidth=2.0, label='Normal teórica')
    titulo_profesional(ax2, f'{titulo_prefix}Distribución de Residuos')
    
    # 3. Q-Q Plot
    stats.probplot(residuos, dist='norm', plot=ax3)
    ax3.get_lines()[0].set(color=C_TERTIARY, markersize=4, alpha=0.6)
    ax3.get_lines()[1].set(color=C_SECONDARY, linewidth=1.5)
    titulo_profesional(ax3, f'{titulo_prefix}Q-Q Normal')
    
    # 4. ACF
    plot_acf(residuos, ax=ax4, lags=15, alpha=0.05,
             color=C_PRIMARY, vlines_kwargs={'colors': C_PRIMARY, 'linewidth': 1.0})
    titulo_profesional(ax4, f'{titulo_prefix}ACF de Residuos')
    
    return axes


def grafica_pronostico(ax, fechas_real, valores_real, fechas_pred, valores_pred,
                       ci_lower=None, ci_upper=None, modelo_nombre='Modelo',
                       color_pred=None):
    """
    Gráfica profesional de pronóstico vs real con intervalos de confianza.
    """
    if color_pred is None:
        color_pred = C_SECONDARY
    
    # Real
    ax.plot(fechas_real, valores_real, color=C_PRIMARY, linewidth=1.8,
            label='Observado', zorder=3)
    
    # Pronóstico
    ax.plot(fechas_pred, valores_pred, color=color_pred, linewidth=2.0,
            linestyle='--', label=f'Pronóstico {modelo_nombre}', zorder=4,
            marker='o', markersize=5, markerfacecolor='white',
            markeredgecolor=color_pred, markeredgewidth=1.5)
    
    # Intervalo de confianza
    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(fechas_pred, ci_lower, ci_upper,
                        alpha=0.20, color=color_pred, label='IC 95%', zorder=1)
        ax.plot(fechas_pred, ci_lower, color=color_pred, linewidth=0.8,
                alpha=0.5, linestyle='--')
        ax.plot(fechas_pred, ci_upper, color=color_pred, linewidth=0.8,
                alpha=0.5, linestyle='--')
    
    formato_pesos_eje(ax)
    return ax


def grafica_comparacion_modelos(ax, fechas, valores_dict, colores_dict=None):
    """
    Superpone pronósticos de múltiples modelos para comparación visual.
    valores_dict = {'SARIMA': series1, 'Prophet': series2, ...}
    """
    if colores_dict is None:
        colores_dict = COLORES_MODELOS
    
    estilos = ['-', '--', '-.', ':', '-', '--']
    markers = ['o', 's', 'D', '^', 'v', 'P']
    
    for i, (nombre, valores) in enumerate(valores_dict.items()):
        color = colores_dict.get(nombre.lower(), list(COLORES_MODELOS.values())[i % 7])
        ax.plot(fechas[:len(valores)], valores, color=color,
                linewidth=1.8, linestyle=estilos[i % len(estilos)],
                marker=markers[i % len(markers)], markersize=5,
                markerfacecolor='white', markeredgecolor=color,
                markeredgewidth=1.2, label=nombre, alpha=0.85)
    
    formato_pesos_eje(ax)
    return ax


def tabla_metricas(ax, metricas_dict, titulo='Métricas de Evaluación'):
    """
    Crea tabla visual de métricas dentro de un axes de matplotlib.
    metricas_dict = {'SARIMA': {'MAPE': 5.2, 'RMSE': 1200}, ...}
    """
    ax.axis('off')
    
    modelos = list(metricas_dict.keys())
    cols = list(list(metricas_dict.values())[0].keys())
    
    cell_text = []
    for modelo in modelos:
        row = [f'{metricas_dict[modelo][c]:.2f}' if isinstance(metricas_dict[modelo][c], float)
               else str(metricas_dict[modelo][c]) for c in cols]
        cell_text.append(row)
    
    tabla = ax.table(cellText=cell_text, rowLabels=modelos, colLabels=cols,
                     cellLoc='center', loc='center')
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.scale(1, 1.5)
    
    # Estilizar encabezados
    for j, col in enumerate(cols):
        tabla[0, j].set_facecolor(C_PRIMARY)
        tabla[0, j].set_text_props(color='white', fontweight='bold')
    
    # Estilizar filas alternadas
    for i in range(len(modelos)):
        color_fila = '#F8F9FA' if i % 2 == 0 else 'white'
        for j in range(len(cols)):
            tabla[i+1, j].set_facecolor(color_fila)
    
    titulo_profesional(ax, titulo)
    return tabla


# ============================================================
# 8. RADAR CHART PROFESIONAL
# ============================================================
def grafica_radar(ax, categorias, valores_dict, colores_dict=None):
    """
    Radar chart profesional para comparación multidimensional de modelos.
    categorias = ['MAPE', 'RMSE', 'MAE', 'R²', 'AIC']
    valores_dict = {'SARIMA': [0.8, 0.6, ...], 'Prophet': [...]}
    """
    if colores_dict is None:
        colores_dict = COLORES_MODELOS
    
    N = len(categorias)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Cerrar polígono
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    
    # Rejilla personalizada
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categorias, fontsize=9, fontfamily=FONT_FAMILY)
    ax.grid(color=C_GRID, linewidth=0.5)
    
    for nombre, valores in valores_dict.items():
        vals = valores + valores[:1]  # Cerrar
        color = colores_dict.get(nombre.lower(), C_PRIMARY)
        ax.plot(angles, vals, color=color, linewidth=2.0, label=nombre)
        ax.fill(angles, vals, color=color, alpha=0.1)
    
    return ax


# ============================================================
# 9. APLICACIÓN AUTOMÁTICA
# ============================================================
aplicar_tema_profesional()

print('  🎨 Tema profesional aplicado — Tipografía serif, paleta académica, DPI 300')

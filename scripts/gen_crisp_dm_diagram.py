"""
Genera el diagrama CRISP-DM combinado: rueda donut + callouts infografia
para el proyecto Rentas Cedidas — ADRES Colombia.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / 'outputs' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# PALETA
# ══════════════════════════════════════════════════════════════
BG = '#F5F5F0'
C_CENTER_BG = '#FFFFFF'
C_CENTER_TXT = '#2C3E50'
C_ARROW_RING = '#7B2D8E'
C_FOOT = '#7F8C8D'
C_DETAIL_TXT = '#2C3E50'
C_TITLE = '#1B3A6B'

PHASES = [
    {'name': '1. Comprension\ndel negocio',    'color': '#1B3A6B'},
    {'name': '2. Comprension\nde los datos',   'color': '#1DA5C9'},
    {'name': '3. Preparacion\nde los datos',   'color': '#1A9B72'},
    {'name': '4. Modelado\ny optimizacion',    'color': '#F59E2A'},
    {'name': '5. Evaluacion\nmulti-horizonte', 'color': '#D6275A'},
    {'name': '6. Despliegue\n(STAR)',           'color': '#7B2D8E'},
]

CALLOUTS = [
    {'title': '1  COMPRENSION DEL NEGOCIO',
     'lines': ['Salud publica y\nsostenibilidad financiera',
               'Volatilidad del 42 %',
               'MAPE objetivo <= 15 %'],
     'pos': (-8.8, 5.5), 'anchor_deg': 61},
    {'title': '2  COMPRENSION DE DATOS',
     'lines': ["'149.648 registros'",
               'IPC - Indice de Precios',
               'SMLV - Salario Minimo',
               '51 meses / 1,143 entidades'],
     'pos': (8.8, 5.5), 'anchor_deg': 1},
    {'title': '3  PREPARACION DE DATOS',
     'lines': ['serie_mensual.csv',
               'Lag_12 estacional',
               'Estacionalidad Bimodal',
               'Train 48 m / Test 3 m'],
     'pos': (9.5, -1.5), 'anchor_deg': -59},
    {'title': '4  MODELADO',
     'lines': ['SARIMAX (0,1,1)(1,1,1,12)',
               'Prophet (changepoints)',
               'XGBoost (lags + macro)',
               'LSTM (seq 12, 64 units)'],
     'pos': (5.5, -7.5), 'anchor_deg': -119},
    {'title': '5  EVALUACION',
     'lines': ['MAPE 5.05 %',
               '(XGBoost superiority)',
               'Prophet 6.30 % / SARIMAX 9.75 %',
               'LSTM 13.58 %'],
     'pos': (-5.5, -7.5), 'anchor_deg': -179},
    {'title': '6  DESPLIEGUE (STAR)',
     'lines': ['STAR System Dashboard',
               'Semaforos: Rojo / Naranja / Verde',
               'Pronostico Ene-Dic 2026',
               'Monitoreo mensual + retrain'],
     'pos': (-9.5, -1.5), 'anchor_deg': 121},
]

n = len(PHASES)
GAP_DEG  = 3.0
SPAN_DEG = (360 - n * GAP_DEG) / n
R_INNER, R_OUTER, R_RING = 1.80, 3.80, 4.30

start_angles = []
a = 90
for i in range(n):
    start_angles.append(a)
    a -= (SPAN_DEG + GAP_DEG)

# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(22, 14), subplot_kw={'aspect': 'equal'})
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.axis('off')

# ── Titulo ───────────────────────────────────────────────────
ax.text(0, 8.8,
        'ANALISIS DE RENTAS CEDIDAS EN COLOMBIA',
        fontsize=22, fontweight='bold', fontfamily='serif',
        color=C_TITLE, ha='center', va='center', zorder=20)

# ── 1. Donut wheel ───────────────────────────────────────────
for i, phase in enumerate(PHASES):
    theta1 = start_angles[i] - SPAN_DEG
    theta2 = start_angles[i]
    wedge = Wedge((0, 0), R_OUTER, theta1, theta2, width=R_OUTER - R_INNER,
                  facecolor=phase['color'], edgecolor='white', linewidth=2.5, zorder=5)
    ax.add_patch(wedge)
    mid_a = np.radians((theta1 + theta2) / 2)
    r_t = (R_INNER + R_OUTER) / 2
    ax.text(r_t * np.cos(mid_a), r_t * np.sin(mid_a), phase['name'],
            fontsize=12, fontweight='bold', color='white', ha='center', va='center',
            fontfamily='serif', linespacing=1.2, zorder=7,
            path_effects=[pe.withStroke(linewidth=2, foreground=phase['color'])])

# Centro
ax.add_patch(plt.Circle((0, 0), R_INNER - 0.10, facecolor=C_CENTER_BG,
                         edgecolor='#D0D0D0', linewidth=1.5, zorder=6))
ax.text(0, 0.45, 'CENTRO\nDE DATOS', fontsize=14, fontweight='bold',
        color=C_CENTER_TXT, ha='center', va='center', fontfamily='serif',
        linespacing=1.1, zorder=10)
ax.text(0, -0.20, 'ADRES', fontsize=16, fontweight='bold',
        color=C_TITLE, ha='center', va='center', fontfamily='serif', zorder=10)
ax.text(0, -0.65, 'Rentas Cedidas\n2021 - 2025', fontsize=9,
        color='#95A5A6', ha='center', va='center', fontfamily='serif',
        linespacing=1.2, zorder=10)

# ── 2. Anillo flechas ────────────────────────────────────────
ax.add_patch(plt.Circle((0, 0), R_RING, fill=False,
                         edgecolor=C_ARROW_RING, linewidth=2.0, zorder=3))
arrow_pos = [start_angles[i] - SPAN_DEG / 2 for i in range(n)]
for i in range(n):
    a_s = np.radians(arrow_pos[i] - 10)
    a_e = np.radians(arrow_pos[(i + 1) % n] + 10)
    ax.add_patch(FancyArrowPatch(
        (R_RING * np.cos(a_s), R_RING * np.sin(a_s)),
        (R_RING * np.cos(a_e), R_RING * np.sin(a_e)),
        arrowstyle='->,head_length=6,head_width=4',
        color=C_ARROW_RING, linewidth=2.0,
        connectionstyle='arc3,rad=-0.25', zorder=4))

# ── 3. Callout boxes ─────────────────────────────────────────
for i, (co, phase) in enumerate(zip(CALLOUTS, PHASES)):
    cx, cy = co['pos']
    color = phase['color']

    bw, bh_title = 3.80, 0.55
    title_box = FancyBboxPatch((cx - bw/2, cy - bh_title/2 + 0.60), bw, bh_title,
                               boxstyle='round,pad=0.12',
                               facecolor=color, edgecolor='white',
                               linewidth=2, zorder=12)
    ax.add_patch(title_box)
    ax.text(cx, cy + 0.60, co['title'],
            fontsize=9.5, fontweight='bold', color='white',
            ha='center', va='center', fontfamily='serif', zorder=13)

    detail_text = '\n'.join(co['lines'])
    n_lines = len(co['lines'])
    bh_det = 0.38 * n_lines + 0.30
    det_y = cy + 0.60 - bh_title/2 - bh_det/2 - 0.10
    detail_box = FancyBboxPatch((cx - bw/2, det_y - bh_det/2), bw, bh_det,
                                boxstyle='round,pad=0.10',
                                facecolor='white', edgecolor=color,
                                linewidth=1.5, alpha=0.95, zorder=11)
    ax.add_patch(detail_box)
    ax.text(cx, det_y, detail_text,
            fontsize=8.5, color=C_DETAIL_TXT, ha='center', va='center',
            fontfamily='serif', linespacing=1.45, zorder=12)

    anchor_rad = np.radians(co['anchor_deg'])
    wx = (R_OUTER + 0.30) * np.cos(anchor_rad)
    wy = (R_OUTER + 0.30) * np.sin(anchor_rad)
    box_cx, box_cy = cx, det_y
    dx = box_cx - wx
    dy = box_cy - wy
    dist = np.sqrt(dx**2 + dy**2)
    bx = box_cx - dx/dist * min(bw/2, abs(dx))
    by = box_cy - dy/dist * min(bh_det/2 + 0.30, abs(dy))
    ax.plot([wx, bx], [wy, by], color=color, lw=1.5, alpha=0.6, ls='--', zorder=3)
    ax.plot(wx, wy, 'o', color=color, markersize=5, zorder=4)

# ── 4. Pie de pagina ─────────────────────────────────────────
ax.text(0, -9.2,
        'Ciclo iterativo de retroalimentacion - el despliegue puede reiniciar el proceso',
        fontsize=10, color=C_FOOT, ha='center', va='center',
        fontfamily='serif', style='italic', zorder=10)
ax.text(0, -9.7,
        'Sistema de Analisis y Pronostico de Rentas Cedidas / ADRES Colombia / 2026',
        fontsize=9, color='#B0B0B0', ha='center', va='center',
        fontfamily='serif', zorder=10)

# ── Limites y guardado ───────────────────────────────────────
ax.set_xlim(-12.5, 12.5)
ax.set_ylim(-10.5, 9.8)
plt.tight_layout()
out = OUT_DIR / 'crisp_dm_ciclo_completo.png'
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=BG)
print(f'Guardado en: {out}')
plt.close()

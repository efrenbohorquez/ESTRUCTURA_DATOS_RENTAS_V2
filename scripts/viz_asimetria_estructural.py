import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import importlib.util

# Cargar configuracion si es posible
try:
    config_path = Path.cwd() / 'notebooks' / '00_config.py'
    if config_path.exists():
        spec = importlib.util.spec_from_file_location("config", config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        
        COL_VALOR = config.COL_VALOR
        C_PRIMARY = config.C_PRIMARY
        C_SECONDARY = config.C_SECONDARY
        OUTPUTS_FIGURES = config.OUTPUTS_FIGURES
        DATA_FILE = config.DATA_FILE
        PROJECT_ROOT = config.PROJECT_ROOT
        PROYECTO_ENTIDAD = config.PROYECTO_ENTIDAD
    else:
        raise FileNotFoundError("Archivo de configuracion no encontrado.")
except Exception as e:
    print(f"Advertencia: No se pudo cargar la configuracion: {e}. Usando valores por defecto.")
    COL_VALOR = 'ValorRecaudo'
    C_PRIMARY = '#1B2A4A'
    C_SECONDARY = '#C0392B'
    OUTPUTS_FIGURES = Path.cwd() / "outputs" / "figures"
    DATA_FILE = Path.cwd() / "BaseRentasVF_limpieza21feb_sin2021_ene_sep.xlsx"
    PROJECT_ROOT = Path.cwd()
    PROYECTO_ENTIDAD = "Departamentos de Colombia"

def visualizar_asimetria():
    print("Analisis de asimetria estructural iniciado...")
    
    # 1. Cargar datos
    if not DATA_FILE.exists():
        print(f"Error: No se encuentra el archivo {DATA_FILE}")
        return

    # Intentar cargar Excel
    try:
        df = pd.read_excel(DATA_FILE)
    except Exception as e:
        print(f"Error al leer Excel: {e}")
        return
    
    # 2. Agrupar por Departamento (Beneficiario Aportante)
    col_depto = None
    for col in ['NombreBeneficiarioAportante', 'Departamento', 'NombreDepartamento', 'DEPTO']:
        if col in df.columns:
            col_depto = col
            break
    
    if not col_depto:
        print("Error: No se encontro columna de Departamento.")
        # Mostrar columnas disponibles para ayudar al debug
        print(f"Columnas detectadas: {list(df.columns)}")
        return

    # Limpiar nombres (quitar 'DEPARTAMENTO DE ', 'GOBERNACION DE ', etc.)
    df[col_depto] = df[col_depto].str.replace('DEPARTAMENTO DE ', '', case=False)
    df[col_depto] = df[col_depto].str.replace('GOBERNACION DE ', '', case=False)
    df[col_depto] = df[col_depto].str.replace('DISTRITO DE ', '', case=False)
    df[col_depto] = df[col_depto].str.strip()


    depto_recaudo = df.groupby(col_depto)[COL_VALOR].sum().sort_values(ascending=False).reset_index()
    depto_recaudo['Porcentaje'] = (depto_recaudo[COL_VALOR] / depto_recaudo[COL_VALOR].sum()) * 100
    depto_recaudo['PorcentajeAcumulado'] = depto_recaudo['Porcentaje'].cumsum()

    # 3. Graficar Pareto de Departamentos
    plt.figure(figsize=(14, 8))
    
    # Barras de recaudo
    ax = sns.barplot(x=col_depto, y='Porcentaje', data=depto_recaudo, color=C_PRIMARY, alpha=0.7)
    plt.xticks(rotation=90)
    plt.ylabel('Porcentaje del Recaudo Total (%)')
    plt.title('Asimetria Estructural: Concentracion del Recaudo por Departamento', fontsize=16, fontweight='bold')
    
    # Linea de Pareto (Acumulado)
    ax2 = ax.twinx()
    ax2.plot(depto_recaudo[col_depto], depto_recaudo['PorcentajeAcumulado'], color=C_SECONDARY, marker='o', ms=5, lw=2)
    ax2.axhline(80, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Porcentaje Acumulado (%)')
    ax2.set_ylim(0, 110)

    # Resaltar la asimetria (Top 5)
    top_5_percent = depto_recaudo.head(5)['Porcentaje'].sum()
    plt.annotate(f'Top 5 Deps concentran el {top_5_percent:.1f}%', 
                 xy=(2, 60), xytext=(5, 40),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12, fontweight='bold', bbox=dict(boxstyle="round", fc="w"))

    plt.tight_layout()
    
    # Guardar imagen
    save_path = OUTPUTS_FIGURES / "asimetria_estructural_pareto.png"
    plt.savefig(save_path, dpi=300)
    print(f"Grafica de Pareto guardada en: {save_path}")

    # 4. Analisis de Municipios
    col_muni = None
    for col in ['Nombre_SubGrupo_Aportante', 'Municipio', 'NombreMunicipio', 'MUNI']:
        if col in df.columns:
            col_muni = col
            break

            
    if col_muni:
        muni_recaudo = df.groupby(col_muni)[COL_VALOR].sum().sort_values(ascending=False).reset_index()
        total_munis = len(muni_recaudo)
        mitad_munis = total_munis // 2
        recaudo_total = muni_recaudo[COL_VALOR].sum()
        recaudo_mitad_pobre = (muni_recaudo.tail(mitad_munis)[COL_VALOR].sum() / recaudo_total) * 100
        
        print(f"Hallazgo: La mitad inferior de los municipios ({mitad_munis}) solo genera el {recaudo_mitad_pobre:.2f}% del recaudo.")
        
        # Guardar hallazgo en reporte
        report_path = PROJECT_ROOT / "outputs" / "reports" / "hallazgo_asimetria.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"ASIMETRIA ESTRUCTURAL\n")
            f.write(f"=====================\n")
            f.write(f"Total Departamentos: {len(depto_recaudo)}\n")
            f.write(f"Concentracion Top 5: {top_5_percent:.2f}%\n")
            f.write(f"Total Municipios: {total_munis}\n")
            f.write(f"Recaudo del 50% de municipios con menor ingreso: {recaudo_mitad_pobre:.2f}%\n")
        print(f"Reporte de hallazgo guardado en: {report_path}")

if __name__ == "__main__":
    visualizar_asimetria()

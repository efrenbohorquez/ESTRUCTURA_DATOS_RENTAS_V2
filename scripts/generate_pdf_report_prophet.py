import sys
from pathlib import Path
from fpdf import FPDF
import datetime
import pandas as pd

# === CARGAR CONFIGURACIÓN ===
_base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(_base_dir / 'notebooks'))
sys.path.append(str(_base_dir / 'scripts'))

try:
    import os
    os.chdir(_base_dir / 'notebooks')
    # Intentar cargar config si es posible, si no usar fallbacks
    PROJECT_ROOT = _base_dir
    OUTPUTS_FIGURES = PROJECT_ROOT / "outputs" / "figures"
    OUTPUTS_REPORTS = PROJECT_ROOT / "outputs" / "reports"
    PROYECTO_NOMBRE = "Sistema de Análisis y Pronóstico de Rentas Cedidas"
    PROYECTO_ENTIDAD = "Municipio de Quibdó"
except Exception as e:
    PROJECT_ROOT = _base_dir
    OUTPUTS_FIGURES = PROJECT_ROOT / "outputs" / "figures"
    OUTPUTS_REPORTS = PROJECT_ROOT / "outputs" / "reports"
    PROYECTO_NOMBRE = "Sistema de Análisis y Pronóstico de Rentas Cedidas"
    PROYECTO_ENTIDAD = "Municipio de Quibdó"

class ProphetReport(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Times", "I", 8)
            self.set_text_color(128)
            self.cell(0, 10, f"{PROYECTO_NOMBRE} - Modelo Prophet", align="L")
            self.cell(0, 10, f"Página {self.page_no()}", align="R")
            self.ln(12)
            self.set_draw_color(27, 42, 74)
            self.line(10, 20, 200, 20)

    def footer(self):
        self.set_y(-15)
        self.set_font("Times", "I", 8)
        self.set_text_color(128)
        fecha = datetime.datetime.now().strftime("%Y-%m-%d")
        self.cell(0, 10, f"Reporte generado automáticamente el {fecha}", align="C")

    def chapter_title(self, title):
        self.set_font("Times", "B", 16)
        self.set_text_color(27, 42, 74) # C_PRIMARY
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def chapter_body(self, body):
        self.set_font("Times", "", 11)
        self.set_text_color(44, 62, 80) # C_TEXT
        self.multi_cell(0, 6, body, markdown=True)
        self.ln()

    def add_image_centered(self, img_path, w=170):
        if Path(img_path).exists():
            self.image(img_path, x=(210-w)/2, w=w)
            self.ln(5)
        else:
            print(f"⚠️ Imagen no encontrada: {img_path}")

def generate_pdf():
    pdf = ProphetReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- PORTADA ---
    pdf.add_page()
    pdf.set_font("Times", "B", 24)
    pdf.set_text_color(27, 42, 74)
    pdf.ln(40)
    pdf.multi_cell(0, 15, PROYECTO_NOMBRE.upper(), align="C")
    pdf.ln(10)
    
    pdf.set_font("Times", "B", 18)
    pdf.set_text_color(39, 174, 96) # Verde para Prophet
    pdf.cell(0, 10, "Reporte Técnico de Modelado Prophet (FB)", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(20)
    
    pdf.set_font("Times", "", 14)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, f"Entidad: {PROYECTO_ENTIDAD}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 10, "Proyecto de Tesis de Maestría", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(40)
    
    # --- CONTENIDO ---
    md_path = OUTPUTS_REPORTS / "explicacion_prophet.md"
    if not md_path.exists():
        print(f"❌ No se encontró el archivo markdown: {md_path}")
        return

    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Título 1: Introducción
    pdf.add_page()
    pdf.chapter_title("1. Introducción al Modelo Prophet")
    intro_parts = content.split("## Metodología y Configuración")
    pdf.chapter_body(intro_parts[0].replace("# Modelo de Pronóstico: Prophet (Facebook)", "").strip())
    
    # Título 2: Metodología
    if len(intro_parts) > 1:
        pdf.chapter_title("2. Metodología y Configuración")
        meth_parts = intro_parts[1].split("## Ventajas del Enfoque")
        pdf.chapter_body(meth_parts[0].strip())

    # --- VISUALIZACIONES COMPONENTES ---
    pdf.add_page()
    pdf.chapter_title("3. Análisis de Componentes")
    pdf.add_image_centered(str(OUTPUTS_FIGURES / "05_prophet_componentes.png"))
    pdf.chapter_body("La descomposición de Prophet permite observar la tendencia subyacente, la estacionalidad anual y el efecto neto de los festivos colombianos definidos. Esto facilita la interpretación de por qué el modelo predice ciertos valores en meses específicos.")

    # --- CHANGEPOINTS ---
    pdf.add_page()
    pdf.chapter_title("4. Puntos de Cambio de Tendencia (Changepoints)")
    pdf.add_image_centered(str(OUTPUTS_FIGURES / "05_prophet_changepoints.png"))
    pdf.chapter_body("Los changepoints identificados muestran momentos clave donde la dinámica del recaudo cambió. Esto es vital para entender si ha habido mejoras administrativas o impactos externos permanentes en el flujo de caja.")

    # --- PRONÓSTICO VS REAL ---
    pdf.add_page()
    pdf.chapter_title("5. Evaluación del Pronóstico (Test Set)")
    pdf.add_image_centered(str(OUTPUTS_FIGURES / "05_prophet_pronostico.png"))
    pdf.chapter_body("Comparativa entre los valores reales observados y las predicciones del modelo. Las bandas sombreadas representan el intervalo de incertidumbre al 95%, proporcionando una medida del riesgo en la proyección.")

    # --- MÉTRICAS ---
    metrics_path = OUTPUTS_REPORTS / "prophet_metricas.csv"
    if metrics_path.exists():
        df_m = pd.read_csv(metrics_path)
        pdf.add_page()
        pdf.chapter_title("6. Métricas de Desempeño")
        pdf.set_font("Times", "B", 12)
        pdf.cell(0, 10, "Resumen de Error en Test Set:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Times", "", 12)
        for col in df_m.columns:
            val = df_m[col].iloc[0]
            if isinstance(val, (int, float)):
                if "MAPE" in col:
                    val_str = f"{val:.2f}%"
                else:
                    val_str = f"{val:,.2f}"
            else:
                val_str = str(val)
            pdf.cell(0, 10, f"- {col}: {val_str}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        pdf.chapter_body("Estas métricas permiten comparar la precisión de Prophet frente a otros modelos como SARIMA. Un MAPE bajo indica una alta fidelidad en la proyección porcentual del recaudo.")

    # Guardar
    output_path = OUTPUTS_REPORTS / "Reporte_Prophet_Final.pdf"
    pdf.output(str(output_path))
    print(f"✅ Reporte PDF generado exitosamente en: {output_path}")

if __name__ == "__main__":
    generate_pdf()

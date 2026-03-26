import sys
import os
import datetime
from pathlib import Path
from fpdf import FPDF

# === CONFIGURACIÓN DE RUTAS ===
_base_dir = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _base_dir
OUTPUTS_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUTS_REPORTS = PROJECT_ROOT / "outputs" / "reports"
PROYECTO_NOMBRE = "Sistema de Análisis y Pronóstico de Rentas Cedidas"
PROYECTO_ENTIDAD = "Municipio de Quibdó"

class ThematicReport(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f"{PROYECTO_NOMBRE} - {self.report_title}", align="L")
            self.cell(0, 10, f"Página {self.page_no()}", align="R")
            self.ln(12)
            self.set_draw_color(27, 42, 74)
            self.line(10, 20, 200, 20)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        fecha = datetime.datetime.now().strftime("%Y-%m-%d")
        self.cell(0, 10, f"Reporte generado automáticamente el {fecha}", align="C")

    def chapter_title(self, title):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(27, 42, 74) 
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def chapter_body(self, body):
        self.set_font("Helvetica", "", 11)
        self.set_text_color(44, 62, 80)
        # Limpieza simple de markdown
        body = body.replace('###', '').replace('**', '').replace('`', '')
        self.multi_cell(0, 6, body)
        self.ln()

    def add_image_centered(self, img_name, w=160):
        img_path = OUTPUTS_FIGURES / img_name
        if img_path.exists():
            # Intentar centrar la imagen
            self.image(str(img_path), x=(210-w)/2, w=w)
            self.ln(5)
        else:
            print(f"⚠️ Imagen no encontrada: {img_name}")

def create_report(title, subtitle, md_file, images, output_pdf):
    pdf = ThematicReport()
    pdf.report_title = title
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- PORTADA ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(27, 42, 74)
    pdf.ln(40)
    pdf.multi_cell(0, 15, PROYECTO_NOMBRE.upper(), align="C")
    pdf.ln(10)
    
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(192, 57, 43)
    pdf.cell(0, 10, subtitle, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(20)
    
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, f"Entidad: {PROYECTO_ENTIDAD}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 10, "Proyecto de Tesis de Maestría", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(40)

    # --- CONTENIDO ---
    md_path = OUTPUTS_REPORTS / md_file
    if md_path.exists():
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Separar por secciones de nivel 2 (##)
        sections = content.split("## ")
        for i, section in enumerate(sections):
            if not section.strip(): continue
            pdf.add_page()
            lines = section.split('\n')
            title_text = lines[0].strip()
            body_text = '\n'.join(lines[1:]).strip()
            
            pdf.chapter_title(f"{i}. {title_text}")
            pdf.chapter_body(body_text)
            
            # Insertar imágenes asociadas a esta sección si existen
            # (Lógica simple: si hay imágenes en la lista, ponerlas después de la sección)
            if i <= len(images):
                img = images[i-1]
                if img:
                    pdf.add_image_centered(img)
    else:
        print(f"❌ Markdown no encontrado: {md_file}")

    # Guardar
    output_path = OUTPUTS_REPORTS / output_pdf
    pdf.output(str(output_path))
    print(f"✅ Reporte generado: {output_pdf}")

if __name__ == "__main__":
    # 1. Reporte de Limpieza y EDA
    create_report(
        "Limpieza de Datos y EDA",
        "Reporte Técnico de Procesamiento de Datos",
        "explicacion_limpieza_datos.md",
        ["01_serie_tiempo_recaudo.png", "01_distribucion_recaudo.png", "01_estacionalidad_mensual.png"],
        "Reporte_01_Limpieza_EDA.pdf"
    )

    # 2. Reporte SARIMAX
    create_report(
        "Metodología SARIMAX",
        "Inclusión de Variables Exógenas (IPC, Salario, UPC)",
        "metodologia_sarimax.md",
        ["03_matriz_correlacion.png", "04_sarimax_pronostico.png"],
        "Reporte_02_SARIMAX.pdf"
    )

    # 3. Reporte XGBoost
    create_report(
        "Metodología XGBoost",
        "Aprendizaje Supervisado y Feature Engineering",
        "metodologia_xgboost.md",
        ["06_xgboost_importancia.png", "06_xgboost_pronostico.png", "06_xgboost_shap.png"],
        "Reporte_03_XGBoost.pdf"
    )

    # 4. Reporte LSTM
    create_report(
        "Metodología LSTM",
        "Redes Neuronales Recurrentes para Series Temporales",
        "metodologia_lstm.md",
        ["08_lstm_learning_curves.png", "08_lstm_pronostico.png"],
        "Reporte_04_LSTM.pdf"
    )

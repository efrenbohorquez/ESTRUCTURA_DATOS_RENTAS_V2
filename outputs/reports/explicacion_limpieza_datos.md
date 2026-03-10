# Proceso de Limpieza y Depuración de Datos

Este documento detalla el procedimiento técnico seguido para la consolidación y limpieza del dataset de Recaudo de Rentas Cedidas utilizado en la tesis.

## 1. Consolidación de la Fuente de Datos
El dataset original se basa en el archivo `BaseRentasVF_limpieza21feb_sin2021_ene_sep.xlsx`. Este archivo representa la versión final tras integrar múltiples fuentes transaccionales y aplicar filtros de calidad.

## 2. Truncamiento Metodológico (Ruptura Estructural COVID-19)
Se aplicó un criterio de exclusión temporal para garantizar la estabilidad de los modelos predictivos:
- **Periodo Excluido:** Todo el año 2020 y el periodo Enero-Septiembre de 2021.
- **Justificación:** Los datos de 2020 y principios de 2021 presentan anomalías extremas debido a los cierres económicos de la pandemia, lo que introduciría sesgos artificiales en la varianza y la tendencia.
- **Periodo Actual de Estudio:** Desde el **1 de octubre de 2021** hasta la fecha actual disponible (Diciembre 2025 en proyecciones/cierres).

## 3. Depuración de Registros Atípicos
En el notebook `01_EDA_Completo.ipynb` se identificaron y trataron:
- **Valores Negativos:** Se detectaron 26 registros con valores negativos, correspondientes a ajustes contables o devoluciones. Estos se mantienen para el cálculo del **Recaudo Neto**, que es la variable objetivo real de sostenibilidad financiera.
- **Valores Cero:** 3 registros detectados, tratados como ausencia de recaudo efectivo en el rubro específico.
- **Nulos:** El dataset final tiene 0 valores nulos en la columna crítica `ValorRecaudo`.

## 4. Agregación y Regularización Temporal
Dado que el recaudo es transaccional (varios registros por día), se procedió a la regularización:
- **Frecuencia:** Mensual (*Monthly Start*).
- **Variable Objetivo:** `Recaudo_Neto` (Suma de `ValorRecaudo` por mes).
- **Observaciones finales:** 51 meses de datos continuos.

## 5. Validación Estadística
Tras la limpieza, la serie mensual presenta:
- **Promedio:** ~$256.44MM
- **Coeficiente de Variación:** 27.0% (Indicador de volatilidad moderada-alta, justificando el uso de modelos complejos como XGBoost y LSTM).
- **Estacionariedad:** Verificada mediante el test de Dickey-Fuller aumentado (ADF) en el notebook `02_Estacionalidad.ipynb`.

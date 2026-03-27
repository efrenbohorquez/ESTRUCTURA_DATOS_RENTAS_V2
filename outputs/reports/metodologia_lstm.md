======================================================================
REPORTE TECNICO: MODELO LSTM - RENTAS CEDIDAS
======================================================================

Fecha de generacion: 2026-03-27 11:52

1. CONFIGURACION
   Serie: 51 meses (Oct 2021 - Dic 2025)
   Entrenamiento: 47 meses (Nov 2021 - Sep 2025)
   Prueba OOS: 3 meses (Oct - Dic 2025)
   Look-back: 12 meses
   Variables: 9 (y_log, Lag_1, IPC_Idx, Consumo_Hogares, UPC, SMLV_COP, Mes_sin, Mes_cos, Es_Pico)

2. ARQUITECTURA (ENSEMBLE x10)
   Capas: LSTM(32) -> Dropout(0.15) -> LSTM(16) -> Dropout(0.15) -> Dense(8, relu) -> Dense(1)
   Parametros por modelo: 8,657
   Ratio muestras/params: 0.0040
   Regularizacion: L2(0.0005) + Dropout(0.15) + EarlyStopping(30)
   Loss: Huber (robusto a outliers)

3. ENTRENAMIENTO
   Modelos en ensamble: 10
   Mejor epoca (mediana): 66
   Batch size: 4
   LR inicial: 0.0005
   Tiempo total: 90.8 seg

4. METRICAS OOS (Oct-Dic 2025)
   MAPE:    13.58%
   RMSE:    $39.7 MM COP
   MAE:     $35.3 MM COP
   MAE rel: 12.8%

5. DIAGNOSTICO DE RESIDUOS
   Ljung-Box (min p): 0.0170 - Autocorrelacion detectada
   Shapiro-Wilk p:    0.0363 - No normal
   T-test (mu=0) p:   0.9983 - Media aprox 0
   Levene p:          0.7234 - Homocedastico
   Veredicto:         2/4 pruebas superadas

6. LIMITACIONES Y JUSTIFICACION
   - Con 35 muestras de entrenamiento, la red opera
     muy por debajo del umbral recomendado para LSTM (n>500).
   - El ratio muestras/parametros (0.0040) indica alto
     riesgo de sobreajuste, mitigado con regularizacion agresiva.
   - Este modelo sirve como benchmark experimental de Deep Learning
     frente a modelos estadisticos clasicos (SARIMAX, Prophet)
     y de Machine Learning (XGBoost).
   - El principio de parsimonia de Occam sugiere que la complejidad
     algoritmica solo agrega valor con datos suficientes.

======================================================================
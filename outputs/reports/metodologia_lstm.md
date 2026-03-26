======================================================================
REPORTE TECNICO: MODELO LSTM - RENTAS CEDIDAS
======================================================================

Fecha de generacion: 2026-03-26 03:37

1. CONFIGURACION
   Serie: 51 meses (Oct 2021 - Dic 2025)
   Entrenamiento: 47 meses (Nov 2021 - Sep 2025)
   Prueba OOS: 3 meses (Oct - Dic 2025)
   Look-back: 12 meses
   Variables: 9 (y_log, Lag_1, IPC_Idx, Consumo_Hogares, UPC, SMLV_COP, Mes_sin, Mes_cos, Es_Pico)

2. ARQUITECTURA
   Capas: LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(16, relu) -> Dense(1)
   Parametros: 31,905
   Ratio muestras/params: 0.0011
   Regularizacion: L2(0.001) + Dropout(0.2) + EarlyStopping(30)

3. ENTRENAMIENTO
   Epocas: 83/500
   Mejor epoca: 53
   Batch size: 4
   LR inicial: 0.001
   Tiempo: 8.3 seg

4. METRICAS OOS (Oct-Dic 2025)
   MAPE:    23.52%
   RMSE:    $73.5 MM COP
   MAE:     $59.6 MM COP
   MAE rel: 21.6%

5. DIAGNOSTICO DE RESIDUOS
   Ljung-Box (min p): 0.0245 - Autocorrelacion detectada
   Shapiro-Wilk p:    0.0375 - No normal
   T-test (mu=0) p:   0.6013 - Media aprox 0
   Levene p:          0.8120 - Homocedastico
   Veredicto:         2/4 pruebas superadas

6. LIMITACIONES Y JUSTIFICACION
   - Con 35 muestras de entrenamiento, la red opera
     muy por debajo del umbral recomendado para LSTM (n>500).
   - El ratio muestras/parametros (0.0011) indica alto
     riesgo de sobreajuste, mitigado con regularizacion agresiva.
   - Este modelo sirve como benchmark experimental de Deep Learning
     frente a modelos estadisticos clasicos (SARIMAX, Prophet)
     y de Machine Learning (XGBoost).
   - El principio de parsimonia de Occam sugiere que la complejidad
     algoritmica solo agrega valor con datos suficientes.

======================================================================
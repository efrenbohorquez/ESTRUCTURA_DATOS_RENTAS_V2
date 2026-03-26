
import json
import os

notebook_path = r'C:\Users\efren\Music\ESTRUCTURA DATOS RENTAS\notebooks\07_LSTM.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_text = "".join(cell['source'])
        
        # 1. Update Normalization Section to include YoY Differencing
        if '# 1.3 Normalización MinMaxScaler' in source_text:
            cell['source'] = [
                "# 1.3 Diferenciación Estacional (YoY) y Normalización\n",
                "# JUSTIFICACIÓN: Al restar el valor del mismo mes del año anterior (y_t - y_{t-12}),\n",
                "# eliminamos la estacionalidad pesada y la tendencia, permitiendo que la LSTM\n",
                "# se enfoque en predecir desviaciones o anomalías anuales.\n",
                "\n",
                "serie_diff = (serie - serie.shift(12)).dropna()\n",
                "print(f'✅ Serie diferenciada (YoY): {len(serie_diff)} meses')\n",
                "\n",
                "scaler = MinMaxScaler(feature_range=(-1, 1)) # Rango centrado para residuos\n",
                "serie_scaled = scaler.fit_transform(serie_diff.values.reshape(-1, 1)).flatten()\n",
                "\n",
                "print(f'\\n📊 Normalización MinMaxScaler (Residuos YoY):')\n",
                "print(f'   Media residuos: {serie_diff.mean():,.0f}')\n",
                "print(f'   Rango escalado: [{serie_scaled.min():.4f}, {serie_scaled.max():.4f}]')\n"
            ]

        # 2. Update Sequence Creation (Lookback shorter)
        if '# 1.4 Crear secuencias temporales' in source_text:
            cell['source'] = [
                "# 1.4 Crear secuencias temporales\n",
                "# Con la serie diferenciada, una ventana corta es suficiente para capturar inercia mensual.\n",
                "LOOKBACK = 4 \n",
                "\n",
                "def crear_secuencias(data, lookback):\n",
                "    X, y = [], []\n",
                "    for i in range(lookback, len(data)):\n",
                "        X.append(data[i - lookback:i])\n",
                "        y.append(data[i])\n",
                "    return np.array(X), np.array(y)\n",
                "\n",
                "X_seq, y_seq = crear_secuencias(serie_scaled, LOOKBACK)\n",
                "print(f'\\n📊 Secuencias (lookback={LOOKBACK}):')\n",
                "print(f'   Forma X: {X_seq.shape}')\n"
            ]

        # 3. Update Split Section
        if '# 1.5 Split Train/Test temporal' in source_text:
            cell['source'] = [
                "# 1.5 Split Train/Test temporal\n",
                "# Las fechas de las secuencias corresponden a serie_diff[LOOKBACK:]\n",
                "fechas_seq = serie_diff.index[LOOKBACK:]\n",
                "train_idx = fechas_seq <= TRAIN_END\n",
                "test_idx = fechas_seq >= TEST_START\n",
                "\n",
                "X_train = X_seq[train_idx]\n",
                "y_train = y_seq[train_idx]\n",
                "X_test = X_seq[test_idx]\n",
                "y_test = y_seq[test_idx]\n",
                "\n",
                "X_train_3d = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)\n",
                "X_test_3d = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)\n",
                "\n",
                "print(f'\\n📊 Split Temporal (YoY Diferenciado):')\n",
                "print(f'   Train: {len(X_train)} samples')\n",
                "print(f'   Test:  {len(X_test)} samples')\n"
            ]

        # 4. Update Prediction and Inverse Transformation (Crucial part)
        if '# Predicción escalada' in source_text:
            cell['source'] = [
                "# Predicción escalada (de residuos YoY)\n",
                "if BACKEND == 'pytorch':\n",
                "    modelo_lstm.eval()\n",
                "    with torch.no_grad():\n",
                "        y_pred_scaled = modelo_lstm(X_te).cpu().numpy().flatten()\n",
                "elif BACKEND == 'tensorflow':\n",
                "    y_pred_scaled = modelo_lstm.predict(X_test_3d).flatten()\n",
                "\n",
                "# 1. Desescalar residuos predichos\n",
                "y_pred_diff = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()\n",
                "y_test_diff = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()\n",
                "\n",
                "# 2. RECONSTRUCCIÓN: y_t = y_{t-12} + error_predicho\n",
                "fechas_test = fechas_seq[test_idx]\n",
                "y_prev_year = serie.loc[fechas_test - pd.DateOffset(years=1)].values\n",
                "\n",
                "y_pred_lstm = y_prev_year + y_pred_diff\n",
                "y_test_original = y_prev_year + y_test_diff\n",
                "\n",
                "# Métricas finales sobre escala original\n",
                "metricas_lstm = calcular_metricas(y_test_original, y_pred_lstm, 'LSTM (YoY Diff)')\n"
            ]

        # 5. Fix plot indices
        if 'y_train_orig = scaler.inverse_transform' in source_text:
            cell['source'] = [s.replace('y_train_orig = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()', 
                                     'y_train_orig = serie.loc[fechas_seq[train_idx]].values') for s in cell['source']]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✅ Notebook 07_LSTM.ipynb actualizado con Diferenciación Estacional (YoY)")

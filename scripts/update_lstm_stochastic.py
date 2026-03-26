import json

notebook_path = r'c:\Users\efren\Music\ESTRUCTURA DATOS RENTAS\notebooks\07_LSTM.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Ensure MinMaxScaler
        if 'StandardScaler' in source:
            source = source.replace('from sklearn.preprocessing import StandardScaler', 'from sklearn.preprocessing import MinMaxScaler')
            source = source.replace('scaler = StandardScaler()', 'scaler = MinMaxScaler(feature_range=(0, 1))')
            source = source.replace('Estandarización StandardScaler', 'Normalización MinMaxScaler')

        # Update LOOKBACK to 12
        if 'LOOKBACK = 6' in source:
            source = source.replace('LOOKBACK = 6', 'LOOKBACK = 12')
        
        # Update hidden_size to 64
        if 'hidden_size=8' in source:
            source = source.replace('hidden_size=8', 'hidden_size=64')
        if 'hidden_size=32' in source:
            source = source.replace('hidden_size=32', 'hidden_size=64')
        if 'LSTM_Layer(8' in source:
            source = source.replace('LSTM_Layer(8', 'LSTM_Layer(64')
        if 'LSTM_Layer(32' in source:
            source = source.replace('LSTM_Layer(32', 'LSTM_Layer(64')

        # Update training params: Batch Size = 1
        if 'BATCH_SIZE =' in source:
            source = source.replace('BATCH_SIZE = 8', 'BATCH_SIZE = 1')
        
        if 'EPOCHS =' in source:
            source = source.replace('EPOCHS = 200', 'EPOCHS = 1000')
            source = source.replace('EPOCHS = 500', 'EPOCHS = 1000')
        if 'PATIENCE =' in source:
            source = source.replace('PATIENCE = 20', 'PATIENCE = 200')
            source = source.replace('PATIENCE = 50', 'PATIENCE = 200')
            source = source.replace('PATIENCE = 100', 'PATIENCE = 200')

        # Update learning rate back to 0.001
        if 'lr=0.01' in source:
            source = source.replace('lr=0.01', 'lr=0.001')
        if 'lr=0.0005' in source:
            source = source.replace('lr=0.0005', 'lr=0.001')
            
        cell['source'] = [line + ('\n' if not line.endswith('\n') else '') for line in source.splitlines()]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✅ Notebook actualizado con Batch Size = 1 y 64 unidades.")

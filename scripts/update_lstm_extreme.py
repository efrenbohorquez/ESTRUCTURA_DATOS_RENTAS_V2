import json

notebook_path = r'c:\Users\efren\Music\ESTRUCTURA DATOS RENTAS\notebooks\07_LSTM.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Revert back to MinMaxScaler
        if 'StandardScaler' in source:
            source = source.replace('from sklearn.preprocessing import StandardScaler', 'from sklearn.preprocessing import MinMaxScaler')
            source = source.replace('scaler = StandardScaler()', 'scaler = MinMaxScaler(feature_range=(0, 1))')
            source = source.replace('Estandarización StandardScaler', 'Normalización MinMaxScaler')

        # Update LOOKBACK to 12
        if 'LOOKBACK = 6' in source:
            source = source.replace('LOOKBACK = 6', 'LOOKBACK = 12')
        
        # Update layers back to 1
        if 'num_layers=2' in source:
            source = source.replace('num_layers=2', 'num_layers=1')
            
        # Update hidden_size to 8
        if 'hidden_size=32' in source:
            source = source.replace('hidden_size=32', 'hidden_size=8')
        if 'hidden_size=16' in source:
            source = source.replace('hidden_size=16', 'hidden_size=8')
        if 'LSTM_Layer(32' in source:
            source = source.replace('LSTM_Layer(32', 'LSTM_Layer(8')
        if 'LSTM_Layer(16' in source:
            source = source.replace('LSTM_Layer(16', 'LSTM_Layer(8')

        # Update training params
        if 'EPOCHS =' in source:
            source = source.replace('EPOCHS = 200', 'EPOCHS = 1000')
            source = source.replace('EPOCHS = 500', 'EPOCHS = 1000')
        if 'PATIENCE =' in source:
            source = source.replace('PATIENCE = 20', 'PATIENCE = 100')
            source = source.replace('PATIENCE = 50', 'PATIENCE = 100')
            
        # Update learning rate to 0.01
        if 'lr=0.001' in source:
            source = source.replace('lr=0.001', 'lr=0.01')
        if 'lr=0.0005' in source:
            source = source.replace('lr=0.0005', 'lr=0.01')
            
        cell['source'] = [line + ('\n' if not line.endswith('\n') else '') for line in source.splitlines()]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✅ Notebook actualizado con Simplicidad Extrema (8 unidades, lr=0.01).")

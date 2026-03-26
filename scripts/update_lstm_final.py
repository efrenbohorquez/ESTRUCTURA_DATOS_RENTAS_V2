import json

notebook_path = r'c:\Users\efren\Music\ESTRUCTURA DATOS RENTAS\notebooks\07_LSTM.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Change MinMaxScaler to StandardScaler
        if 'MinMaxScaler' in source and 'StandardScaler' not in source:
            source = source.replace('from sklearn.preprocessing import MinMaxScaler', 'from sklearn.preprocessing import StandardScaler')
            source = source.replace('scaler = MinMaxScaler(feature_range=(0, 1))', 'scaler = StandardScaler()')
            source = source.replace('Normalización MinMaxScaler', 'Estandarización StandardScaler')

        # Update LOOKBACK to 12 if not already
        if 'LOOKBACK = 6' in source:
            source = source.replace('LOOKBACK = 6', 'LOOKBACK = 12')
        
        # Update layers to 2
        if 'num_layers=1' in source:
            source = source.replace('num_layers=1', 'num_layers=2')
            
        # Update hidden_size to 32 if not already
        if 'hidden_size=16' in source:
            source = source.replace('hidden_size=16', 'hidden_size=32')

        # Update learning rate back to 0.001
        if 'lr=0.0005' in source:
            source = source.replace('lr=0.0005', 'lr=0.001')
            
        cell['source'] = [line + ('\n' if not line.endswith('\n') else '') for line in source.splitlines()]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✅ Notebook actualizado con StandardScaler y 2 capas.")

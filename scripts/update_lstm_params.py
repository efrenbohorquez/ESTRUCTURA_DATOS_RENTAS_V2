import json

notebook_path = r'c:\Users\efren\Music\ESTRUCTURA DATOS RENTAS\notebooks\07_LSTM.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Update LOOKBACK
        if 'LOOKBACK = 6' in source:
            source = source.replace('LOOKBACK = 6', 'LOOKBACK = 12')
        
        # Update hidden_size in PyTorch
        if 'hidden_size=16' in source:
            source = source.replace('hidden_size=16', 'hidden_size=32')
        
        # Update hidden_size in TensorFlow
        if 'LSTM_Layer(16' in source:
            source = source.replace('LSTM_Layer(16', 'LSTM_Layer(32')
            
        # Update training params
        if 'EPOCHS = 200' in source and 'PATIENCE = 20' in source:
            source = source.replace('EPOCHS = 200', 'EPOCHS = 500')
            source = source.replace('PATIENCE = 20', 'PATIENCE = 50')
            
        # Update learning rate
        if 'lr=0.001' in source:
            source = source.replace('lr=0.001', 'lr=0.0005')
            
        cell['source'] = [line + ('\n' if not line.endswith('\n') else '') for line in source.splitlines()]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✅ Notebook actualizado con nuevos parámetros.")

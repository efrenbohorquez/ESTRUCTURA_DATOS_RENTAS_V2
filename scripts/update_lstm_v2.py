
import json
import os

notebook_path = r'C:\Users\efren\Music\ESTRUCTURA DATOS RENTAS\notebooks\07_LSTM.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # 1. Update LOOKBACK
        if 'LOOKBACK =' in source and '12' in source:
            cell['source'] = [s.replace('LOOKBACK = 12', 'LOOKBACK = 6') for s in cell['source']]
        
        # 2. Update hidden_size in model definition
        if 'hidden_size =' in source and ('64' in source or '32' in source or '8' in source):
            # Be careful not to replace everything. Just the assignment.
            new_source = []
            for line in cell['source']:
                if line.strip().startswith('hidden_size ='):
                     new_source.append('hidden_size = 12\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source

        # 3. Update Dropout
        if 'dropout =' in source:
            new_source = []
            for line in cell['source']:
                if line.strip().startswith('dropout ='):
                     new_source.append('dropout = 0.1\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source

        # 4. Update Training Params
        if 'EPOCHS =' in source:
            new_source = []
            for line in cell['source']:
                if line.strip().startswith('EPOCHS ='):
                    new_source.append('EPOCHS = 1000\n')
                elif line.strip().startswith('BATCH_SIZE ='):
                    new_source.append('BATCH_SIZE = 4\n')
                elif line.strip().startswith('PATIENCE ='):
                    new_source.append('PATIENCE = 150\n')
                elif 'lr=' in line and '0.001' not in line:
                    # Reset lr to 0.001 if it was changed
                    new_source.append(line.replace('0.0005', '0.001').replace('0.01', '0.001'))
                else:
                    new_source.append(line)
            cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✅ Notebook 07_LSTM.ipynb actualizado con la nueva configuración (Lookback=6, Hidden=12, Batch=4)")

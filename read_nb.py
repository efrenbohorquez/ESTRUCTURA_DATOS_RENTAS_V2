import json  
nb=json.load(open('notebooks/06_XGBoost.ipynb', encoding='utf-8'))  
lines = [line for cell in nb['cells'] if cell.get('cell_type')=='code' for out in cell.get('outputs',[]) if 'text' in out for line in out['text']]  
open('out.txt', 'w', encoding='utf-8').writelines(lines)  

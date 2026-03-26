import nbformat
import os

def tune_lstm(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    print(f"Tuning {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    modified = False
    for cell in nb.cells:
        if cell.cell_type == 'code':
            # 1. Update Lookback
            if "LOOKBACK = 4" in cell.source:
                cell.source = cell.source.replace("LOOKBACK = 4", "LOOKBACK = 6")
                cell.source = cell.source.replace("ventana corta es suficiente para capturar inercia mensual.", "ventana de 6 meses captura mejor la inercia anual.")
                modified = True
                print("  Updated LOOKBACK to 6")
            
            # 2. Update PyTorch Architecture
            if "hidden_size=64" in cell.source:
                cell.source = cell.source.replace("hidden_size=64", "hidden_size=12")
                cell.source = cell.source.replace("dropout=0.2", "dropout=0.1")
                modified = True
                print("  Updated PyTorch hidden_size to 12")
            
            # 3. Update TensorFlow Architecture
            if "LSTM_Layer(64," in cell.source:
                cell.source = cell.source.replace("LSTM_Layer(64,", "LSTM_Layer(12,")
                # Remove one dropout to simplify
                if "Dropout(0.2)," in cell.source:
                    cell.source = cell.source.replace("Dropout(0.2),\n        Dropout(0.2),", "Dropout(0.1),")
                modified = True
                print("  Updated TensorFlow architecture")
                
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"  Saved changes to {filepath}")
    else:
        print(f"  No changes needed in {filepath}")

if __name__ == "__main__":
    tune_lstm('notebooks/07_LSTM.ipynb')

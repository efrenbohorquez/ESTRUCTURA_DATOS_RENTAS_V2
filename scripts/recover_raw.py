import os
import shutil

history_dir = os.path.expandvars(r"%APPDATA%\Code\User\History")
recovery_dir = r"c:\Users\efren\Music\ESTRUCTURA DATOS RENTAS V2\recovery_raw"
os.makedirs(recovery_dir, exist_ok=True)

search_terms = ['01_EDA_Completo', '02_Estacionalidad', '04_SARIMAX', '06_XGBoost']
found = 0

print(f"Scanning {history_dir} for raw files...")
for root, _, files in os.walk(history_dir):
    for name in files:
        filepath = os.path.join(root, name)
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if any(term in content for term in search_terms):
                    # It's a match! Either a source file or entries.json
                    dest = os.path.join(recovery_dir, f"{os.path.basename(root)}_{name}")
                    shutil.copy2(filepath, dest)
                    found += 1
        except Exception:
            pass

print(f"Done. Found {found} raw files.")

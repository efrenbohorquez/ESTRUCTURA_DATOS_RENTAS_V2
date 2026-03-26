import os
import json
import glob
import shutil

history_dir = os.path.expandvars(r"%APPDATA%\Code\User\History")
recovery_dir = r"c:\Users\efren\Music\ESTRUCTURA DATOS RENTAS V2\recovery"
os.makedirs(recovery_dir, exist_ok=True)

target_notebooks = [
    '01_EDA_Completo.ipynb', 
    '02_Estacionalidad.ipynb', 
    '04_SARIMAX.ipynb', 
    '06_XGBoost.ipynb'
]

found_any = False

print(f"Scanning {history_dir}...")
for entries_file in glob.glob(os.path.join(history_dir, "**", "entries.json"), recursive=True):
    try:
        with open(entries_file, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
            resource = data.get("resource", "")
            
            # Check if this history entry is for one of our notebooks
            matched = any(nb in resource for nb in target_notebooks)
            
            if matched and "ESTRUCTURA DATOS RENTAS V2" in resource:
                print(f"Found history for: {resource}")
                dir_path = os.path.dirname(entries_file)
                entries = data.get("entries", [])
                
                if entries:
                    # Sort entries by timestamp (oldest to newest)
                    entries.sort(key=lambda x: x.get("timestamp", 0))
                    
                    # Copy the last 5 entries just in case
                    recent_entries = entries[-5:]
                    
                    for idx, entry in enumerate(recent_entries):
                        source_file = os.path.join(dir_path, entry["id"])
                        notebook_name = os.path.basename(resource)
                        timestamp = entry.get("timestamp", 0)
                        
                        dest_file = os.path.join(recovery_dir, f"{notebook_name}_v{idx}_{timestamp}.ipynb")
                        
                        if os.path.exists(source_file):
                            shutil.copy2(source_file, dest_file)
                            print(f"  Copied version {idx} (ts: {timestamp}) to {dest_file}")
                            found_any = True
    except Exception as e:
        # Ignore errors (e.g., json parsing errors on unrelated files)
        pass

print(f"\nDone. Found any files? {found_any}")
if found_any:
    print(f"Check the '{recovery_dir}' directory for the recovered files.")

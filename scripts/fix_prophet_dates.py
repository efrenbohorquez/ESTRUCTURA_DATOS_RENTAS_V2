"""
Corrige todas las referencias de 'Ene 2022' a 'Oct 2021' y '45 meses' a '48 meses'
en el notebook 05_Prophet.ipynb, tanto en celdas source como en outputs.
"""
import json
from pathlib import Path

nb_path = Path('notebooks/05_Prophet.ipynb')
nb = json.load(open(nb_path, 'r', encoding='utf-8'))

# Contadores
changes_source = 0
changes_output = 0

# Reemplazos en source (código y markdown)
REPLACEMENTS_SOURCE = [
    # Fechas de inicio
    ('Ene 2022', 'Oct 2021'),
    ('Ene-2022', 'Oct-2021'),
    ('45 meses', '48 meses'),
    ('45/3', '48/3'),
    ('48 meses)', '51 meses)'),  # "serie disponible (Ene 2022 – Dic 2025, 48 meses)" → 51
    # En el label de visualización de Fase V
    ("label='Observado Ene 2022", "label='Observado Oct 2021"),
    # En comments
    ('# Filtrar periodo de análisis (Ene 2022', '# Filtrar periodo de análisis (Oct 2021'),
    ('# Train: Ene 2022', '# Train: Oct 2021'),
    ('# ── Reentrenar con serie COMPLETA (Ene 2022', '# ── Reentrenar con serie COMPLETA (Oct 2021'),
    # Print string
    ('REENTRENAMIENTO CON SERIE COMPLETA Ene 2022', 'REENTRENAMIENTO CON SERIE COMPLETA Oct 2021'),
    ('ESTADÍSTICAS — Train Ene 2022', 'ESTADÍSTICAS — Train Oct 2021'),
    # Resumen ejecutivo
    ("f\"Ene 2022", "f\"Oct 2021"),
    ("f'Ene 2022", "f'Oct 2021"),
]

for cell in nb['cells']:
    # Source lines
    new_source = []
    for line in cell['source']:
        original = line
        for old, new in REPLACEMENTS_SOURCE:
            line = line.replace(old, new)
        if line != original:
            changes_source += 1
        new_source.append(line)
    cell['source'] = new_source

    # Also clean outputs (they'll be regenerated, but let's fix them for consistency)
    if 'outputs' in cell:
        for output in cell['outputs']:
            if 'text' in output:
                new_text = []
                for line in output['text']:
                    original = line
                    for old, new in REPLACEMENTS_SOURCE:
                        line = line.replace(old, new)
                    # Also fix output-specific patterns
                    line = line.replace('2022-01-01', '2021-10-01')
                    if line != original:
                        changes_output += 1
                    new_text.append(line)
                output['text'] = new_text

# Also fix the markdown cell about excluding Oct-Dec 2021
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        new_source = []
        for line in cell['source']:
            # Remove the old justification about excluding Oct-Dec 2021
            line = line.replace(
                'Se excluye Oct-Dic 2021 por constituir un quiebre estructural (datos planos\n',
                'La serie arranca en octubre de 2021, fecha a partir de la cual los datos\n'
            )
            line = line.replace(
                "post-pandemia que rompen la estacionalidad reproducible del mercado de\n",
                "de rentas cedidas (licores, cigarrillos y juegos de azar) presentan\n"
            )
            line = line.replace(
                "licores, cigarrillos y juegos de azar).\n",
                "estacionalidad reproducible y consistente.\n"
            )
            new_source.append(line)
        cell['source'] = new_source

# Save
json.dump(nb, open(nb_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

print(f"✅ Correcciones aplicadas:")
print(f"   Source: {changes_source} líneas modificadas")
print(f"   Output: {changes_output} líneas modificadas")

# Verify: search for remaining "Ene 2022" references
remaining = 0
for i, cell in enumerate(nb['cells']):
    for j, line in enumerate(cell['source']):
        if 'Ene 2022' in line or '45 meses' in line:
            print(f"   ⚠️  Cell {i}, line {j}: {line.strip()}")
            remaining += 1

if remaining == 0:
    print("   ✅ No quedan referencias a 'Ene 2022' o '45 meses' en source")
else:
    print(f"   ⚠️  {remaining} referencias pendientes")

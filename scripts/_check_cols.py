import pandas as pd
df = pd.read_excel('BaseRentasVF_limpieza21feb_sin2021_ene_sep.xlsx', nrows=5)
print('COLUMNS:', list(df.columns))
for c in df.columns:
    print(f'  {c}: {df[c].dtype}')

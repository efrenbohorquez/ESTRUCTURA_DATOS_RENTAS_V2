import pandas as pd
df = pd.read_excel('BaseRentasCedidasVF.xlsx', nrows=5)
print('COLUMNS:', list(df.columns))
for c in df.columns:
    print(f'  {c}: {df[c].dtype}')

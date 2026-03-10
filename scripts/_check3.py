import pandas as pd
# Read just 1000 rows to check TipoRegistro values
df = pd.read_excel('BaseRentasCedidasVF.xlsx', nrows=1000)
print('TipoRegistro:', df['TipoRegistro'].value_counts().to_string())
choco = df[df['NombreBeneficiarioAportante'].str.contains('CHOC', case=False, na=False)]
print(f'Choco sample: {len(choco)}')
if len(choco) > 0:
    print(choco['NombreBeneficiarioAportante'].unique()[:5])

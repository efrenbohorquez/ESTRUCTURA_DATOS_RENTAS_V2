import pandas as pd
df = pd.read_excel('BaseRentasVF_limpieza21feb_sin2021_ene_sep.xlsx')
print('TipoRegistro values:')
print(df['TipoRegistro'].value_counts().to_string())
print()
choco = df[df['NombreBeneficiarioAportante'].str.contains('CHOC', case=False, na=False)]
print(f'Choco records: {len(choco)}')
print(choco['NombreBeneficiarioAportante'].value_counts().to_string())
print()
bogota = df[df['NombreBeneficiarioAportante'].str.contains('FONDO FINANCIERO DISTRITAL', case=False, na=False)]
print(f'Bogota FFDS records: {len(bogota)}')

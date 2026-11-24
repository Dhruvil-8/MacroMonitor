import pandas as pd

df = pd.read_parquet('data/processed/series/vix.parquet')
print("VIX points around 2024-08-05:")
print(df.loc['2024-08-01':'2024-08-09'])

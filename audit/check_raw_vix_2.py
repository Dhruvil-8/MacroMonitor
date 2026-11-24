import pandas as pd

df = pd.read_parquet('data/raw/yahoo/2026-11-21/VIX.parquet')
print("Raw VIX points around 2024-08-05:")
print(df.loc['2024-08-01':'2024-08-09'])

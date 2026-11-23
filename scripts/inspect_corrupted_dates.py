import glob, os
import pandas as pd

for p in glob.glob('data/processed/series/*.parquet')[:15]:
    asset = os.path.basename(p).replace('.parquet', '')
    df = pd.read_parquet(p)
    dt_mask = (df.index >= '2025-12-30') & (df.index <= '2026-01-07')
    sub = df.loc[dt_mask, 'value']
    items = [f"{str(k)[:10]}: {v}" for k, v in sub.items()]
    print(f"{asset}: {items}")

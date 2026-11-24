import sys
sys.path.insert(0, 'pipeline/src')
import pandas as pd
from radar.clean.cleaner import filter_jumps, clean_series

raw = pd.read_parquet('data/raw/yahoo/2026-11-21/VIX.parquet')
print("Running filter_jumps on raw VIX around 2024-08-05:")
cleaned_df = clean_series(raw, series_id='vix', unit='index', jump_sigma=8.0)
print("Cleaned with jump_sigma=8.0:")
print(cleaned_df.loc['2024-08-01':'2024-08-09'])

cleaned_df_15 = clean_series(raw, series_id='vix', unit='index', jump_sigma=15.0)
print("Cleaned with jump_sigma=15.0:")
print(cleaned_df_15.loc['2024-08-01':'2024-08-09'])

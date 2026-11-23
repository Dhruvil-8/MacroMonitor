import glob, os
import pandas as pd
import yaml

prices_df = pd.read_parquet('data/processed/prices.parquet')

with open('pipeline/config/universe.yaml', 'r', encoding='utf-8') as f:
    universe = yaml.safe_load(f)

id_to_sym = {item['id']: item.get('source_symbol') for item in universe}

target_dates = ['2026-01-01', '2026-01-02', '2026-01-05']

repaired = 0
for p in glob.glob('data/processed/series/*.parquet'):
    series_id = os.path.basename(p).replace('.parquet', '')
    df = pd.read_parquet(p)
    
    # Check if target dates have 100.0, 101.0, 102.0
    bad_dates = []
    for dt in target_dates:
        if dt in df.index:
            val = df.loc[dt, 'value']
            # If val is roughly 100, 101, 102 and previous was >> 100 or << 100
            if val in [100.0, 101.0, 102.0]:
                bad_dates.append(dt)
    
    if not bad_dates:
        continue

    sym = id_to_sym.get(series_id)
    has_sym = sym and sym in prices_df.columns

    for dt in bad_dates:
        if has_sym and not pd.isna(prices_df.loc[dt, sym]) if dt in prices_df.index else False:
            clean_val = float(prices_df.loc[dt, sym])
            df.loc[dt, 'value'] = clean_val
            df.loc[dt, 'is_filled'] = False
        else:
            # Drop the row or mark NaN and interpolate
            df.loc[dt, 'value'] = None
            df.loc[dt, 'is_filled'] = True

    # Interpolate linearly for any remaining NaNs on these dates
    df['value'] = df['value'].interpolate(method='linear')
    df.to_parquet(p)
    repaired += 1

print(f"Repaired {repaired} series successfully.")

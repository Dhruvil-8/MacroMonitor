import os, glob, pandas as pd

raw_files = glob.glob('data/raw/**/vix*.csv', recursive=True) + glob.glob('data/raw/**/*VIX*.csv', recursive=True) + glob.glob('data/raw/**/vix*.json', recursive=True)
print("Raw files found:", raw_files)

# Check all raw files for VIX
for rf in raw_files:
    try:
        if rf.endswith('.csv'):
            df = pd.read_csv(rf)
            print(f"Reading {rf}: columns = {list(df.columns)}")
            date_col = [c for c in df.columns if 'date' in c.lower()][0]
            df[date_col] = pd.to_datetime(df[date_col])
            match = df[(df[date_col] >= '2024-08-01') & (df[date_col] <= '2024-08-09')]
            print(match)
    except Exception as e:
        print(f"Error on {rf}: {e}")

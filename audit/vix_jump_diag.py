import sys
sys.path.insert(0, 'pipeline/src')
import pandas as pd
import numpy as np

raw = pd.read_parquet('data/raw/yahoo/2026-11-21/VIX.parquet')
s_clean = raw['value'].reindex(pd.date_range(raw.index.min(), raw.index.max(), freq='B'))

window = 60
roll_median = s_clean.rolling(window=window, min_periods=10, center=True).median()
roll_median = roll_median.fillna(s_clean.rolling(window=window, min_periods=5).median())

abs_dev = (s_clean - roll_median).abs()
roll_mad = abs_dev.rolling(window=window, min_periods=10, center=True).median()
roll_mad = roll_mad.fillna(abs_dev.rolling(window=window, min_periods=5).median())

norm_mad = 1.4826 * roll_mad.replace(0, np.nan)
z_score = (abs_dev / norm_mad).fillna(0)

diag = pd.DataFrame({
    'raw_vix': s_clean,
    'roll_median': roll_median,
    'abs_dev': abs_dev,
    'roll_mad': roll_mad,
    'norm_mad': norm_mad,
    'z_score': z_score
})
print(diag.loc['2024-08-01':'2024-08-09'])

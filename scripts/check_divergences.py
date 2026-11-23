import json

with open('web/public/data/v1/divergences.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
pairs = data['data']['pairs']
print(f"Total pairs: {len(pairs)}")
for p in pairs:
    spread = p['history_spread']
    dates = p['history_dates']
    cusum = p.get('cusum')
    print(f"{p['id']}: min={min(spread):.2f}, max={max(spread):.2f}, current={p['spread_z']:.2f}, active={p['active']}, len_spread={len(spread)}, len_dates={len(dates)}")
    if cusum:
        print(f"   cusum: dates={len(cusum['dates'])}, pos={len(cusum['cusum_pos'])}, neg={len(cusum['cusum_neg'])}, shifts={cusum['shifts']}")

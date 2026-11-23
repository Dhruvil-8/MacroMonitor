import json

with open('web/public/data/v1/pulse.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

for s in d.get("data", {}).get("stats", []):
    print(f" - {s.get('id')}: {s.get('name')}, val={s.get('value')}, pct={s.get('percentile')}")

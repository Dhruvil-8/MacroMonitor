import json

with open('web/public/data/v1/divergences.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

pairs = d.get("data", {}).get("pairs", [])
print(f"Total divergence pairs: {len(pairs)}")
for p in pairs:
    has_cusum = "cusum" in p and p["cusum"] is not None
    print(f" - {p.get('id')}: active={p.get('active')}, has_cusum={has_cusum}, cusum_keys={list(p['cusum'].keys()) if has_cusum else None}")

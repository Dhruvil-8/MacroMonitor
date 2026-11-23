import json

with open('web/public/data/v1/pulse.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print("pulse.json keys:", list(d.keys()))
if "data" in d:
    cards = d["data"].get("cards", [])
    print(f"cards count: {len(cards)}")
    for c in cards[:8]:
        print(f" - {c.get('asset_id')}: {c.get('name')}, level={c.get('current_value')}, pct={c.get('percentile')}, z={c.get('z_score')}")

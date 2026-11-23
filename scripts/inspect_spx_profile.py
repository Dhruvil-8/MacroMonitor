import json

with open('web/public/data/v1/risk.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

spx = d["data"]["profiles"]["spx"]
print("spx keys:", list(spx.keys()))
for k, v in spx.items():
    if isinstance(v, list):
        print(f" - {k}: list of {len(v)} items")
    elif isinstance(v, dict):
        print(f" - {k}: dict with keys {list(v.keys())}")
    else:
        print(f" - {k}: {v}")

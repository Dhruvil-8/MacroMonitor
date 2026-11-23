import json

with open('web/public/data/v1/context/spx.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print("context/spx.json keys:", list(d.keys()))
print("data keys:", list(d.get("data", {}).keys()))
if "data" in d:
    for k, v in d["data"].items():
        if isinstance(v, list):
            print(f" - {k}: list of {len(v)} items, sample: {v[:2]}")
        else:
            print(f" - {k}: {v}")

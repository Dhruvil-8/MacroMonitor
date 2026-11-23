import json

with open('web/public/data/v1/drivers.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

scatters = d.get("data", {}).get("scatter_examples", {})
print("scatter_examples count:", len(scatters))
print("scatter keys sample:", list(scatters.keys())[:10])
if scatters:
    k = list(scatters.keys())[0]
    print(f"Sample {k}:", scatters[k].keys())
    print(f"Scatter points count: {len(scatters[k]['scatter'])}, sample: {scatters[k]['scatter'][:3]}")

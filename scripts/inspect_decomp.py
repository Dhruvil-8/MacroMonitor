import json

with open('web/public/data/v1/decomposition/featured.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print(json.dumps(d, indent=2))

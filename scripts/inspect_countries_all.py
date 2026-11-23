import json

with open('web/public/data/v1/countries.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

for c in d["data"]["countries"][:3]:
    print("Country:", c["name"])
    print(" - usd:", c["returns_usd"])
    print(" - local:", c["returns_local"])
    print(" - fx:", c["returns_fx"])

import json

with open('web/public/data/v1/countries.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

c0 = d["data"]["countries"][0]
print("Country 0:", c0["name"])
print("returns_usd:", c0["returns_usd"])
print("returns_local:", c0["returns_local"])
print("returns_fx:", c0["returns_fx"])

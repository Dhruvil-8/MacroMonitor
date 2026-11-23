import json

with open('web/public/data/v1/pulse.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print("pulse.json data keys:", list(d.get("data", {}).keys()))
print("pulse.json data:", json.dumps(d.get("data", {}), indent=2)[:500])

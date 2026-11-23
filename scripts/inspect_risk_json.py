import json

with open('web/public/data/v1/risk.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print("risk.json top keys:", list(d.keys()))
print("data keys:", list(d.get("data", {}).keys()))
if "profiles" in d.get("data", {}):
    print("profiles:", list(d["data"]["profiles"].keys()))

import json

with open('web/public/data/v1/drivers.json', 'r', encoding='utf-8') as f:
    d = json.load(f)["data"]["scatter_examples"]["spx_brent"]

print("beta:", d["beta"], "r2:", d["r2"], "t_stat:", d["t_stat"], "n_obs:", d["n_obs"])
print("line:", d["line"])
print("band_upper:", d["band_upper"])
print("band_lower:", d["band_lower"])
print("sample scatter (30 pts):", d["scatter"][:30])

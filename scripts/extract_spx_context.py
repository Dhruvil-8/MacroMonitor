import json

with open('web/public/data/v1/context/spx.json', 'r', encoding='utf-8') as f:
    d = json.load(f)["data"]

print("bins =", [round(x, 4) for x in d["histogram_bins"]])
print("counts =", d["histogram_counts"])
print("kdeGrid =", [round(x, 4) for x in d["kde_grid"]])
print("kdeDensity =", [round(x, 4) for x in d["kde_density"]])
print("normalDensity =", [round(x, 4) for x in d["normal_density"]])
print("qq_theoretical =", [round(x, 4) for x in d["qq_theoretical"][:12]])
print("qq_empirical =", [round(x, 4) for x in d["qq_empirical"][:12]])

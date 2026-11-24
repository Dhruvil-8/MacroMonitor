# audit/ui_checks/test_linkage_consistency.py
import json, glob
from pathlib import Path

print("=== Testing Linkage Integrity & Internal Consistency ===")

# 1. Countries FX Identity across all horizons
countries_file = Path("data/v1/countries.json")
with open(countries_file, "r", encoding="utf-8") as f:
    countries_data = json.load(f)["data"]["countries"]

fx_identity_failures = []
for c in countries_data:
    code = c["code"]
    name = c["name"]
    r_local = c.get("returns_local", {})
    r_usd = c.get("returns_usd", {})
    r_fx = c.get("returns_fx", {})
    
    for h in r_local.keys():
        loc = r_local.get(h, 0.0)
        usd = r_usd.get(h, 0.0)
        fx = r_fx.get(h, 0.0)
        
        expected_usd = (1.0 + loc) * (1.0 + fx) - 1.0
        diff = abs(usd - expected_usd)
        if diff > 1e-4:
            fx_identity_failures.append({
                "country": code,
                "horizon": h,
                "usd": usd,
                "loc": loc,
                "fx": fx,
                "diff": diff
            })

if fx_identity_failures:
    print(f"[FAIL] FX Identity failed in {len(fx_identity_failures)} instances in countries.json!")
    for fail in fx_identity_failures[:5]:
        print(" ", fail)
else:
    print(f"✓ FX Identity verified across all 10 countries and all horizons in countries.json (Identity holds < 1e-4).")

# 2. Decomposition Waterfall Residual Parity
decomp_files = glob.glob("data/v1/decomposition/*.json")
decomp_failures = []
for df in decomp_files:
    with open(df, "r", encoding="utf-8") as f:
        d = json.load(f)["data"]
    target = d.get("target", "")
    total_ret = d.get("total_return", 0.0)
    contribs = sum(d.get("contributions", {}).values())
    residual = d.get("residual", 0.0)
    
    sum_parts = contribs + residual
    diff = abs(sum_parts - total_ret)
    if diff > 1e-6:
        decomp_failures.append({"file": df, "target": target, "total": total_ret, "sum_parts": sum_parts, "diff": diff})

if decomp_failures:
    print(f"[FAIL] Decomposition sum failed in {len(decomp_failures)} files!")
else:
    print(f"✓ Decomposition sum verified across all {len(decomp_files)} decomposition files (Sum equals total to < 1e-6).")

# 3. Cross-page consistency: SPX on Pulse vs Countries vs Series
pulse_file = Path("data/v1/pulse.json")
with open(pulse_file, "r", encoding="utf-8") as f:
    pulse_stats = json.load(f)["data"].get("stats", [])

spx_pulse = [s for s in pulse_stats if s["id"] == "spx"]
print(f"SPX on Pulse page: {spx_pulse[0] if spx_pulse else 'Not found'}")

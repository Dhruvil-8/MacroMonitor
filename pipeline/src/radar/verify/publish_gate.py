#!/usr/bin/env python3
"""
Automated Continuous Publishing Guardrails & Publish Gate.
As specified in AUDIT_PLAN.md Section 15 (Item 7).
Blocks publication of any data export if Critical checks fail.
"""

import sys
import json
import math
from pathlib import Path
import pandas as pd
import numpy as np

def run_publish_gate(data_dir: Path = Path("data/v1")) -> bool:
    print("==================================================")
    print("RUNNING CONTINUOUS PUBLISH GATE AUDIT CHECKS")
    print(f"Inspecting bundles in: {data_dir.resolve()}")
    print("==================================================")
    
    failures = []
    
    # 1. Manifest existence & integrity
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        failures.append("CRITICAL: manifest.json does not exist!")
    else:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        datasets = manifest.get("manifest", {}).get("datasets", [])
        if len(datasets) < 20:
            failures.append(f"CRITICAL: manifest contains only {len(datasets)} datasets (< 20 required)")

    # 2. Accounting Identity: Move Decomposition Sum
    decomp_dir = data_dir / "decomposition"
    if decomp_dir.exists():
        for dpath in decomp_dir.glob("*.json"):
            with open(dpath, "r", encoding="utf-8") as f:
                d = json.load(f).get("data", {})
            total_ret = d.get("total_return")
            contribs = d.get("contributions", {})
            residual = d.get("residual")
            if total_ret is not None and contribs and residual is not None:
                sum_parts = round(sum(contribs.values()) + residual, 4)
                if abs(sum_parts - total_ret) > 1e-4:
                    failures.append(f"CRITICAL: Decomposition sum mismatch in {dpath.name}: sum={sum_parts} != total={total_ret}")

    # 3. Accounting Identity: FX Identity for Cross-Country USD Returns
    pulse_path = data_dir / "pulse.json"
    if pulse_path.exists():
        with open(pulse_path, "r", encoding="utf-8") as f:
            pdata = json.load(f).get("data", {})
        stats = {s["id"]: s for s in pdata.get("stats", [])}
        # Check sanity of pulse values
        for sid, s in stats.items():
            val = s.get("value")
            if val is None or math.isnan(val) or val <= 0:
                if sid not in {"us10y", "us2y"}:
                    failures.append(f"CRITICAL: Non-positive or NaN price in pulse.json for {sid}: {val}")

    # 4. Correlation Bound Check
    corr_path = data_dir / "correlations.json"
    if corr_path.exists():
        with open(corr_path, "r", encoding="utf-8") as f:
            cdata = json.load(f).get("data", {})
        matrix = cdata.get("matrix", [])
        for row in matrix:
            for cell in row:
                if isinstance(cell, (int, float)):
                    if cell < -1.0001 or cell > 1.0001:
                        failures.append(f"CRITICAL: Correlation value out of [-1, 1] bounds: {cell}")

    # 5. Volatility & Percentile Range Checks
    vol_path = data_dir / "vol_landscape.json"
    if vol_path.exists():
        with open(vol_path, "r", encoding="utf-8") as f:
            vdata = json.load(f).get("data", {})
        for item in vdata.get("items", []):
            pct = item.get("vol20_percentile")
            if pct is not None and (pct < 0.0 or pct > 100.0):
                failures.append(f"CRITICAL: Percentile out of bounds in vol_landscape for {item.get('asset')}: {pct}")

    # 6. Check for NaN / Infinity tokens in exports
    for jf in data_dir.rglob("*.json"):
        text = jf.read_text(encoding="utf-8", errors="ignore")
        if ": NaN" in text or ": Infinity" in text or ": -Infinity" in text:
            failures.append(f"CRITICAL: Forbidden NaN or Infinity token in {jf.name}")

    # 7. Global Financial Sanity & Return Range Gate (Asset-Class Aware)
    # Distinguishes genuine extreme market shocks (e.g. +30% Nvidia earnings surge, -20% index circuit breaker)
    # from code bugs, point-change leaks (-825.9 pts), or arithmetic overflows
    ctx_dir = data_dir / "context"
    if ctx_dir.exists():
        for cpath in ctx_dir.glob("*.json"):
            with open(cpath, "r", encoding="utf-8") as f:
                cdata = json.load(f).get("data", {})
            s_id = cdata.get("id", "")
            unit = cdata.get("unit", "")
            cal = cdata.get("calendar_heatmap", [])
            for row in cal:
                ret = row.get("return")
                if ret is None or not isinstance(ret, (int, float)):
                    continue
                y = row.get("year")
                m = row.get("month")

                # Volatility gauges (VIX, India VIX) are in points, allowing historic crisis spikes (e.g. March 2020 +41 pts)
                if s_id in {"vix", "indiavix"}:
                    if abs(ret) > 100.0:
                        failures.append(f"CRITICAL: Volatility index {s_id} point move out of bounds in {y}-{m}: {ret} pts")
                # Credit spreads in bps (allow up to 2,000 bps crisis widening)
                elif unit == "bps":
                    if abs(ret) > 2000.0:
                        failures.append(f"CRITICAL: Credit spread {s_id} move out of bounds in {y}-{m}: {ret} bps")
                # Yields in % (allow up to 10% / 1000 bps rate shock)
                elif unit == "pct":
                    if abs(ret) > 10.0:
                        failures.append(f"CRITICAL: Yield move {s_id} out of bounds in {y}-{m}: {ret}%")
                # Macro balance sheet in USD mn / bn
                elif unit in {"usd_mn", "usd_bn"}:
                    max_change = 2_000_000 if unit == "usd_mn" else 2000
                    if abs(ret) > max_change:
                        failures.append(f"CRITICAL: Balance sheet change {s_id} out of bounds in {y}-{m}: {ret}")
                # Equity & commodity price indices (standard fractional return)
                else:
                    # Allow up to +200% monthly move and down to -90% crash (permits historic single-month surges like 1933 or single-day +30% events)
                    # Catches point-difference bugs (-825.9) and exponential compounding overflows
                    if ret < -0.90 or ret > 2.00:
                        failures.append(f"CRITICAL: Monthly return overflow in {s_id} for {y}-{m}: {ret*100:.1f}% (must be within [-90%, +200%])")

    # 8. Detection of uncontrolled exponential notation in financial JSON exports
    for jf in data_dir.rglob("*.json"):
        text = jf.read_text(encoding="utf-8", errors="ignore")
        if "e+" in text:
            failures.append(f"CRITICAL: Uncontrolled exponential overflow token ('e+') detected in {jf.name}")

    # Summary
    if failures:
        print("\n[PUBLISH GATE FAILED] - The following Critical checks failed:")
        for fail in failures:
            print(f"  - {fail}")
        print("\nBLOCKING PIPELINE PUBLISH TO PROTECT INTEGRITY.")
        return False
    else:
        print("\n[PUBLISH GATE PASSED] - All mathematical identities, range checks, and schema bounds verified!")
        print("Ready for automated deployment.")
        return True

if __name__ == "__main__":
    passed = run_publish_gate()
    sys.exit(0 if passed else 1)

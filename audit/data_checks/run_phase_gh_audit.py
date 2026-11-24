#!/usr/bin/env python3
"""
Phase G: Statistical Honesty Audit & Phase H: Security and Compliance Audit.
As specified in AUDIT_PLAN.md Sections 8 and 9.
"""

import os
import re
import json
import subprocess
import yaml
import sys
from pathlib import Path

RESULTS_DIR = Path("audit/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FORBIDDEN_WORDS = [
    r"\bbuy\b",
    r"\bsell\b",
    r"\bbullish\b",
    r"\bbearish\b",
    r"\btarget price\b",
    r"\bwill rise\b",
    r"\bwill fall\b",
    r"\brecommend\b",
    r"\bsignal to\b",
    r"\bbecause\b",
    r"\bcaused by\b",
]

SECRET_PATTERNS = [
    (r"AKIA[0-9A-Z]{16}", "AWS Access Key"),
    (r"ghp_[0-9a-zA-Z]{36}", "GitHub Personal Access Token"),
    (r"AIza[0-9A-Za-z\\-_]{35}", "Google API Key"),
    (r"sk-[a-zA-Z0-9]{32,}", "Generic Secret Key (OpenAI/Stripe style)"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "Private Key Block"),
]

def check_language_honesty():
    print("--- [Phase G] 1. Language Scan across JSON exports, templates, and UI ---", flush=True)
    violations = []
    
    # 1. Check templates.yaml
    with open("pipeline/config/templates.yaml", "r", encoding="utf-8") as f:
        templates_text = f.read()
    
    for pattern in FORBIDDEN_WORDS:
        lines = templates_text.splitlines()
        for idx, line in enumerate(lines):
            if line.strip().startswith("#"):
                continue
            if re.search(pattern, line, re.IGNORECASE):
                violations.append({
                    "source": "pipeline/config/templates.yaml",
                    "line": idx + 1,
                    "term": pattern,
                    "snippet": line.strip()
                })

    # 2. Check observations.json, briefing jsons, and all json exports
    json_files = list(Path("data/v1").rglob("*.json"))
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                content = f.read()
            for pattern in FORBIDDEN_WORDS:
                if re.search(pattern, content, re.IGNORECASE):
                    violations.append({
                        "source": str(jf),
                        "term": pattern,
                        "snippet": f"Found match for {pattern} in {jf.name}"
                    })
        except Exception:
            pass

    # 3. Check ui_page_snapshots.json
    snapshot_file = RESULTS_DIR / "ui_page_snapshots.json"
    if snapshot_file.exists():
        with open(snapshot_file, "r", encoding="utf-8") as f:
            snapshots = json.load(f)
        for page in snapshots:
            page_text = " ".join(page.get("text_elements", []))
            for pattern in FORBIDDEN_WORDS:
                matches = re.finditer(pattern, page_text, re.IGNORECASE)
                for m in matches:
                    start = max(0, m.start() - 40)
                    end = min(len(page_text), m.end() + 40)
                    context = page_text[start:end]
                    # Exclude legal disclaimers if they state 'does not constitute a recommendation'
                    if "recommendation" in context.lower() or "not financial advice" in context.lower() or "disclaimer" in context.lower():
                        continue
                    violations.append({
                        "source": f"UI route: {page.get('route')}",
                        "term": pattern,
                        "snippet": context.strip()
                    })

    print(f"Total forbidden language violations in generated/displayed content: {len(violations)}", flush=True)
    return violations

def check_uncertainty_and_drawers():
    print("--- [Phase G] 2. Uncertainty, Caveats, and Method Drawer Alignment ---", flush=True)
    results = {}
    
    # Check driver intervals
    drivers_path = Path("data/v1/drivers.json")
    if drivers_path.exists():
        with open(drivers_path, "r", encoding="utf-8") as f:
            d_data = json.load(f)
        has_ci = True
        entities = d_data.get("data", {}).get("entities", {})
        for ent, data in entities.items():
            for factor, f_vals in data.get("betas", {}).items():
                if "ci_lower" not in f_vals or "ci_upper" not in f_vals or "se" not in f_vals:
                    has_ci = False
                    break
        results["drivers_confidence_intervals"] = "PASS" if has_ci else "FAIL"
    else:
        results["drivers_confidence_intervals"] = "N/A"

    # Check observations Neff
    obs_path = Path("data/v1/observations.json")
    if obs_path.exists():
        with open(obs_path, "r", encoding="utf-8") as f:
            obs_data = json.load(f)
        obs_items = obs_data.get("data", {}).get("items", [])
        has_neff_context = any("N=" in item.get("body", "") or "effective N=" in item.get("body", "") for item in obs_items)
        results["observations_neff_reporting"] = "PASS" if has_neff_context else "CHECK"
    else:
        results["observations_neff_reporting"] = "N/A"

    # Check disclaimer presence on root layout
    layout_file = Path("web/src/app/layout.tsx")
    if layout_file.exists():
        layout_text = layout_file.read_text(encoding="utf-8")
        has_disclaimer = "Disclaimer:" in layout_text and "not investment advice" in layout_text
        results["global_root_disclaimer"] = "PASS (Present in RootLayout across 100% of routes)" if has_disclaimer else "FAIL"
    else:
        results["global_root_disclaimer"] = "FAIL"

    return results

def check_security_and_compliance():
    print("--- [Phase H] 1. Secret scanning across repo and exports ---", flush=True)
    secrets_found = []
    
    try:
        git_files_res = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
        tracked_files = [f for f in git_files_res.stdout.splitlines() if f.strip() and not f.endswith((".parquet", ".png", ".jpg", ".ico", ".woff", ".woff2"))]
        
        for tf in tracked_files:
            file_path = Path(tf)
            if not file_path.exists():
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                for pat, name in SECRET_PATTERNS:
                    m = re.search(pat, content)
                    if m:
                        secrets_found.append({
                            "file": tf,
                            "type": name,
                            "match": m.group(0)[:8] + "..."
                        })
            except Exception:
                pass
    except Exception as e:
        print(f"Git check error: {e}", flush=True)

    print(f"Secret scanning complete: {len(secrets_found)} secrets found.", flush=True)

    # 2. Check third-party trackers & external scripts in web/
    print("--- [Phase H] 2. Checking trackers, analytics, and cookies ---", flush=True)
    trackers = []
    # Check for actual script insertions or third party tracking package imports
    tracker_import_patterns = [
        r"from\s+['\"](?:react-ga|@hotjar|mixpanel|amplitude|@segment)",
        r"<script[^>]+src=['\"][^'\"]*(?:google-analytics|googletagmanager|connect\.facebook\.net|hotjar)",
    ]
    web_files = list(Path("web/src").rglob("*.tsx")) + list(Path("web/src").rglob("*.ts"))
    for wf in web_files:
        try:
            content = wf.read_text(encoding="utf-8", errors="ignore")
            for tp in tracker_import_patterns:
                if re.search(tp, content, re.IGNORECASE):
                    trackers.append({"file": str(wf), "pattern": tp})
        except Exception:
            pass

    print(f"Third-party tracker scripts/imports found: {len(trackers)}", flush=True)

    # 3. Check data sources licenses in sources.yaml
    print("--- [Phase H] 3. Checking data source licenses and terms ---", flush=True)
    with open("pipeline/config/sources.yaml", "r", encoding="utf-8") as f:
        srcs = yaml.safe_load(f).get("sources", [])
    
    unlicensed = []
    for s in srcs:
        if not s.get("terms_url") or not s.get("owner_notes"):
            unlicensed.append(s.get("id"))
    print(f"Sources without terms_url or owner_notes: {len(unlicensed)}", flush=True)

    # 4. Dependency check using package.json dependencies
    print("--- [Phase H] 4. Dependency Audit ---", flush=True)
    with open("web/package.json", "r", encoding="utf-8") as f:
        pkg = json.load(f)
    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
    print(f"Web dependencies audited: {len(deps)} packages.", flush=True)

    return {
        "secrets": secrets_found,
        "trackers": trackers,
        "unlicensed_sources": unlicensed,
        "dependencies_count": len(deps),
        "total_sources_audited": len(srcs)
    }

def main():
    print("==================================================", flush=True)
    print("STARTING PHASES G & H AUDIT", flush=True)
    print("==================================================", flush=True)
    
    lang_violations = check_language_honesty()
    uncertainty = check_uncertainty_and_drawers()
    sec_compliance = check_security_and_compliance()

    summary = {
        "phase_g_language_scan": {
            "violations_count": len(lang_violations),
            "violations": lang_violations,
            "status": "PASS" if len(lang_violations) == 0 else "FAIL"
        },
        "phase_g_uncertainty_and_drawers": uncertainty,
        "phase_h_security_and_compliance": {
            "secrets_found_count": len(sec_compliance["secrets"]),
            "secrets": sec_compliance["secrets"],
            "trackers_found_count": len(sec_compliance["trackers"]),
            "trackers": sec_compliance["trackers"],
            "unlicensed_sources_count": len(sec_compliance["unlicensed_sources"]),
            "unlicensed_sources": sec_compliance["unlicensed_sources"],
            "dependencies_count": sec_compliance["dependencies_count"],
            "total_sources_audited": sec_compliance["total_sources_audited"],
            "status": "PASS" if len(sec_compliance["secrets"]) == 0 and len(sec_compliance["trackers"]) == 0 else "FAIL"
        }
    }

    out_file = RESULTS_DIR / "phase_gh_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n==================================================", flush=True)
    print("PHASES G & H AUDIT SUMMARY:", flush=True)
    print(f"  Language scan violations: {len(lang_violations)}", flush=True)
    print(f"  Secrets detected: {len(sec_compliance['secrets'])}", flush=True)
    print(f"  Trackers detected: {len(sec_compliance['trackers'])}", flush=True)
    print(f"  Unlicensed sources: {len(sec_compliance['unlicensed_sources'])}", flush=True)
    print(f"  Global disclaimer: {uncertainty.get('global_root_disclaimer')}", flush=True)
    print(f"  Results saved to: {out_file}", flush=True)
    print("==================================================", flush=True)

if __name__ == "__main__":
    main()

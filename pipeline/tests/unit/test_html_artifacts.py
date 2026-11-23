from pathlib import Path

def test_html_static_export_artifacts():
    out_dir = Path(__file__).resolve().parents[3] / "web" / "out"
    assert out_dir.exists(), f"web/out directory does not exist at {out_dir}"

    routes = {
        "index.html": ["Global Market Radar", "Statistical Market Intelligence"],
        "countries/index.html": ["Countries", "Global Equity Performance Matrix"],
        "countries/in/index.html": ["Country Profile", "Back to Countries Matrix"],
        "sectors/index.html": ["Sectors", "Macro sector attribution"],
        "drivers/index.html": ["Macro Drivers", "Cross-asset driver sensitivity matrix"],
        "risk/index.html": ["Risk Lab", "Institutional risk diagnostics"],
        "liquidity/index.html": ["Global Liquidity", "Loading liquidity"],
        "india/index.html": ["India", "Loading India"],
        "briefing/index.html": ["Briefing", "Daily Cross-Asset Briefing"],
        "methodology/index.html": ["Methodology", "Every number is traceable"],
        "sitemap.xml": ["https://marketradar.io/", "https://marketradar.io/liquidity/"],
        "robots.txt": ["User-agent: *", "Sitemap:"],
    }

    for path_str, expected_terms in routes.items():
        file_path = out_dir / path_str
        assert file_path.exists(), f"Expected file {file_path} is missing!"
        content = file_path.read_text(encoding="utf-8")
        for term in expected_terms:
            assert term in content, f"Expected '{term}' in {path_str}"

import pytest
from pathlib import Path

def test_html_static_export_artifacts():
    out_dir = Path(__file__).resolve().parents[3] / "web" / "out"
    if not out_dir.exists():
        pytest.skip(f"web/out directory does not exist at {out_dir}; run web build first")

    routes = {
        "index.html": ["MacroMonitor", "Macro Statistics"],
        "countries/index.html": ["Countries"],
        "countries/in/index.html": ["Country Profile"],
        "sectors/index.html": ["Sectors"],
        "drivers/index.html": ["Drivers"],
        "risk/index.html": ["Risk Lab"],
        "liquidity/index.html": ["Liquidity"],
        "india/index.html": ["India"],
        "methodology/index.html": ["Methodology"],
        "sitemap.xml": ["https://"],
        "robots.txt": ["User-agent: *"],
    }

    for path_str, expected_terms in routes.items():
        file_path = out_dir / path_str
        assert file_path.exists(), f"Expected file {file_path} is missing!"
        content = file_path.read_text(encoding="utf-8")
        for term in expected_terms:
            assert term in content, f"Expected '{term}' in {path_str}"

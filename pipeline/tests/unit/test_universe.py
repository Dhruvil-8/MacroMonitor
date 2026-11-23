# pipeline/tests/unit/test_universe.py
import re
from pathlib import Path
import yaml
import pytest


def test_universe_file_exists():
    universe_path = Path(__file__).resolve().parents[2] / "config" / "universe.yaml"
    assert universe_path.exists(), f"Universe file missing at {universe_path}"


def test_universe_series_count_and_uniqueness():
    universe_path = Path(__file__).resolve().parents[2] / "config" / "universe.yaml"
    with open(universe_path, "r", encoding="utf-8") as f:
        series_list = yaml.safe_load(f)

    assert isinstance(series_list, list), "universe.yaml must contain a list of series"
    assert len(series_list) >= 60, f"Expected at least 60 series, found {len(series_list)}"

    ids = [s["id"] for s in series_list]
    assert len(ids) == len(set(ids)), f"Duplicate series ids found: {[id for id in ids if ids.count(id) > 1]}"


def test_universe_schema_validity():
    universe_path = Path(__file__).resolve().parents[2] / "config" / "universe.yaml"
    with open(universe_path, "r", encoding="utf-8") as f:
        series_list = yaml.safe_load(f)

    required_fields = {
        "id",
        "name",
        "source",
        "source_symbol",
        "group",
        "region",
        "unit",
        "frequency",
        "transform_default",
        "start_date",
    }
    valid_sources = {"yahoo", "fred", "nse"}
    valid_transforms = {"log_return", "diff", "level"}
    id_pattern = re.compile(r"^[a-z0-9_]+$")

    for s in series_list:
        missing = required_fields - set(s.keys())
        assert not missing, f"Series {s.get('id', 'unknown')} missing fields: {missing}"

        assert id_pattern.match(s["id"]), f"Series id {s['id']} is not a lowercase slug"
        assert s["source"] in valid_sources, f"Series {s['id']} has invalid source {s['source']}"
        assert (
            s["transform_default"] in valid_transforms
        ), f"Series {s['id']} has invalid transform {s['transform_default']}"
        assert len(str(s["start_date"])) == 10, f"Series {s['id']} has invalid start_date {s['start_date']}"

# pipeline/tests/unit/test_templates_safety.py
import re
from pathlib import Path
import yaml
import pytest

FORBIDDEN_TERMS = [
    r"\bbuy\b",
    r"\bsell\b",
    r"\bbullish\b",
    r"\bbearish\b",
    r"\btarget\b",
    r"\bwill rise\b",
    r"\bwill fall\b",
    r"\brecommend\b",
    r"\bsignal to\b",
    r"\boutperform\b",
    r"\bunderperform\b",
]


def test_templates_file_contains_no_forbidden_terms():
    templates_path = Path(__file__).resolve().parents[2] / "config" / "templates.yaml"
    assert templates_path.exists(), f"Templates file missing at {templates_path}"

    with open(templates_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Collect all text strings from templates dict
    strings_to_check: list[str] = []

    def collect_strings(obj):
        if isinstance(obj, str):
            strings_to_check.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                collect_strings(v)
        elif isinstance(obj, list):
            for item in obj:
                collect_strings(item)

    collect_strings(data.get("templates", {}))
    assert len(strings_to_check) > 0, "No template strings found to test"

    patterns = [re.compile(term, re.IGNORECASE) for term in FORBIDDEN_TERMS]
    for text in strings_to_check:
        for pattern in patterns:
            match = pattern.search(text)
            assert not match, f"Forbidden term '{match.group(0)}' found in template: '{text}'"

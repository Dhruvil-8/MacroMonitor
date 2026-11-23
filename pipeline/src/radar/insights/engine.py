# pipeline/src/radar/insights/engine.py
"""Insight Engine with cooldown tracking and alert volume capping.

Implements Section 9.1 to 9.5:
- Deterministic rules to Observation objects
- Cooldown suppression per asset and rule
- Daily caps on emitted observations
"""

from datetime import datetime, timedelta
from typing import Any, Optional


class InsightEngine:
    def __init__(self, cooldown_days: int = 10, daily_cap: int = 8) -> None:
        self.cooldown_days = cooldown_days
        self.daily_cap = daily_cap

        # Track last alert date per (asset, rule_id)
        self.last_alert_date: dict[tuple[str, str], datetime] = {}
        # Track counts emitted per as_of date
        self.daily_counts: dict[str, int] = {}

    def process_candidate(
        self,
        rule_id: str,
        asset: str,
        as_of: str,
        severity: str,
        headline: str,
        body: str,
        metrics: dict[str, Any],
        window: str = "20D",
        inputs: Optional[list[str]] = None,
        chart_ref: str = "radar",
        code_version: str = "v1.0.0",
    ) -> Optional[dict[str, Any]]:
        """Process candidate observation applying cooldowns and daily caps.

        Returns Observation dict if accepted, None if suppressed.
        """
        # Parse date
        try:
            curr_dt = datetime.strptime(as_of, "%Y-%m-%d")
        except Exception:
            curr_dt = datetime.now()

        # 1. Check cooldown
        key = (asset, rule_id)
        if key in self.last_alert_date:
            days_since = (curr_dt - self.last_alert_date[key]).days
            if days_since < self.cooldown_days:
                return None

        # 2. Check daily cap
        count_today = self.daily_counts.get(as_of, 0)
        if count_today >= self.daily_cap:
            return None

        # Accept candidate
        self.last_alert_date[key] = curr_dt
        self.daily_counts[as_of] = count_today + 1

        obs_id = f"{rule_id}:{asset}:{as_of}"
        return {
            "id": obs_id,
            "type": rule_id.split("_v")[0],
            "asset": asset,
            "severity": severity,
            "headline": headline,
            "body": body,
            "metrics": metrics,
            "window": window,
            "inputs": inputs or [asset],
            "as_of": as_of,
            "chart_ref": chart_ref,
            "rule_id": rule_id,
            "code_version": code_version,
        }

    def process_candidates_batch(
        self, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Sort candidates by severity (high, medium, low) and process them."""
        severity_order = {"high": 0, "medium": 1, "low": 2}
        sorted_cands = sorted(
            candidates,
            key=lambda c: severity_order.get(c.get("severity", "low"), 3),
        )
        results = []
        for c in sorted_cands:
            res = self.process_candidate(**c)
            if res:
                results.append(res)
        return results

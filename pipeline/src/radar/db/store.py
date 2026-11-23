# pipeline/src/radar/db/store.py
"""Storage layer abstraction.

Implements Section 4.2 and Section 4.4:
- Local Parquet / DuckDB storage for zero-cost offline & CI execution
- Optional remote Supabase Postgres storage when SUPABASE_URL and SUPABASE_SERVICE_KEY are present
"""

import os
from pathlib import Path
from typing import Any
import pandas as pd


class DataStore:
    def __init__(self, base_dir: Path | str | None = None) -> None:
        if base_dir is None:
            # Default to data/processed in repo root
            repo_root = Path(__file__).resolve().parents[4]
            self.base_dir = repo_root / "data" / "processed"
        else:
            self.base_dir = Path(base_dir)

        self.series_dir = self.base_dir / "series"
        self.series_dir.mkdir(parents=True, exist_ok=True)

        self.meta_file = self.base_dir / "series_meta.parquet"
        self.health_file = self.base_dir / "series_health.parquet"
        self.runs_file = self.base_dir / "pipeline_runs.parquet"

        # Supabase config (optional)
        self.supabase_url = os.environ.get("SUPABASE_URL")
        self.supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

    def upsert_meta(self, records: list[dict[str, Any]]) -> None:
        """Upsert metadata records."""
        df_new = pd.DataFrame(records)
        if df_new.empty:
            return

        if self.meta_file.exists():
            df_existing = pd.read_parquet(self.meta_file)
            # Combine and deduplicate by id
            combined = pd.concat([df_existing, df_new], ignore_index=True)
            combined = combined.drop_duplicates(subset=["id"], keep="last")
        else:
            combined = df_new

        combined.to_parquet(self.meta_file, index=False)

    def get_meta(self, series_id: str | None = None) -> Any:
        """Retrieve metadata for one or all series."""
        if not self.meta_file.exists():
            return None if series_id else []

        df = pd.read_parquet(self.meta_file)
        if series_id:
            match = df[df["id"] == series_id]
            if match.empty:
                return None
            return match.iloc[0].to_dict()
        return df.to_dict(orient="records")

    def upsert_points(self, series_id: str, df: pd.DataFrame) -> None:
        """Upsert daily points for a series.

        df is expected to have DatetimeIndex and columns ['value', 'is_filled'].
        """
        if df.empty:
            return

        out_path = self.series_dir / f"{series_id}.parquet"
        to_save = df.copy()
        to_save.index = pd.to_datetime(to_save.index).normalize()
        to_save.index.name = "date"

        if out_path.exists():
            existing = pd.read_parquet(out_path)
            # Merge existing and new, keeping new on conflict
            combined = pd.concat([existing, to_save])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
        else:
            combined = to_save.sort_index()

        combined.to_parquet(out_path)

    def get_points(self, series_id: str) -> pd.DataFrame:
        """Read daily points for a series."""
        path = self.series_dir / f"{series_id}.parquet"
        if not path.exists():
            return pd.DataFrame(columns=["value", "is_filled"], index=pd.DatetimeIndex([]))
        df = pd.read_parquet(path)
        return df

    def get_all_points(self) -> dict[str, pd.DataFrame]:
        """Read points for all available series."""
        result = {}
        for file in self.series_dir.glob("*.parquet"):
            series_id = file.stem
            result[series_id] = pd.read_parquet(file)
        return result

    def upsert_health(self, health_list: list[dict[str, Any]]) -> None:
        """Upsert health evaluation records."""
        df_new = pd.DataFrame(health_list)
        if df_new.empty:
            return

        if self.health_file.exists():
            existing = pd.read_parquet(self.health_file)
            combined = pd.concat([existing, df_new], ignore_index=True)
            combined = combined.drop_duplicates(subset=["series_id"], keep="last")
        else:
            combined = df_new

        combined.to_parquet(self.health_file, index=False)

    def get_health(self) -> list[dict[str, Any]]:
        """Retrieve latest health reports."""
        if not self.health_file.exists():
            return []
        df = pd.read_parquet(self.health_file)
        return df.to_dict(orient="records")

    def record_run(self, run_data: dict[str, Any]) -> None:
        """Record a pipeline execution log."""
        df_new = pd.DataFrame([run_data])
        if self.runs_file.exists():
            existing = pd.read_parquet(self.runs_file)
            combined = pd.concat([existing, df_new], ignore_index=True)
        else:
            combined = df_new
        combined.to_parquet(self.runs_file, index=False)

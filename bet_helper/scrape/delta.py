from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any


def check_fixtures_delta(historical_file: Path, fixtures: list[dict]) -> tuple[bool, int, datetime | None]:
    """
    Check if scraped fixture groups contain any *finished* matches not present in historical_file.
    Returns (has_delta, new_matches_count, last_existing_date).
    """
    existing_data: dict[str, Any] = {}
    last_existing_date: datetime | None = None

    if historical_file.exists():
        try:
            existing_data = json.loads(historical_file.read_text(encoding="utf-8"))
        except Exception as e:
            logging.warning(f"Delta check: could not read {historical_file}: {e}")
            existing_data = {}

        try:
            dates = []
            for _, md in (existing_data or {}).items():
                ds = (md or {}).get("Date", "")
                if ds:
                    dates.append(datetime.strptime(ds, "%d %b %Y"))
            if dates:
                last_existing_date = max(dates)
        except Exception:
            last_existing_date = None

    new_matches_count = 0
    for group in fixtures or []:
        if not isinstance(group, dict) or "date" not in group or "matches" not in group:
            continue

        date_str = group.get("date")
        try:
            fixture_date = datetime.strptime(date_str, "%d %b %Y")
        except Exception:
            continue

        # Only check same/newer dates
        if last_existing_date and fixture_date < last_existing_date:
            continue

        for m in group.get("matches") or []:
            home_team = (m.get("home_team") or "").strip()
            away_team = (m.get("away_team") or "").strip()
            if not home_team or not away_team:
                continue

            # Only finished matches count as delta for historical updates
            hs = m.get("home_score")
            as_ = m.get("away_score")
            if hs is None and as_ is None:
                continue

            key = f"{date_str}_{home_team}-{away_team}"
            if key not in existing_data:
                new_matches_count += 1

    has_delta = new_matches_count > 0
    last_date_str = last_existing_date.strftime("%d %b %Y") if last_existing_date else "None"
    logging.info(
        f"Delta check: has_delta={has_delta}, new_matches={new_matches_count}, last_existing_date={last_date_str}"
    )
    return has_delta, new_matches_count, last_existing_date


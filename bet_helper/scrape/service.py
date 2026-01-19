from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from bet_helper.storage import historical_path, upcoming_path, load_json, save_json, migrate_legacy_historical_files
from bet_helper.scrape.delta import check_fixtures_delta
from bet_helper.scrape.footystats import scrape_fixtures_only, scrape_upcoming_matches, newest_finished_date_fast

from scrapers.footystats_scraper import FootyStatsScraper


@dataclass
class ScrapeReport:
    league: str
    has_delta: bool
    new_matches_count: int
    last_existing_date: str | None
    upcoming_matches_count: int
    updated_historical: bool
    historical_path: str
    upcoming_path: str
    timestamp: str


def _odds_str_from_match_dict(match_dict: dict) -> str:
    # FootyStatsScraper yields odds as floats/strings or None; normalize to "h|d|a"
    h = match_dict.get("home_odds")
    d = match_dict.get("draw_odds")
    a = match_dict.get("away_odds")
    def _norm(x: Any) -> str:
        if x is None:
            return "0"
        try:
            return str(float(x))
        except Exception:
            s = str(x).strip()
            return "0" if not s or s == "-" else s
    return f"{_norm(h)}|{_norm(d)}|{_norm(a)}"


def update_historical_from_fixtures(league: str, fixtures: list[dict]) -> int:
    """
    Update historical_matches_{league}.json from scraped fixture groups.
    Only writes matches with results (scores present).
    Returns number of new matches added.
    """
    hist_file = historical_path(league)
    existing: dict = load_json(hist_file, default={}) or {}
    historical_data = dict(existing)

    # Build a quick "last saved date" to avoid rescanning deep if desired
    last_saved_date = None
    try:
        dates = []
        for _, md in existing.items():
            ds = (md or {}).get("Date")
            if ds:
                dates.append(datetime.strptime(ds, "%d %b %Y"))
        if dates:
            last_saved_date = max(dates)
    except Exception:
        last_saved_date = None

    scraper = FootyStatsScraper(username=None, password=None)
    added = 0
    try:
        for group in fixtures:
            date = group.get("date")
            matches = group.get("matches") or []
            if not date or not matches:
                continue

            # Parse group date for skip-older logic
            try:
                group_dt = datetime.strptime(date, "%d %b %Y")
            except Exception:
                group_dt = None

            # If this date is older than our last saved date, skip it (faster)
            if last_saved_date and group_dt and group_dt < last_saved_date:
                continue

            for m in matches:
                hs = m.get("home_score")
                as_ = m.get("away_score")
                if hs is None and as_ is None:
                    continue  # future

                home = (m.get("home_team") or "").strip()
                away = (m.get("away_team") or "").strip()
                if not home or not away:
                    continue

                key = f"{date}_{home}-{away}"
                if key in historical_data:
                    continue

                # Scrape H2H stats if we have a URL
                h2h_url = m.get("h2h_url")
                h2h_stats = None
                if h2h_url:
                    try:
                        h2h_stats = scraper.scrape_fixture_details(h2h_url)
                    except Exception:
                        h2h_stats = None

                try:
                    hs_i = int(hs)
                    as_i = int(as_)
                except Exception:
                    continue

                historical_data[key] = {
                    "Date": date,
                    "Match": f"{home}-{away}",
                    "Score": f"{hs_i}:{as_i}",
                    "Odds": _odds_str_from_match_dict(m),
                    "Result": "Home" if hs_i > as_i else ("Draw" if hs_i == as_i else "Away"),
                }
                if h2h_stats:
                    historical_data[key]["h2h_stats"] = h2h_stats

                added += 1

    finally:
        try:
            scraper._close_driver()
        except Exception:
            pass

    if added > 0:
        save_json(hist_file, historical_data)

    return added


def scrape_league(league: str, update_upcoming: bool = True) -> ScrapeReport:
    """
    Scrape fixtures; if delta exists update historical file; optionally refresh upcoming fixtures cache.
    """
    migrate_legacy_historical_files()
    # Fast delta check: compare local last saved date vs newest finished on FootyStats.
    # Only do the heavy fixtures scrape if we detect a delta.
    existing = load_json(historical_path(league), default={}) or {}
    last_saved_date = None
    try:
        dates = []
        for _, md in existing.items():
            ds = (md or {}).get("Date")
            if ds:
                dates.append(datetime.strptime(ds, "%d %b %Y"))
        if dates:
            last_saved_date = max(dates)
    except Exception:
        last_saved_date = None

    newest_finished_str = newest_finished_date_fast(league)
    newest_finished_dt = None
    if newest_finished_str:
        try:
            newest_finished_dt = datetime.strptime(newest_finished_str, "%d %b %Y")
        except Exception:
            newest_finished_dt = None

    if newest_finished_dt and last_saved_date and newest_finished_dt <= last_saved_date:
        logging.info(
            f"[scrape] Fast delta: newest_finished={newest_finished_str} equals last_saved={last_saved_date.strftime('%d %b %Y')} -> no delta"
        )
        fixtures = []  # not needed
        has_delta, new_matches_count, last_existing_date = False, 0, last_saved_date
    else:
        # Fallback to heavy scrape when we can't determine delta quickly or if it's newer.
        # When newest_finished_dt > last_saved_date, we detected a delta, but date groups
        # may not be in chronological order, so we can't safely use min_date early-stop.
        # We'll scrape all groups and filter by date afterwards.
        scrape_since = None  # Don't use min_date optimization when delta detected
        fixtures = scrape_fixtures_only(league, since=scrape_since)
        has_delta, new_matches_count, last_existing_date = check_fixtures_delta(historical_path(league), fixtures)

    updated_historical = False
    if has_delta:
        logging.info(f"[scrape] Delta detected for {league}: {new_matches_count} new matches. Updating historical file...")
        added = update_historical_from_fixtures(league, fixtures)
        updated_historical = added > 0
        logging.info(f"[scrape] Historical update complete for {league}: added={added}")
    else:
        logging.info(f"[scrape] No delta for {league}. Skipping historical update.")

    upcoming_matches_count = 0
    if update_upcoming:
        try:
            upcoming_payload = scrape_upcoming_matches(league, past_fixtures_cache=fixtures)
            upcoming_matches_count = len(upcoming_payload)
            save_json(upcoming_path(league), upcoming_payload)
        except Exception as e:
            logging.warning(f"[scrape] Could not refresh upcoming cache for {league}: {e}")

    return ScrapeReport(
        league=league,
        has_delta=bool(has_delta),
        new_matches_count=int(new_matches_count or 0),
        last_existing_date=str(last_existing_date) if last_existing_date else None,
        upcoming_matches_count=int(upcoming_matches_count),
        updated_historical=bool(updated_historical),
        historical_path=str(historical_path(league)),
        upcoming_path=str(upcoming_path(league)),
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


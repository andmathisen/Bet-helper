from __future__ import annotations

import logging
from datetime import datetime, timedelta

from scrapers.footystats_scraper import FootyStatsScraper
from scrapers.league_mapping import code_to_path


def newest_finished_date_fast(league: str) -> str | None:
    """
    Fast delta helper: returns newest finished fixture date ("DD MMM YYYY") or None.
    Uses a lightweight scan of the fixtures page (no expanding hidden groups).
    """
    league_path = code_to_path(league)
    if league_path is None:
        raise ValueError(f"Unknown league code: {league}")
    scraper = FootyStatsScraper()
    try:
        return scraper.get_newest_finished_fixture_date(league_path)
    finally:
        scraper._close_driver()


def scrape_fixtures_only(league: str, since: datetime | None = None) -> list[dict]:
    league_path = code_to_path(league)
    if league_path is None:
        raise ValueError(f"Unknown league code: {league}")
    scraper = FootyStatsScraper()
    try:
        fixtures = scraper.scrape_fixtures(league_path, include_past=True, include_future=False, min_date=since)
        logging.info(f"Scraped {len(fixtures)} fixture groups for {league}")
        return fixtures
    finally:
        scraper._close_driver()


def _current_gameweek_from_past_fixtures(past_fixtures: list[dict]) -> int | None:
    # Footystats uses descending gameweek numbers; the most recent completed gw is the minimum with completed.
    gameweek_info = {}
    for date_group in past_fixtures or []:
        gw = date_group.get("gameweek")
        if gw is None:
            continue
        for m in date_group.get("matches") or []:
            hs = m.get("home_score")
            as_ = m.get("away_score")
            if hs is None and as_ is None:
                continue
            info = gameweek_info.setdefault(gw, {"has_completed": False})
            info["has_completed"] = True

    completed = [gw for gw, info in gameweek_info.items() if info.get("has_completed")]
    if not completed:
        return None
    return min(completed)


def _current_gameweek_from_future_fixtures(future_fixtures: list[dict]) -> int | None:
    """
    Fallback: Try to determine current gameweek from future fixtures.
    Looks for gameweeks with matches happening today or very soon.
    """
    today = datetime.now().date()
    soon_threshold = today + timedelta(days=3)  # Within next 3 days
    
    gameweek_dates: dict[int, list[datetime]] = {}
    
    for date_group in future_fixtures or []:
        gw = date_group.get("gameweek")
        date_str = date_group.get("date")
        if gw is None or not date_str:
            continue
        
        try:
            match_date = datetime.strptime(date_str, "%d %b %Y")
            if gw not in gameweek_dates:
                gameweek_dates[gw] = []
            gameweek_dates[gw].append(match_date)
        except Exception:
            continue
    
    # Find gameweek with matches closest to today (within threshold)
    for gw in sorted(gameweek_dates.keys()):
        dates = gameweek_dates[gw]
        if any(d.date() <= soon_threshold for d in dates):
            return gw
    
    return None


def filter_relevant_future_matches(future_fixtures: list[dict], past_fixtures: list[dict]) -> list[dict]:
    """
    Keep:
      - unplayed matches in current gameweek (if any)
      - unplayed matches in next gameweek
    
    If current gameweek cannot be determined from past fixtures, tries to infer it from future fixtures.
    If that also fails, includes all unplayed matches within the next 7 days.
    """
    current_gw = _current_gameweek_from_past_fixtures(past_fixtures)
    
    # Fallback: try to determine from future fixtures if past fixtures don't have gameweek info
    if current_gw is None:
        current_gw = _current_gameweek_from_future_fixtures(future_fixtures)
        if current_gw is not None:
            logging.info(f"Inferred current gameweek {current_gw} from future fixtures (past fixtures lacked gameweek info)")
    
    # If still can't determine, use date-based fallback: include matches within next 7 days
    use_date_fallback = current_gw is None
    if use_date_fallback:
        today = datetime.now().date()
        date_threshold = today + timedelta(days=7)
        logging.warning("No gameweek info available - falling back to date-based filtering (next 7 days)")
    
    next_gw = current_gw - 1 if current_gw is not None else None
    filtered = []
    cur_cnt = 0
    next_cnt = 0
    fallback_cnt = 0

    for date_group in future_fixtures or []:
        gw = date_group.get("gameweek")
        date_str = date_group.get("date")
        matches = date_group.get("matches") or []
        keep = []
        
        # Determine if this date group should be included
        include_group = False
        if not use_date_fallback and gw is not None:
            include_group = (gw == current_gw or gw == next_gw)
        elif use_date_fallback:
            # Date-based fallback: include if date is within threshold
            try:
                match_date = datetime.strptime(date_str, "%d %b %Y")
                include_group = match_date.date() <= date_threshold
            except Exception:
                continue

        if include_group:
            for m in matches:
                hs = m.get("home_score")
                as_ = m.get("away_score")
                if hs is None and as_ is None:  # Unplayed matches only
                    keep.append(m)
                    if use_date_fallback:
                        fallback_cnt += 1
                    elif gw == current_gw:
                        cur_cnt += 1
                    else:
                        next_cnt += 1

        if keep:
            dg = dict(date_group)
            dg["matches"] = keep
            filtered.append(dg)

    if use_date_fallback:
        logging.info(f"Filtered future matches: {fallback_cnt} matches within next 7 days (date-based fallback)")
    else:
        logging.info(f"Filtered future matches: {cur_cnt} in current gameweek ({current_gw}), {next_cnt} in next gameweek ({next_gw})")
    return filtered


def scrape_upcoming_matches(league: str, past_fixtures_cache: list[dict] | None = None) -> list[dict]:
    """
    Returns a list of dicts: {home, away, odds:{home,draw,away}}
    """
    league_path = code_to_path(league)
    if league_path is None:
        raise ValueError(f"Unknown league code: {league}")

    scraper = FootyStatsScraper()
    try:
        # If cache is None or empty, we need to scrape past fixtures for gameweek detection
        if past_fixtures_cache is None or len(past_fixtures_cache) == 0:
            past_fixtures = scraper.scrape_fixtures(league_path, include_past=True, include_future=False)
            if past_fixtures_cache is not None:
                logging.debug(f"Re-scraping past fixtures for {league} (empty cache provided, needed for gameweek detection)")
        else:
            past_fixtures = past_fixtures_cache
            logging.debug(f"Using cached past fixtures for {league} ({len(past_fixtures)} groups)")

        future_fixtures = scraper.scrape_fixtures(league_path, include_past=False, include_future=True)
        filtered = filter_relevant_future_matches(future_fixtures, past_fixtures=past_fixtures)

        upcoming = []
        for date_group in filtered:
            for m in date_group.get("matches") or []:
                home = (m.get("home_team") or "").strip()
                away = (m.get("away_team") or "").strip()
                if not home or not away:
                    continue
                upcoming.append(
                    {
                        "home": home,
                        "away": away,
                        "odds": {
                            "home": m.get("home_odds") or 0,
                            "draw": m.get("draw_odds") or 0,
                            "away": m.get("away_odds") or 0,
                        },
                    }
                )

        logging.info(f"Scraped {len(upcoming)} relevant future matches for {league} (current + next round only)")
        return upcoming
    finally:
        scraper._close_driver()


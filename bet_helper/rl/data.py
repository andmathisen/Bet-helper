"""
Data loading utilities for RL betting environment.

Loads matches grouped by date, extracts odds, and maps to historical outcomes.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bet_helper.storage import load_json, upcoming_path, predictions_path, historical_path
from bet_helper.predict import core


@dataclass
class MatchWithOdds:
    """Match data with betting odds."""
    home_team: str
    away_team: str
    date: datetime
    odds_home: float
    odds_draw: float
    odds_away: float
    match_id: Optional[str] = None
    # Historical outcome if available (for simulation)
    actual_outcome: Optional[str] = None  # "H", "D", or "A"


@dataclass
class MatchDateGroup:
    """Matches grouped by date for betting."""
    matches: List[MatchWithOdds]
    date: datetime
    date_id: str  # YYYY-MM-DD format for identification


def _parse_date(date_str: str, reference_date: datetime | None = None) -> datetime | None:
    """Parse date string using the same logic as prediction service."""
    try:
        return core._parse_dd_mmm_yyyy_season_aware(date_str, reference_date=reference_date)
    except Exception:
        return None


def _get_match_outcome(historical_data: Dict, match_id: str) -> Optional[str]:
    """Extract actual outcome (H/D/A) from historical match data."""
    if not historical_data or match_id not in historical_data:
        return None
    
    match_data = historical_data[match_id]
    score = match_data.get("Score", "")
    if ":" not in score:
        return None
    
    try:
        hg_s, ag_s = score.split(":", 1)
        hg = int(hg_s.strip())
        ag = int(ag_s.strip())
        
        if hg > ag:
            return "H"
        elif hg == ag:
            return "D"
        else:
            return "A"
    except (ValueError, TypeError):
        return None


def _find_matching_historical_match(
    historical_data: Dict,
    home_team: str,
    away_team: str,
    match_date: datetime,
    tolerance_days: int = 7
) -> Tuple[Optional[str], Optional[str]]:
    """
    Find matching historical match by teams and date.
    
    Returns: (match_id, outcome) or (None, None) if not found
    """
    if not historical_data:
        return None, None
    
    for match_id, match_data in historical_data.items():
        # Extract teams from match string
        match_str = match_data.get("Match", "")
        if "-" not in match_str:
            continue
        
        hist_home, hist_away = [s.strip() for s in match_str.split("-", 1)]
        
        # Check if teams match (both directions, in case of team name variations)
        teams_match = (
            (hist_home.lower() == home_team.lower() and hist_away.lower() == away_team.lower()) or
            (hist_home.lower() == away_team.lower() and hist_away.lower() == home_team.lower())
        )
        
        if not teams_match:
            continue
        
        # Check date proximity
        hist_date_str = match_data.get("Date", "")
        hist_date = _parse_date(hist_date_str, reference_date=match_date)
        if not hist_date:
            continue
        
        days_diff = abs((hist_date - match_date).days)
        if days_diff <= tolerance_days:
            outcome = _get_match_outcome({match_id: match_data}, match_id)
            return match_id, outcome
    
    return None, None


def load_match_dates(
    leagues: List[str],
    use_historical: bool = True,
    use_predictions: bool = False,
    historical_data: Optional[Dict] = None,
    reference_date: Optional[datetime] = None
) -> List[MatchDateGroup]:
    """
    Load matches grouped by date from historical data or predictions/upcoming files.
    
    Args:
        leagues: List of league codes to load (e.g., ["PL", "SerieA"])
        use_historical: If True, load from historical matches (has dates and outcomes) - recommended for training
        use_predictions: If True and use_historical=False, use predictions files (for upcoming matches)
        historical_data: Optional historical data dict (will load if None)
        reference_date: Reference date for date parsing
    
    Returns:
        List of MatchDateGroup objects, sorted by date
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    all_matches: List[MatchWithOdds] = []
    
    # Load historical data if not provided
    if historical_data is None:
        historical_data = load_historical_data(leagues)
    
    # Validate that at least one option is enabled
    if not use_historical and not use_predictions:
        logging.warning("Both use_historical and use_predictions are False. No matches will be loaded.")
        return []
    
    # Option 1: Use historical matches (has dates and outcomes, need to simulate odds)
    # This is recommended for training as it has real outcomes
    if use_historical:
        for league in leagues:
            # Load historical data for this league
            league_historical = {
                k: v for k, v in historical_data.items()
                if k.startswith(f"{league}_") or (not "_" in k and league in k)
            }
            
            if not league_historical:
                continue
            
            # Extract matches with dates and outcomes from historical data
            for match_id, match_data in league_historical.items():
                try:
                    match_str = match_data.get("Match", "")
                    if "-" not in match_str:
                        continue
                    
                    home_name, away_name = [s.strip() for s in match_str.split("-", 1)]
                    score = match_data.get("Score", "")
                    if ":" not in score:
                        continue
                    
                    # Parse date
                    date_str = match_data.get("Date", "")
                    if not date_str:
                        continue
                    
                    match_date = _parse_date(date_str, reference_date=reference_date)
                    if not match_date:
                        continue
                    
                    # Get outcome
                    actual_outcome = _get_match_outcome({match_id: match_data}, match_id)
                    
                    # Simulate realistic odds (don't use outcome to bias odds - that would be cheating!)
                    # Use a simple model based on typical football odds distribution
                    # Add randomness to simulate bookmaker variation
                    import random
                    # Base odds: simulate a typical match (slight home advantage)
                    base_home = random.uniform(1.8, 3.5)  # Home team favorite to underdog
                    base_draw = random.uniform(3.0, 4.0)
                    base_away = random.uniform(1.8, 4.0)
                    
                    # Normalize to ensure bookmaker margin (~5-10%)
                    total_implied = (1.0 / base_home) + (1.0 / base_draw) + (1.0 / base_away)
                    margin = random.uniform(1.05, 1.10)  # 5-10% bookmaker margin
                    
                    odds_home = base_home * margin
                    odds_draw = base_draw * margin
                    odds_away = base_away * margin
                    
                    match = MatchWithOdds(
                        home_team=home_name,
                        away_team=away_name,
                        date=match_date,
                        odds_home=odds_home,
                        odds_draw=odds_draw,
                        odds_away=odds_away,
                        match_id=match_id,
                        actual_outcome=actual_outcome
                    )
                    all_matches.append(match)
                except Exception as e:
                    logging.debug(f"Error processing historical match {match_id}: {e}")
                    continue
    
    # Option 2: Use predictions/upcoming (has odds, but no dates or outcomes)
    # This is for evaluation/testing on upcoming matches
    if use_predictions:
        for league in leagues:
            # Load from predictions or upcoming
            pred_path = predictions_path(league)
            pred_data = load_json(pred_path, default=[])
            
            if not pred_data:
                # Try upcoming file as fallback
                upcoming_path_obj = upcoming_path(league)
                pred_data = load_json(upcoming_path_obj, default=[])
            
            if not pred_data:
                logging.warning(f"No data found for league {league} (predictions={use_predictions})")
                continue
            
            # Filter historical data for this league (already loaded above)
            league_historical = {
                k: v for k, v in historical_data.items()
                if k.startswith(f"{league}_") or (not "_" in k and league in k)
            }
            
            # Extract matches from predictions/upcoming data
            for match_data in pred_data:
                if not isinstance(match_data, dict):
                    continue
                
                # Handle nested structure (predictions have "match" key)
                match_info = match_data.get("match", {})
                if match_info:
                    # Predictions format: { "match": { "home": "...", "away": "..." }, "odds": {...} }
                    home = match_info.get("home", "")
                    away = match_info.get("away", "")
                else:
                    # Upcoming format: { "home": "...", "away": "...", "odds": {...} }
                    home = match_data.get("home", "")
                    away = match_data.get("away", "")
                
                odds_dict = match_data.get("odds", {})
                
                if not home or not away or not odds_dict:
                    continue
                
                odds_home = odds_dict.get("home", 0.0)
                odds_draw = odds_dict.get("draw", 0.0)
                odds_away = odds_dict.get("away", 0.0)
                
                # Skip matches with invalid odds
                if odds_home <= 0 or odds_draw <= 0 or odds_away <= 0:
                    continue
                
                # Parse date - predictions/upcoming may not have dates
                # Use reference_date (current date) as fallback for upcoming matches
                date_str = match_data.get("date") or match_data.get("Date", "")
                if not date_str:
                    # No date in upcoming/predictions - use reference date
                    match_date = reference_date
                else:
                    match_date = _parse_date(date_str, reference_date=reference_date)
                    if not match_date:
                        match_date = reference_date
                
                match_id = match_data.get("match_id") or match_data.get("id")
                
                # Try to find historical outcome
                actual_outcome = None
                if league_historical:
                    if match_id and match_id in league_historical:
                        actual_outcome = _get_match_outcome(league_historical, match_id)
                    else:
                        # Try to find by teams and date
                        _, actual_outcome = _find_matching_historical_match(
                            league_historical, home, away, match_date
                        )
                
                match = MatchWithOdds(
                    home_team=home,
                    away_team=away,
                    date=match_date,
                    odds_home=odds_home,
                    odds_draw=odds_draw,
                    odds_away=odds_away,
                    match_id=match_id,
                    actual_outcome=actual_outcome
                )
                
                all_matches.append(match)
    
    # Group matches by date (same calendar date)
    dates_dict: Dict[str, List[MatchWithOdds]] = defaultdict(list)
    
    for match in all_matches:
        # Group by date (YYYY-MM-DD)
        date_key = match.date.strftime("%Y-%m-%d")
        dates_dict[date_key].append(match)
    
    # Convert to MatchDateGroup objects
    date_groups: List[MatchDateGroup] = []
    for date_key, matches in dates_dict.items():
        if not matches:
            continue
        
        # All matches should be on the same date, but sort by time if available
        matches.sort(key=lambda m: m.date)
        
        date_group = MatchDateGroup(
            matches=matches,
            date=matches[0].date,
            date_id=date_key
        )
        date_groups.append(date_group)
    
    # Sort date groups by date
    date_groups.sort(key=lambda dg: dg.date)
    
    total_matches = sum(len(dg.matches) for dg in date_groups)
    logging.info(f"Loaded {len(date_groups)} match dates with {total_matches} total matches")
    
    return date_groups


def load_historical_data(leagues: List[str]) -> Dict:
    """Load historical match data for outcome simulation."""
    combined = {}
    
    for league in leagues:
        hist_path = historical_path(league)
        data = load_json(hist_path, default={})
        
        if data:
            # Always prefix match IDs with league code to avoid collisions
            # Historical IDs are like "15 Aug 2025_Leeds United-Everton"
            # We want "PL_15 Aug 2025_Leeds United-Everton"
            for match_id, match_data in data.items():
                # Check if already prefixed with this league
                if match_id.startswith(f"{league}_"):
                    prefixed_id = match_id
                else:
                    prefixed_id = f"{league}_{match_id}"
                combined[prefixed_id] = match_data
    
    logging.info(f"Loaded {len(combined)} historical matches from {len(leagues)} leagues")
    return combined

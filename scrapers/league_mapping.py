"""Mapping from old league codes to footystats.org paths."""
from typing import Dict, Optional


# Mapping from old league codes (used in main.py menu) to footystats.org paths
# Example: "LaLiga" -> "/spain/la-liga"
LEAGUE_CODE_TO_PATH: Dict[str, str] = {
    "PL": "/england/premier-league",
    "SerieA": "/italy/serie-a",
    "SerieB": "/italy/serie-b",
    "LaLiga": "/spain/la-liga",
    "Bundesliga": "/germany/bundesliga",
    "Bundesliga2": "/germany/2-bundesliga",
    "Championship": "/england/championship",
    "LigaPortugal": "/portugal/liga-portugal",
    "Ligue1": "/france/ligue-1",
    "Ligue2": "/france/ligue-2",
    "LeagueOne": "/england/efl-league-one",
    "LeagueTwo": "/england/efl-league-two",
    "Eredivisie": "/netherlands/eredivisie"
}


def code_to_path(league_code: str) -> Optional[str]:
    """
    Convert old league code to footystats.org path.
    
    Args:
        league_code: Old league code (e.g., "LaLiga", "PL")
        
    Returns:
        footystats.org path (e.g., "/spain/la-liga") or None if not found
    """
    return LEAGUE_CODE_TO_PATH.get(league_code)


def path_to_code(league_path: str) -> Optional[str]:
    """
    Convert footystats.org path to old league code (reverse mapping).
    
    Args:
        league_path: footystats.org path (e.g., "/spain/la-liga")
        
    Returns:
        Old league code (e.g., "LaLiga") or None if not found
    """
    reverse_mapping = {path: code for code, path in LEAGUE_CODE_TO_PATH.items()}
    return reverse_mapping.get(league_path)
"""
Simple data models for teams and matches.
Replaces legacy match.py classes with dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MatchData:
    """Simple match data structure."""
    date: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int

    def get_score(self) -> tuple[int, int]:
        """Return (home_goals, away_goals)."""
        return (self.home_goals, self.away_goals)


@dataclass
class TeamData:
    """Simple team data structure with form history."""
    name: str
    form: list[MatchData] = field(default_factory=list)

    def add_match(self, match: MatchData) -> None:
        """
        Add a match to form history (newest first).
        
        Matches should be processed in descending date order (newest first).
        This method appends to the end, so processing newest→oldest results in form=[newest, ..., oldest].
        """
        self.form.append(match)

    def get_name(self) -> str:
        """Get team name."""
        return self.name


def calculate_team_form(matches: list[MatchData], team_name: str, n: int = 5) -> tuple[float, float, float, float, float, float, float]:
    """
    Calculate form statistics for a team from the last n matches.
    
    Returns tuple: (win_rate_all, win_rate_home, win_rate_away,
                   avg_scored_home, avg_scored_away,
                   avg_conceded_home, avg_conceded_away)
    """
    matches_home = 0
    matches_away = 0
    win_rate_all = 0
    win_rate_home = 0
    win_rate_away = 0
    scored_home = 0
    scored_away = 0
    conceded_home = 0
    conceded_away = 0

    n_matches = min(len(matches), n)
    for i in range(n_matches):
        m = matches[i]
        h_goals, a_goals = m.get_score()

        if m.home_team == team_name:
            matches_home += 1
            scored_home += h_goals
            conceded_home += a_goals
            if h_goals == a_goals:
                win_rate_all += 1
                win_rate_home += 1
            elif h_goals > a_goals:
                win_rate_all += 3
                win_rate_home += 3
        else:
            matches_away += 1
            scored_away += a_goals
            conceded_away += h_goals
            if h_goals == a_goals:
                win_rate_all += 1
                win_rate_away += 1
            elif h_goals < a_goals:
                win_rate_all += 3
                win_rate_away += 3

    matches_home_safe = max(matches_home, 1)
    matches_away_safe = max(matches_away, 1)
    n_matches_safe = max(n_matches, 1)

    return (
        round(win_rate_all / n_matches_safe, 2),
        round(win_rate_home / matches_home_safe, 2),
        round(win_rate_away / matches_away_safe, 2),
        round(scored_home / matches_home_safe, 2),
        round(scored_away / matches_away_safe, 2),
        round(conceded_home / matches_home_safe, 2),
        round(conceded_away / matches_away_safe, 2),
    )

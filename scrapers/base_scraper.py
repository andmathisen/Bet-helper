"""Abstract base class for football data scrapers."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Dict, Any, Optional


class BaseScraper(ABC):
    """
    Abstract base class defining the interface for all football data scrapers.
    
    All scraper implementations must inherit from this class and implement
    all abstract methods to ensure compatibility with the main application.
    
    All methods return structured data (dicts/lists) instead of strings for better
    type safety and easier processing.
    """
    
    @abstractmethod
    def scrape_league_table(self, league_path: str) -> List[Dict[str, Any]]:
        """
        Scrape league table/standings data.
        
        Args:
            league_path: Path to the league (e.g., "/spain/la-liga" for footystats.org)
            
        Returns:
            List of dictionaries, each representing a team with the following keys:
            - position (int): Team position in table
            - team_name (str): Team name
            - matches_played (int): Matches played
            - wins (int): Number of wins
            - draws (int): Number of draws
            - losses (int): Number of losses
            - goals_scored (int): Goals scored
            - goals_conceded (int): Goals conceded
            - points (int): Total points
            
        Raises:
            Exception: If scraping fails
        """
        pass
    
    @abstractmethod
    def scrape_fixtures(self, league_path: str, include_past: bool = True, 
                       include_future: bool = True) -> List[Dict[str, Any]]:
        """
        Scrape fixture data (both past and future matches).
        
        Args:
            league_path: Path to the league (e.g., "/spain/la-liga" for footystats.org)
            include_past: Whether to include past/finished fixtures
            include_future: Whether to include future/scheduled fixtures
            
        Returns:
            List of dictionaries, each representing a date group with matches:
            {
                "date": "08 Jan 2026",  # Date string
                "matches": [
                    {
                        "time": "21:00",
                        "home_team": "Arsenal",
                        "away_team": "Liverpool",
                        "home_score": 2,  # None for future matches
                        "away_score": 1,  # None for future matches
                        "home_odds": 1.59,  # None if not available
                        "draw_odds": 4.22,  # None if not available
                        "away_odds": 5.60,  # None if not available
                        "h2h_url": "https://..."  # Optional H2H stats URL
                    },
                    ...
                ]
            }
            
        Raises:
            Exception: If scraping fails
        """
        pass
    
    @abstractmethod
    def scrape_future_fixtures(self, league_path: str) -> List[Dict[str, Any]]:
        """
        Scrape future/scheduled fixtures only.
        
        Args:
            league_path: Path to the league (e.g., "/spain/la-liga" for footystats.org)
            
        Returns:
            List of dicts with match data: {"home_team": str, "away_team": str, "home_odds": float, "draw_odds": float, "away_odds": float}
            
        Raises:
            Exception: If scraping fails
        """
        pass
    
    @abstractmethod
    def scrape_fixture_details(self, fixture_url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape detailed fixture data from H2H/stats page.
        
        This method supplements the basic fixture data with additional statistics
        like head-to-head records, team form, etc.
        
        Args:
            fixture_url: Full URL to the fixture details page
                        (e.g., "https://footystats.org/spain/real-club-deportivo-mallorca-vs-girona-fc-h2h-stats#8200733")
            
        Returns:
            Dictionary containing detailed fixture statistics, or None if scraping fails.
            The structure is flexible but should include relevant H2H and team stats.
            
        Raises:
            Exception: If scraping fails
        """
        pass
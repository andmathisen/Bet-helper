"""
Reinforcement Learning environment for betting strategy optimization.

Gymnasium-compatible environment where an agent learns to place bets on football matches
to maximize token balance.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from bet_helper.rl.data import MatchDateGroup, MatchWithOdds
from bet_helper.predict.ml_model import (
    extract_features_for_match,
    fit_ml_model,
    predict_with_ml_model,
)
from bet_helper.models import TeamData, MatchData
from bet_helper.storage import load_json, historical_path


class BettingEnv(gym.Env):
    """
    RL environment for learning betting strategies on football matches.
    
    The agent can place bets on one or more matches happening on the same date.
    - Single bets: Bet on one match outcome
    - Accumulator bets: Bet on multiple matches on the same date (all must win)
    
    State: ML model features + current tokens + date context
    Action: Bet type + match selections + outcomes + bet amount
    Reward: Normalized profit/loss from bets
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        match_dates: List[MatchDateGroup],
        historical_data: Dict,
        initial_tokens: float = 1000.0,
        max_matches_per_bet: int = 5,
        feature_extraction_teams: Optional[Dict[str, TeamData]] = None,
        ml_model: Optional[Any] = None,
        ml_feature_names: Optional[List[str]] = None,
        use_ml_predictions: bool = True,
    ):
        """
        Initialize betting environment.
        
        Args:
            match_dates: List of MatchDateGroup objects (matches grouped by date)
            historical_data: Historical match data for feature extraction
            initial_tokens: Starting token balance
            max_matches_per_bet: Maximum number of matches in an accumulator
            feature_extraction_teams: Pre-built TeamData dict for feature extraction
            ml_model: Optional pre-trained ML model (XGBoost). If None, will load from cache.
            ml_feature_names: Optional feature names for ML model. If None, will load from cache.
            use_ml_predictions: If True, use ML model for outcome simulation and include predictions in observations
        """
        super().__init__()
        
        self.match_dates = match_dates
        self.historical_data = historical_data
        self.initial_tokens = initial_tokens
        self.max_matches_per_bet = max_matches_per_bet
        self.use_ml_predictions = use_ml_predictions
        
        # Build teams dict for feature extraction if not provided
        if feature_extraction_teams is None:
            self._build_teams_dict()
        else:
            self.teams_dict = feature_extraction_teams
        
        # Load or use provided ML model
        if use_ml_predictions:
            if ml_model is None or ml_feature_names is None:
                logging.info("Loading ML model for hybrid approach (RL + ML predictions)...")
                # Load ML model from cache (uses all leagues by default)
                # We need a league for fit_ml_model, but it uses all leagues anyway when use_all_leagues=True
                try:
                    from scrapers.league_mapping import LEAGUE_CODE_TO_PATH
                    first_league = sorted(LEAGUE_CODE_TO_PATH.keys())[0] if LEAGUE_CODE_TO_PATH else "PL"
                except Exception:
                    first_league = "PL"
                
                ml_model, ml_feature_names, _ = fit_ml_model(
                    historical_data, first_league, use_all_leagues=True
                )
                
                if ml_model is None or ml_feature_names is None:
                    logging.warning("Could not load ML model - falling back to odds-based simulation")
                    self.use_ml_predictions = False
                    self.ml_model = None
                    self.ml_feature_names = None
                else:
                    self.ml_model = ml_model
                    self.ml_feature_names = ml_feature_names
                    logging.info(f"✓ ML model loaded successfully ({len(ml_feature_names)} features)")
            else:
                self.ml_model = ml_model
                self.ml_feature_names = ml_feature_names
                logging.info(f"✓ Using provided ML model ({len(ml_feature_names)} features)")
        else:
            self.ml_model = None
            self.ml_feature_names = None
        
        # Determine feature size by extracting features for a sample match
        sample_features = self._extract_features_for_match_sample()
        self.feature_dim = len(sample_features)
        
        # Maximum matches on any single date
        self.max_matches_per_date = max(len(md.matches) for md in match_dates) if match_dates else 10
        
        # Observation space: Hybrid approach includes:
        # - Raw ML features (feature_dim)
        # - ML predicted probabilities (3: P(Home), P(Draw), P(Away))
        # - Value indicators (3: difference between ML probs and implied odds from bookmaker)
        # - Current tokens (normalized)
        # - Matches available on this date (normalized)
        # - Date progress (normalized - how far through the date sequence)
        
        obs_dim = self.feature_dim + 3  # Raw features + tokens + matches + progress
        if use_ml_predictions and self.ml_model:
            obs_dim += 6  # ML predictions (3) + value indicators (3)
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Action space: MultiDiscrete for PPO compatibility
        # Format: [action_type, num_matches, match_idx_0, ..., match_idx_N, outcome_0, ..., outcome_N, bet_amount_bin]
        # - action_type: 0=no_bet, 1=single, 2=accumulator (3 choices)
        # - num_matches: Number of matches (1-based, 0 to max_matches_per_bet-1, so add 1 when using)
        # - match_indices: max_matches_per_bet indices (each 0 to max_matches_per_date)
        # - outcomes: max_matches_per_bet outcomes (each 0=Home, 1=Draw, 2=Away)
        # - bet_amount_bin: Discretized bet fraction (20 bins: 0.05, 0.10, ..., 1.0)
        self.num_bet_bins = 20  # Discretize bet amount into 20 bins
        action_shape = [
            3,  # action_type
            max_matches_per_bet,  # num_matches (will be converted to 1-based)
        ] + [self.max_matches_per_date + 1] * max_matches_per_bet + [3] * max_matches_per_bet + [self.num_bet_bins]  # match indices + outcomes + bet_amount_bin
        
        self.action_space = spaces.MultiDiscrete(action_shape)
        
        # State tracking
        self.current_date_idx = 0
        self.tokens = initial_tokens
        self.bets_placed_today: List[Dict] = []
        self.total_profit = 0.0
        self.total_bets = 0
        self.wins = 0
        
    def _build_teams_dict(self) -> None:
        """Build TeamData dictionary from historical data for feature extraction."""
        from bet_helper.predict.service import _build_teams_from_historical
        
        logging.info("Building teams dictionary from historical data for feature extraction...")
        self.teams_dict = _build_teams_from_historical(self.historical_data)
        logging.info(f"Built teams dict with {len(self.teams_dict)} teams")
    
    def _extract_features_for_match_sample(self) -> Dict[str, float]:
        """Extract features for a sample match to determine feature dimension."""
        if not self.match_dates or not self.match_dates[0].matches:
            # Return dummy features if no data
            return {f"feature_{i}": 0.0 for i in range(50)}
        
        # Use first match from first date
        match = self.match_dates[0].matches[0]
        
        try:
            home_team = self.teams_dict.get(match.home_team)
            away_team = self.teams_dict.get(match.away_team)
            
            if not home_team or not away_team:
                # Return dummy features
                return {f"feature_{i}": 0.0 for i in range(50)}
            
            # Extract features using existing pipeline
            from bet_helper.predict.ml_model import _build_team_h2h_index
            
            match_date = match.date
            h2h_index = _build_team_h2h_index(self.historical_data, reference_date=match_date)
            
            features = extract_features_for_match(
                match.home_team,
                match.away_team,
                home_team.form,
                away_team.form,
                h2h_index,
                match_date,
                league=None  # Will be inferred if needed
            )
            
            return features
        except Exception as e:
            logging.warning(f"Error extracting sample features: {e}")
            return {f"feature_{i}": 0.0 for i in range(50)}
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to start of first date."""
        super().reset(seed=seed)
        
        self.current_date_idx = 0
        self.tokens = self.initial_tokens
        self.bets_placed_today = []
        self.total_profit = 0.0
        self.total_bets = 0
        self.wins = 0
        
        observation = self._get_observation()
        info = {
            "tokens": self.tokens,
            "date_idx": self.current_date_idx,
            "total_dates": len(self.match_dates),
            "current_date": self.match_dates[self.current_date_idx].date.strftime("%Y-%m-%d") if self.match_dates else None
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute betting action for current date (matches happening on the same date).
        
        Args:
            action: MultiDiscrete array [action_type, num_matches, match_idx_0, ..., match_idx_N, outcome_0, ..., outcome_N, bet_amount_bin]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        reward = 0.0
        info = {}
        
        # Parse flattened action
        action_type = int(action[0])
        num_matches = int(action[1]) + 1  # Convert 0-based to 1-based count
        match_start_idx = 2
        outcome_start_idx = 2 + self.max_matches_per_bet
        bet_amount_bin = int(action[-1])
        
        # Extract match indices and outcomes
        match_indices = action[match_start_idx:outcome_start_idx].astype(int)
        outcomes = action[outcome_start_idx:-1].astype(int)
        
        # Convert bet_amount_bin to fraction (bins: 0.05, 0.10, ..., 1.0)
        bet_fraction = (bet_amount_bin + 1) / self.num_bet_bins  # Maps 0-19 to 0.05-1.0
        
        # Validate action
        if self.current_date_idx >= len(self.match_dates):
            # No more dates, return zero reward
            return self._get_observation(), 0.0, True, False, {"message": "No more dates"}
        
        current_date_group = self.match_dates[self.current_date_idx]
        
        if action_type == 0:  # No bet
            reward = -0.01  # Small penalty for inaction
            info["action"] = "no_bet"
        elif action_type == 1:  # Single bet
            reward, bet_info = self._process_single_bet(
                current_date_group, match_indices[0], outcomes[0], bet_fraction
            )
            info.update(bet_info)
            info["action"] = "single"
        elif action_type == 2:  # Accumulator bet
            if num_matches < 2:
                reward = -0.1  # Invalid: accumulator needs at least 2 matches
                info["action"] = "invalid_accumulator"
            else:
                reward, bet_info = self._process_accumulator_bet(
                    current_date_group, match_indices[:num_matches], outcomes[:num_matches], bet_fraction
                )
                info.update(bet_info)
                info["action"] = "accumulator"
        else:
            reward = -0.1  # Invalid action type
            info["action"] = "invalid"
        
        self.total_profit += reward * self.initial_tokens  # Denormalize for tracking
        self.total_bets += 1
        
        # Move to next date after processing bet
        self.current_date_idx += 1
        
        # Check termination conditions
        terminated = self.current_date_idx >= len(self.match_dates)
        truncated = self.tokens <= 0  # Bankruptcy
        
        current_date_str = None
        if self.current_date_idx < len(self.match_dates):
            current_date_str = self.match_dates[self.current_date_idx].date.strftime("%Y-%m-%d")
        
        info.update({
            "tokens": self.tokens,
            "date_idx": self.current_date_idx,
            "current_date": current_date_str,
            "total_profit": self.total_profit,
            "total_bets": self.total_bets,
            "wins": self.wins,
            "win_rate": self.wins / max(self.total_bets, 1)
        })
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _process_single_bet(
        self,
        date_group: MatchDateGroup,
        match_idx: int,
        outcome: int,
        bet_fraction: float
    ) -> Tuple[float, Dict]:
        """Process a single match bet. Returns (reward, info_dict)."""
        if match_idx >= len(date_group.matches):
            return -0.1, {"error": "Invalid match index"}
        
        match = date_group.matches[match_idx]
        bet_amount = self.tokens * bet_fraction
        
        if bet_amount <= 0 or bet_amount > self.tokens:
            return -0.1, {"error": "Invalid bet amount"}
        
        # Get odds for selected outcome
        outcome_key = ["home", "draw", "away"][outcome]
        odds = {
            "home": match.odds_home,
            "draw": match.odds_draw,
            "away": match.odds_away
        }[outcome_key]
        
        if odds <= 0:
            return -0.1, {"error": "Invalid odds"}
        
        # Get actual outcome (from historical data or simulate)
        actual_outcome = match.actual_outcome
        if actual_outcome is None:
            # Simulate outcome - prefer ML predictions if available (more realistic)
            if self.use_ml_predictions and self.ml_model:
                ml_probs = self._get_ml_prediction(match)
                if ml_probs:
                    # Use ML model predictions for simulation (more accurate than odds)
                    p_home, p_draw, p_away = ml_probs
                    actual_outcome = np.random.choice(
                        ["H", "D", "A"],
                        p=[p_home, p_draw, p_away]
                    )
                else:
                    # Fallback to odds-based simulation if ML prediction fails
                    probs = {
                        "home": 1.0 / match.odds_home,
                        "draw": 1.0 / match.odds_draw,
                        "away": 1.0 / match.odds_away
                    }
                    total = sum(probs.values())
                    normalized = {k: v / total for k, v in probs.items()}
                    actual_outcome = np.random.choice(
                        ["H", "D", "A"],
                        p=[normalized["home"], normalized["draw"], normalized["away"]]
                    )
            else:
                # Simulate based on odds (inverse probability)
                probs = {
                    "home": 1.0 / match.odds_home,
                    "draw": 1.0 / match.odds_draw,
                    "away": 1.0 / match.odds_away
                }
                total = sum(probs.values())
                normalized = {k: v / total for k, v in probs.items()}
                actual_outcome = np.random.choice(
                    ["H", "D", "A"],
                    p=[normalized["home"], normalized["draw"], normalized["away"]]
                )
        
        # Map outcome to index
        outcome_map = {"H": 0, "D": 1, "A": 2}
        actual_outcome_idx = outcome_map.get(actual_outcome, 0)
        
        # Calculate reward
        if outcome == actual_outcome_idx:
            # Win: profit = bet_amount * (odds - 1)
            winnings = bet_amount * odds
            net_profit = winnings - bet_amount
            self.tokens += net_profit
            reward = net_profit / self.initial_tokens  # Normalized
            self.wins += 1
            info = {"result": "win", "winnings": winnings, "profit": net_profit}
        else:
            # Loss: lose bet amount
            self.tokens -= bet_amount
            reward = -bet_amount / self.initial_tokens  # Normalized loss
            info = {"result": "loss", "loss": bet_amount}
        
        return reward, info
    
    def _process_accumulator_bet(
        self,
        date_group: MatchDateGroup,
        match_indices: np.ndarray,
        outcomes: np.ndarray,
        bet_fraction: float
    ) -> Tuple[float, Dict]:
        """Process an accumulator bet on multiple matches from the same date. Returns (reward, info_dict)."""
        bet_amount = self.tokens * bet_fraction
        
        if bet_amount <= 0 or bet_amount > self.tokens:
            return -0.1, {"error": "Invalid bet amount"}
        
        # Validate match indices
        valid_indices = [idx for idx in match_indices if idx < len(date_group.matches)]
        if len(valid_indices) < 2:
            return -0.1, {"error": "Accumulator needs at least 2 valid matches"}
        
        # Calculate accumulator odds and check outcomes
        accumulator_odds = 1.0
        all_correct = True
        match_results = []
        
        for match_idx, outcome in zip(valid_indices, outcomes[:len(valid_indices)]):
            match = date_group.matches[match_idx]
            outcome_key = ["home", "draw", "away"][outcome]
            odds = {
                "home": match.odds_home,
                "draw": match.odds_draw,
                "away": match.odds_away
            }[outcome_key]
            
            accumulator_odds *= odds
            
            # Get actual outcome
            actual_outcome = match.actual_outcome
            if actual_outcome is None:
                # Simulate outcome - prefer ML predictions if available
                if self.use_ml_predictions and self.ml_model:
                    ml_probs = self._get_ml_prediction(match)
                    if ml_probs:
                        # Use ML model predictions for simulation
                        p_home, p_draw, p_away = ml_probs
                        actual_outcome = np.random.choice(
                            ["H", "D", "A"],
                            p=[p_home, p_draw, p_away]
                        )
                    else:
                        # Fallback to odds-based simulation
                        probs = {
                            "home": 1.0 / match.odds_home,
                            "draw": 1.0 / match.odds_draw,
                            "away": 1.0 / match.odds_away
                        }
                        total = sum(probs.values())
                        normalized = {k: v / total for k, v in probs.items()}
                        actual_outcome = np.random.choice(
                            ["H", "D", "A"],
                            p=[normalized["home"], normalized["draw"], normalized["away"]]
                        )
                else:
                    # Simulate based on odds
                    probs = {
                        "home": 1.0 / match.odds_home,
                        "draw": 1.0 / match.odds_draw,
                        "away": 1.0 / match.odds_away
                    }
                    total = sum(probs.values())
                    normalized = {k: v / total for k, v in probs.items()}
                    actual_outcome = np.random.choice(
                        ["H", "D", "A"],
                        p=[normalized["home"], normalized["draw"], normalized["away"]]
                    )
            
            outcome_map = {"H": 0, "D": 1, "A": 2}
            actual_outcome_idx = outcome_map.get(actual_outcome, 0)
            
            match_results.append({
                "match": f"{match.home_team} vs {match.away_team}",
                "predicted": outcome,
                "actual": actual_outcome_idx,
                "correct": outcome == actual_outcome_idx
            })
            
            if outcome != actual_outcome_idx:
                all_correct = False
        
        # Calculate reward
        if all_correct:
            winnings = bet_amount * accumulator_odds
            net_profit = winnings - bet_amount
            self.tokens += net_profit
            reward = net_profit / self.initial_tokens
            self.wins += 1
            info = {"result": "win", "winnings": winnings, "profit": net_profit, "odds": accumulator_odds}
        else:
            self.tokens -= bet_amount
            reward = -bet_amount / self.initial_tokens
            info = {"result": "loss", "loss": bet_amount, "matches": match_results}
        
        return reward, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation with hybrid features (raw + ML predictions)."""
        if self.current_date_idx >= len(self.match_dates):
            # Return zero observation if no more dates
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        current_date_group = self.match_dates[self.current_date_idx]
        
        if not current_date_group.matches:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Extract features from first match (representative of matches on this date)
        # In future, could aggregate features from all matches on the date
        match = current_date_group.matches[0]
        
        try:
            features = self._extract_match_features(match)
        except Exception as e:
            logging.warning(f"Error extracting features: {e}")
            features = np.zeros(self.feature_dim, dtype=np.float32)
        
        # Extract ML predictions and value indicators if ML model is available
        ml_preds = np.zeros(3, dtype=np.float32)  # [P(Home), P(Draw), P(Away)]
        value_indicators = np.zeros(3, dtype=np.float32)  # Value differences
        
        if self.use_ml_predictions and self.ml_model and self.ml_feature_names:
            try:
                # Get ML prediction
                ml_probs = self._get_ml_prediction(match)
                if ml_probs:
                    ml_preds = np.array(ml_probs, dtype=np.float32)  # [p_home, p_draw, p_away]
                    
                    # Calculate implied probabilities from bookmaker odds
                    # Implied prob = 1 / (decimal_odds * margin_factor)
                    # We'll use 1/odds directly (assuming margin is already in odds)
                    odds_home = match.odds_home
                    odds_draw = match.odds_draw
                    odds_away = match.odds_away
                    
                    # Calculate implied probabilities (normalized to sum to 1)
                    implied_home = 1.0 / odds_home if odds_home > 0 else 0.33
                    implied_draw = 1.0 / odds_draw if odds_draw > 0 else 0.33
                    implied_away = 1.0 / odds_away if odds_away > 0 else 0.33
                    implied_sum = implied_home + implied_draw + implied_away
                    if implied_sum > 0:
                        implied_home /= implied_sum
                        implied_draw /= implied_sum
                        implied_away /= implied_sum
                    
                    # Value indicators: difference between ML predictions and implied odds
                    # Positive value = ML thinks outcome is more likely than bookmaker suggests
                    value_indicators = np.array([
                        ml_probs[0] - implied_home,  # Home value
                        ml_probs[1] - implied_draw,  # Draw value
                        ml_probs[2] - implied_away,  # Away value
                    ], dtype=np.float32)
            except Exception as e:
                logging.debug(f"Error getting ML prediction: {e}")
                # Use zeros if ML prediction fails
        
        # Add game state info
        tokens_normalized = self.tokens / self.initial_tokens
        matches_available = len(current_date_group.matches) / 10.0  # Normalized
        date_progress = self.current_date_idx / max(len(self.match_dates), 1)
        
        # Build observation: raw features + ML predictions + value indicators + state
        observation_parts = [
            features,
            np.array([tokens_normalized, matches_available, date_progress], dtype=np.float32)
        ]
        
        if self.use_ml_predictions and self.ml_model:
            observation_parts.extend([ml_preds, value_indicators])
        
        observation = np.concatenate(observation_parts)
        
        return observation.astype(np.float32)
    
    def _get_ml_prediction(self, match: MatchWithOdds) -> Optional[Tuple[float, float, float]]:
        """Get ML model prediction for a match. Returns (p_home, p_draw, p_away) or None."""
        if not self.ml_model or not self.ml_feature_names:
            return None
        
        try:
            home_team = self.teams_dict.get(match.home_team)
            away_team = self.teams_dict.get(match.away_team)
            
            if not home_team or not away_team:
                return None
            
            # Build H2H index for feature extraction
            from bet_helper.predict.ml_model import _build_team_h2h_index
            h2h_index = _build_team_h2h_index(
                self.historical_data,
                reference_date=match.date
            )
            
            # Extract league from match_id if available
            league = None
            if match.match_id and "_" in match.match_id:
                league = match.match_id.split("_")[0]
            
            # Get ML prediction
            ml_probs = predict_with_ml_model(
                match.home_team,
                match.away_team,
                home_team.form,
                away_team.form,
                h2h_index,
                self.ml_model,
                self.ml_feature_names,
                reference_date=match.date,
                league=league
            )
            
            return ml_probs  # Returns (p_home, p_draw, p_away)
        except Exception as e:
            logging.debug(f"Error in ML prediction: {e}")
            return None
    
    def _extract_match_features(self, match: MatchWithOdds) -> np.ndarray:
        """Extract ML model features for a match."""
        home_team = self.teams_dict.get(match.home_team)
        away_team = self.teams_dict.get(match.away_team)
        
        if not home_team or not away_team:
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        try:
            from bet_helper.predict.ml_model import _build_team_h2h_index
            
            h2h_index = _build_team_h2h_index(
                self.historical_data,
                reference_date=match.date
            )
            
            features_dict = extract_features_for_match(
                match.home_team,
                match.away_team,
                home_team.form,
                away_team.form,
                h2h_index,
                match.date,
                league=None
            )
            
            # Convert dict to sorted array
            feature_names = sorted(features_dict.keys())
            features = np.array([features_dict[name] for name in feature_names], dtype=np.float32)
            
            return features
        except Exception as e:
            logging.warning(f"Error in feature extraction: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def render(self):
        """Render environment state (optional, for debugging)."""
        if self.current_date_idx < len(self.match_dates):
            date_group = self.match_dates[self.current_date_idx]
            date_str = date_group.date.strftime("%Y-%m-%d")
            print(f"Date {self.current_date_idx + 1}/{len(self.match_dates)}: {date_str}")
            print(f"Tokens: {self.tokens:.2f}")
            print(f"Matches: {len(date_group.matches)}")
            print(f"Total profit: {self.total_profit:.2f}")
            print(f"Win rate: {self.wins / max(self.total_bets, 1):.2%}")

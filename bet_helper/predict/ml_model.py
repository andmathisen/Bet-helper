"""
Machine learning model for H/D/A prediction using XGBoost.
Uses all available features: form, xG, corners, cards, shots, SOT, etc.
"""
from __future__ import annotations

import logging
import pickle
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import xgboost as xgb
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    XGBOOST_AVAILABLE = True
except Exception as e:
    # Handle ImportError, OSError (e.g., missing libomp), XGBoostError, or any other loading error
    # XGBoost may raise XGBoostError during import if libomp is missing
    XGBOOST_AVAILABLE = False
    xgb = None
    np = None
    # Log at debug level to avoid noise if XGBoost is intentionally not available
    logging.info(f"XGBoost not available: {type(e).__name__}: {e}")

try:
    import shap
    SHAP_AVAILABLE = True
    
    # Monkey-patch SHAP's XGBTreeModelLoader to handle XGBoost 3.1+ base_score string format
    # This fixes: ValueError: could not convert string to float: '[3.0560273E-1,2.5976232E-1,4.3463498E-1]'
    # SHAP 0.50.0+ should have this fix, but if it doesn't work, this patch ensures it does
    try:
        if hasattr(shap.explainers, '_tree') and hasattr(shap.explainers._tree, 'XGBTreeModelLoader'):
            original_init = shap.explainers._tree.XGBTreeModelLoader.__init__
            
            def patched_init(self, xgb_model):
                """Monkey-patched XGBTreeModelLoader to handle string base_score from XGBoost 3.1+"""
                try:
                    # Try original initialization first
                    return original_init(self, xgb_model)
                except (ValueError, TypeError) as e:
                    error_str = str(e)
                    # Check if this is the base_score conversion error we're trying to fix
                    if "base_score" in error_str.lower() or "could not convert string to float" in error_str.lower():
                        # Try to fix the base_score in the model before retrying
                        try:
                            import re
                            if hasattr(xgb_model, 'get_booster'):
                                booster = xgb_model.get_booster()
                                config_str = booster.save_config()
                                config_dict = json.loads(config_str)
                                
                                # Navigate to base_score
                                learner = config_dict.get("learner", {})
                                learner_model_param = learner.get("learner_model_param", {})
                                base_score_val = learner_model_param.get("base_score", None)
                                
                                # Fix if it's a string representation of a list
                                if base_score_val and isinstance(base_score_val, str) and (base_score_val.startswith('[') or ',' in base_score_val):
                                    cleaned = base_score_val.strip('[]')
                                    values = [float(x.strip()) for x in re.split(r',\s*', cleaned)]
                                    fixed_base_score = str(sum(values) / len(values)) if len(values) > 0 else "0.0"
                                    
                                    learner_model_param["base_score"] = fixed_base_score
                                    learner["learner_model_param"] = learner_model_param
                                    config_dict["learner"] = learner
                                    booster.load_config(json.dumps(config_dict))
                                    logging.info(f"[SHAP Patch] Fixed base_score in model from '{base_score_val}' to '{fixed_base_score}', retrying TreeExplainer initialization")
                                    
                                    # Retry original initialization with fixed model
                                    return original_init(self, xgb_model)
                                else:
                                    # base_score wasn't a string list, so this isn't our issue
                                    raise e
                            else:
                                # Model doesn't have get_booster, so this isn't an XGBoost model
                                raise e
                        except Exception as fix_err:
                            logging.warning(f"[SHAP Patch] Failed to fix base_score: {fix_err}")
                            # If fix fails, re-raise original error
                            raise e
                    # If it's not a base_score error, re-raise it
                    raise e
            
            # Apply the monkey-patch
            shap.explainers._tree.XGBTreeModelLoader.__init__ = patched_init
            logging.info("[SHAP Patch] Applied monkey-patch for XGBoost 3.1+ base_score handling")
    except Exception as patch_err:
        # If monkey-patch fails, log but continue - TreeExplainer might still work or we'll use fallback
        logging.debug(f"Could not apply SHAP monkey-patch: {patch_err}")
        
except Exception as e:
    SHAP_AVAILABLE = False
    shap = None
    logging.debug(f"SHAP not available: {type(e).__name__}: {e}")

from bet_helper.storage import data_dir, historical_path, load_json
from bet_helper.predict.core import (
    _parse_dd_mmm_yyyy_season_aware,
    _build_team_h2h_index,
    _normalize_club_name_for_stats,
    _weighted_stat_profile,
    _create_historical_data_fingerprint,
)
from bet_helper.models import TeamData, MatchData, calculate_team_form

# Explicit class order: LabelEncoder encodes alphabetically (A, D, H)
# This must match the order used throughout the codebase
# LabelEncoder sorts alphabetically: A=0, D=1, H=2
CLASS_ORDER = ["A", "D", "H"]  # Away, Draw, Home (alphabetical order)
CLASS_NAMES = ["away", "draw", "home"]  # Corresponding output names
CLASS_TO_INDEX = {"A": 0, "D": 1, "H": 2}  # Map label to encoded index
INDEX_TO_CLASS = {0: "A", 1: "D", 2: "H"}  # Map encoded index to label


def extract_features_for_match(
    home_name: str,
    away_name: str,
    home_form: list[MatchData],
    away_form: list[MatchData],
    team_h2h_index: dict | None,
    reference_date: datetime,
    league: str | None = None,
) -> dict[str, float]:
    """
    Extract all features for a match to use as ML model input.
    
    Returns:
        Dictionary of feature_name -> feature_value
    """
    features = {}
    
    # Form features (last 5 and 10 matches)
    def _get_form_features(form_matches: list[MatchData], team_name: str, prefix: str):
        if not form_matches:
            return {}
        form_5 = calculate_team_form(form_matches[:5], team_name, n=5)
        form_10 = calculate_team_form(form_matches[:10], team_name, n=10)
        
        # Extract: avg_scored_home, avg_scored_away, avg_conceded_home, avg_conceded_away
        # Indices: 3, 4, 5, 6
        return {
            f"{prefix}_avg_scored_home_5": form_5[3],
            f"{prefix}_avg_scored_away_5": form_5[4],
            f"{prefix}_avg_conceded_home_5": form_5[5],
            f"{prefix}_avg_conceded_away_5": form_5[6],
            f"{prefix}_avg_scored_home_10": form_10[3],
            f"{prefix}_avg_scored_away_10": form_10[4],
            f"{prefix}_avg_conceded_home_10": form_10[5],
            f"{prefix}_avg_conceded_away_10": form_10[6],
            f"{prefix}_ppg_5": form_5[0],  # points per game
            f"{prefix}_ppg_10": form_10[0],
        }
    
    home_form_feat = _get_form_features(home_form, home_name, "home")
    away_form_feat = _get_form_features(away_form, away_name, "away")
    features.update(home_form_feat)
    features.update(away_form_feat)
    
    # xG statistics (recency-weighted, 30-day half-life)
    if team_h2h_index:
        home_key = _normalize_club_name_for_stats(home_name)
        away_key = _normalize_club_name_for_stats(away_name)
        home_entries = team_h2h_index.get(home_key, [])
        away_entries = team_h2h_index.get(away_key, [])
        
        # Home team stats
        home_prof_home = _weighted_stat_profile(home_entries, "xg_for", "xg_against", half_life_days=30.0, home_only=True)
        home_prof_all = _weighted_stat_profile(home_entries, "xg_for", "xg_against", half_life_days=30.0, home_only=None)
        home_prof = home_prof_home or home_prof_all
        if home_prof:
            features["home_xg_for"] = float(home_prof["for"])
            features["home_xg_against"] = float(home_prof["against"])
        else:
            features["home_xg_for"] = 0.0
            features["home_xg_against"] = 0.0
        
        # Away team stats
        away_prof_away = _weighted_stat_profile(away_entries, "xg_for", "xg_against", half_life_days=30.0, home_only=False)
        away_prof_all = _weighted_stat_profile(away_entries, "xg_for", "xg_against", half_life_days=30.0, home_only=None)
        away_prof = away_prof_away or away_prof_all
        if away_prof:
            features["away_xg_for"] = float(away_prof["for"])
            features["away_xg_against"] = float(away_prof["against"])
        else:
            features["away_xg_for"] = 0.0
            features["away_xg_against"] = 0.0
        
        # Other statistics (corners, cards, shots, SOT)
        for stat in ["corners", "cards", "shots", "sot"]:
            for_key = f"{stat}_for"
            against_key = f"{stat}_against"
            
            home_prof_stat = _weighted_stat_profile(home_entries, for_key, against_key, half_life_days=30.0, home_only=True) or \
                           _weighted_stat_profile(home_entries, for_key, against_key, half_life_days=30.0, home_only=None)
            if home_prof_stat:
                features[f"home_{stat}_for"] = float(home_prof_stat["for"])
                features[f"home_{stat}_against"] = float(home_prof_stat["against"])
            else:
                features[f"home_{stat}_for"] = 0.0
                features[f"home_{stat}_against"] = 0.0
            
            away_prof_stat = _weighted_stat_profile(away_entries, for_key, against_key, half_life_days=30.0, home_only=False) or \
                           _weighted_stat_profile(away_entries, for_key, against_key, half_life_days=30.0, home_only=None)
            if away_prof_stat:
                features[f"away_{stat}_for"] = float(away_prof_stat["for"])
                features[f"away_{stat}_against"] = float(away_prof_stat["against"])
            else:
                features[f"away_{stat}_for"] = 0.0
                features[f"away_{stat}_against"] = 0.0
    else:
        # Default values if no H2H data
        for stat in ["xg", "corners", "cards", "shots", "sot"]:
            features[f"home_{stat}_for"] = 0.0
            features[f"home_{stat}_against"] = 0.0
            features[f"away_{stat}_for"] = 0.0
            features[f"away_{stat}_against"] = 0.0
    
    # Match context features
    features["home_advantage"] = 1.0  # Always 1.0 for home team
    
    # Days since last match (if form available)
    if home_form:
        try:
            last_match_date = _parse_dd_mmm_yyyy_season_aware(home_form[0].date, reference_date=reference_date)
            if last_match_date:
                days_since_home = (reference_date - last_match_date).days
                features["home_days_since_last_match"] = float(days_since_home)
            else:
                features["home_days_since_last_match"] = 7.0
        except Exception:
            features["home_days_since_last_match"] = 7.0
    else:
        features["home_days_since_last_match"] = 7.0
    
    if away_form:
        try:
            last_match_date = _parse_dd_mmm_yyyy_season_aware(away_form[0].date, reference_date=reference_date)
            if last_match_date:
                days_since_away = (reference_date - last_match_date).days
                features["away_days_since_last_match"] = float(days_since_away)
            else:
                features["away_days_since_last_match"] = 7.0
        except Exception:
            features["away_days_since_last_match"] = 7.0
    else:
        features["away_days_since_last_match"] = 7.0
    
    # League feature (one-hot encoded per league)
    # Get all possible leagues for consistent encoding
    try:
        from scrapers.league_mapping import LEAGUE_CODE_TO_PATH
        all_leagues = sorted(LEAGUE_CODE_TO_PATH.keys())
        
        if league and league in all_leagues:
            # One-hot encode league: set feature to 1.0 for matching league, 0.0 for others
            for league_code in all_leagues:
                features[f"league_{league_code}"] = 1.0 if league_code == league else 0.0
        else:
            # Default: all 0.0 if league is unknown/missing
            for league_code in all_leagues:
                features[f"league_{league_code}"] = 0.0
    except Exception:
        # If league mapping not available, skip league features
        pass
    
    return features


def prepare_training_data_from_historical(
    historical_data: dict,
    min_matches: int = 100,  # Minimum total dataset size
    min_team_matches: int = 5,  # Minimum matches per team (for form calculation)
    reference_date: datetime | None = None,
) -> tuple[list[dict[str, float]], list[str]]:
    """
    Prepare training data from historical matches using time-aware feature extraction.
    
    Args:
        min_matches: Minimum total dataset size required
        min_team_matches: Minimum number of past matches each team must have (for feature calculation)
    
    Returns:
        Tuple of (features_list, labels_list) where labels are "H", "D", or "A"
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    if not XGBOOST_AVAILABLE:
        logging.warning("XGBoost not available - cannot prepare training data")
        return [], []
    
    # Parse and sort matches chronologically
    matches_with_dates: list[tuple[datetime, str, dict]] = []
    for match_id, match_data in historical_data.items():
        try:
            match_str = match_data.get("Match", "")
            if "-" not in match_str:
                continue
            score = match_data.get("Score", "")
            if ":" not in score:
                continue
            
            date_str = match_data.get("Date", "")
            if not date_str:
                continue
            
            match_date = _parse_dd_mmm_yyyy_season_aware(date_str, reference_date=reference_date)
            if not match_date:
                continue
            
            matches_with_dates.append((match_date, match_id, match_data))
        except Exception:
            continue
    
    matches_with_dates.sort(key=lambda x: x[0])  # Oldest first
    
    total_matches = len(matches_with_dates)
    if total_matches < min_matches:
        logging.debug(f"Not enough matches ({total_matches} < {min_matches}) for ML training")
        return [], []
    
    logging.info(f"Processing {total_matches} historical matches for training data...")
    logging.info(f"  Ensuring each team has at least {min_team_matches} matches in history before including their match")
    
    features_list = []
    labels_list = []
    processed_count = 0
    skipped_count = 0
    skipped_insufficient_history = 0
    
    # Process each match using only past data
    for idx, (match_date, match_id, match_data) in enumerate(matches_with_dates):
        try:
            # Filter to past data only
            past_data = {}
            for hist_id, hist_match in historical_data.items():
                hist_date_str = hist_match.get("Date", "")
                if not hist_date_str:
                    continue
                hist_date = _parse_dd_mmm_yyyy_season_aware(hist_date_str, reference_date=reference_date)
                if hist_date and hist_date < match_date:
                    past_data[hist_id] = hist_match
            
            if len(past_data) < min_matches:
                skipped_insufficient_history += 1
                continue
            
            # Extract match info
            match_str = match_data.get("Match", "")
            home_name, away_name = [s.strip() for s in match_str.split("-", 1)]
            
            # Build teams/form from past data only and count how many times each team appears
            # Sort past matches by date descending (newest first) so form is ordered correctly
            past_items = list(past_data.items())
            def _dt_past(md: dict) -> datetime:
                d = _parse_dd_mmm_yyyy_season_aware((md or {}).get("Date", ""), reference_date=match_date)
                return d or datetime(1970, 1, 1)
            past_items.sort(key=lambda kv: _dt_past(kv[1]), reverse=True)
            
            teams = {}
            team_match_counts = {}  # Track how many matches each team has in past_data
            
            for hist_id, hist_match in past_items:
                try:
                    hist_match_str = hist_match.get("Match", "")
                    if "-" not in hist_match_str:
                        continue
                    hist_home, hist_away = [s.strip() for s in hist_match_str.split("-", 1)]
                    hist_score = hist_match.get("Score", "")
                    if ":" not in hist_score:
                        continue
                    hg_s, ag_s = hist_score.split(":", 1)
                    hg = int(hg_s.strip())
                    ag = int(ag_s.strip())
                    hist_date_str = hist_match.get("Date", "")
                    
                    # Initialize teams if not seen before
                    if hist_home not in teams:
                        teams[hist_home] = TeamData(name=hist_home)
                        team_match_counts[hist_home] = 0
                    if hist_away not in teams:
                        teams[hist_away] = TeamData(name=hist_away)
                        team_match_counts[hist_away] = 0
                    
                    # Count matches for each team
                    team_match_counts[hist_home] += 1
                    team_match_counts[hist_away] += 1
                    
                    m = MatchData(date=hist_date_str, home_team=hist_home, away_team=hist_away, home_goals=hg, away_goals=ag)
                    teams[hist_home].add_match(m)
                    teams[hist_away].add_match(m)
                except Exception:
                    continue
            
            # Ensure both teams exist in past_data
            if home_name not in teams or away_name not in teams:
                skipped_insufficient_history += 1
                continue
            
            # Ensure each team has appeared in at least min_team_matches past matches
            # This ensures we have enough history to calculate form features (last 5-10 matches)
            home_matches = team_match_counts.get(home_name, 0)
            away_matches = team_match_counts.get(away_name, 0)
            
            if home_matches < min_team_matches or away_matches < min_team_matches:
                skipped_insufficient_history += 1
                continue
            
            home_team = teams[home_name]
            away_team = teams[away_name]
            
            # Build H2H index from past data only
            team_h2h_index = _build_team_h2h_index(past_data, reference_date=match_date)
            
            # Extract league from match_id prefix (format: "LEAGUE_match_id" when combined)
            match_league = None
            if "_" in match_id:
                prefix = match_id.split("_", 1)[0]
                try:
                    from scrapers.league_mapping import LEAGUE_CODE_TO_PATH
                    if prefix in LEAGUE_CODE_TO_PATH:
                        match_league = prefix
                except Exception:
                    pass
            
            # Extract features
            features = extract_features_for_match(
                home_name, away_name,
                home_team.form, away_team.form,
                team_h2h_index, match_date,
                league=match_league
            )
            
            # Extract label (actual outcome)
            score = match_data.get("Score", "")
            if ":" not in score:
                continue
            hg_s, ag_s = score.split(":", 1)
            try:
                hg = int(hg_s.strip())
                ag = int(ag_s.strip())
                
                if hg > ag:
                    label = "H"
                elif hg == ag:
                    label = "D"
                else:
                    label = "A"
                
                features_list.append(features)
                labels_list.append(label)
                processed_count += 1
                
                # Log progress every 50 matches or periodically
                if processed_count % 50 == 0:
                    progress_pct = (processed_count / total_matches * 100) if total_matches > 0 else 0
                    logging.info(f"  Progress: {processed_count}/{total_matches} matches processed ({progress_pct:.1f}%) - "
                               f"{len(features_list)} valid samples, {skipped_count} skipped "
                               f"({skipped_insufficient_history} insufficient history)")
            except (ValueError, TypeError) as e:
                skipped_count += 1
                continue
            
        except Exception as e:
            skipped_count += 1
            logging.debug(f"Error preparing training data for match {match_id}: {e}")
            continue
    
    logging.info(f"Training data preparation complete: {len(features_list)} valid samples from {total_matches} matches")
    logging.info(f"  Skipped: {skipped_count} total ({skipped_insufficient_history} insufficient team history, "
                f"{skipped_count - skipped_insufficient_history} other errors)")
    logging.info(f"  Success rate: {(len(features_list)/total_matches*100):.1f}%")
    return features_list, labels_list


def train_ml_model(
    historical_data: dict,
    league: str,
    min_matches: int = 100,  # Minimum total dataset size
    min_team_matches: int = 5,  # Minimum matches per team (for form calculation)
    reference_date: datetime | None = None,
) -> tuple[Any, list[str] | None]:
    """
    Train XGBoost model for H/D/A prediction.
    
    Args:
        min_matches: Minimum total dataset size required
        min_team_matches: Minimum number of past matches each team must have (for feature calculation)
    
    Returns:
        Tuple of (trained_model, feature_names, baseline) or (None, None, None) if training fails
    """
    if not XGBOOST_AVAILABLE:
        logging.warning("XGBoost not available - cannot train ML model")
        return None, None
    
    if reference_date is None:
        reference_date = datetime.now()
    
    logging.info(f"Preparing training data for {league} ML model...")
    features_list, labels_list = prepare_training_data_from_historical(historical_data, min_matches, min_team_matches, reference_date)
    
    if len(features_list) < min_matches:
        logging.warning(f"Not enough training samples ({len(features_list)} < {min_matches}) for {league}")
        return None, None
    
    # Convert to numpy arrays
    # Get feature names (from first sample - all should have same keys)
    logging.info("Converting training data to numpy arrays...")
    feature_names = sorted(features_list[0].keys())
    logging.info(f"  Extracted {len(feature_names)} features: {', '.join(feature_names[:5])}{'...' if len(feature_names) > 5 else ''}")
    
    X = np.array([[features[f] for f in feature_names] for features in features_list], dtype=np.float32)
    y = np.array(labels_list)
    logging.info(f"  Training set shape: X={X.shape}, y={y.shape}")
    
    # Encode labels
    # LabelEncoder encodes alphabetically: A=0, D=1, H=2 (matching CLASS_ORDER)
    logging.info("Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Log label distribution
    label_counts = {label: list(y).count(label) for label in ["H", "D", "A"]}
    label_dist = {label: (count / len(y)) * 100 for label, count in label_counts.items()}
    logging.info(f"  Label distribution: Home={label_counts.get('H', 0)} ({label_dist.get('H', 0):.1f}%), "
                f"Draw={label_counts.get('D', 0)} ({label_dist.get('D', 0):.1f}%), "
                f"Away={label_counts.get('A', 0)} ({label_dist.get('A', 0):.1f}%)")
    
    # Verify class order matches our expectation
    if len(label_encoder.classes_) == 3:
        expected_classes = np.array(CLASS_ORDER)  # ["A", "D", "H"]
        if not np.array_equal(label_encoder.classes_, expected_classes):
            logging.warning(f"LabelEncoder classes {label_encoder.classes_} don't match expected {expected_classes}. "
                          f"SHAP explanations may use incorrect class mapping.")
        else:
            logging.info(f"  LabelEncoder class order verified: {list(label_encoder.classes_)}")
    
    # Train XGBoost model
    logging.info(f"Training XGBoost model on {len(X)} samples with {len(feature_names)} features...")
    logging.info(f"  Model parameters: n_estimators=100, max_depth=4, learning_rate=0.1")
    logging.info(f"  Regularization: reg_alpha=1.0 (L1), reg_lambda=1.0 (L2), min_child_weight=3")
    
    # Split data for validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    logging.info(f"  Train/validation split: {len(X_train)} train, {len(X_val)} validation samples")
    
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=100,
        max_depth=4,  # Reduced from 6 to reduce overfitting
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,  # L1 regularization - encourages feature sparsity
        reg_lambda=1.0,  # L2 regularization - reduces feature weights
        min_child_weight=3,  # Minimum sum of instance weight needed in a child (higher = more conservative)
        gamma=0.1,  # Minimum loss reduction required to make a split (higher = more conservative)
        random_state=42,
        eval_metric="mlogloss",
        early_stopping_rounds=10,  # Stop if validation loss doesn't improve for 10 rounds (XGBoost 2.0+ requires this in constructor)
        verbose=1,  # Show training progress (0=silent, 1=warnings, 2=info, 3=debug)
    )
    
    logging.info("  Starting XGBoost training (this may take a while)...")
    start_time = datetime.now()
    
    # Fit with early stopping on validation set
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False  # Suppress per-iteration output, we'll log summary
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Log feature importance
    feature_importance = model.feature_importances_
    importance_dict = dict(zip(feature_names, feature_importance))
    top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    
    logging.info(f"  ✓ XGBoost training completed in {training_time:.1f} seconds")
    logging.info(f"  Best iteration: {model.best_iteration if hasattr(model, 'best_iteration') else 'N/A'}")
    logging.info(f"  Top 10 most important features:")
    for feat, imp in top_features:
        logging.info(f"    {feat}: {imp:.4f}")
    
    # Use full training set for final model (if we want to retrain on all data)
    # For now, model is trained on X_train, but we'll use it on X for baseline
    # In practice, if we want to use all data, we'd retrain without early stopping
    # For baseline computation, we'll use the validation model or retrain on full data
    
    # Compute baseline as average predicted probabilities on full training data
    # Retrain on full dataset for final model (early stopping already determined optimal complexity)
    logging.info("Retraining on full dataset for final model...")
    
    # Determine best iteration (check for None, not just truthy since 0 is valid)
    if hasattr(model, 'best_iteration') and model.best_iteration is not None:
        best_iter = model.best_iteration + 5  # Add 5 for safety margin
        logging.info(f"  Using best_iteration={model.best_iteration} + 5 = {best_iter} trees")
    else:
        best_iter = 100  # Default if best_iteration not available
        logging.info(f"  Using default n_estimators={best_iter} (best_iteration not available)")
    
    model_full = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=best_iter,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        eval_metric="mlogloss",
        verbose=0,
    )
    model_full.fit(X, y_encoded)
    
    # Use full model for predictions
    model = model_full
    
    # Compute baseline as average predicted probabilities on training data
    # This is the gold-standard approach, independent of SHAP's buggy expected_value
    # for multiclass XGBoost 3.1+ models
    logging.info("Computing baseline from training data predictions...")
    training_probs = model.predict_proba(X)
    baseline_array = training_probs.mean(axis=0)  # Shape: (3,) for [A=0, D=1, H=2]
    
    # Convert to dict using explicit class order
    # LabelEncoder encodes alphabetically: A=0, D=1, H=2
    baseline = {
        "away": float(baseline_array[0]),   # Class 0 = A (Away)
        "draw": float(baseline_array[1]),   # Class 1 = D (Draw)
        "home": float(baseline_array[2]),   # Class 2 = H (Home)
    }
    
    logging.info(f"XGBoost model trained successfully for {league}")
    logging.info(f"Computed baseline from training data: Draw={baseline['draw']:.4f}, "
                f"Home={baseline['home']:.4f}, Away={baseline['away']:.4f}")
    
    return model, feature_names, baseline


def _load_cached_ml_model(cache_file: Path) -> tuple[Any, list[str] | None, str | None, dict[str, float] | None]:
    """Load cached ML model from file.
    
    Returns:
        Tuple of (model, feature_names, fingerprint, baseline) where baseline is {"draw": float, "home": float, "away": float}
    """
    if not XGBOOST_AVAILABLE:
        return None, None, None, None
    
    try:
        if not cache_file.exists():
            return None, None, None, None
        
        cache_data = json.loads(cache_file.read_text(encoding="utf-8"))
        fingerprint = cache_data.get("fingerprint")
        feature_names = cache_data.get("feature_names", [])
        model_path = cache_file.parent / cache_data.get("model_file", "")
        baseline = cache_data.get("baseline")  # Load baseline from cache (may be None for old caches)
        
        if not model_path.exists():
            return None, None, None, None
        
        model = pickle.loads(model_path.read_bytes())
        return model, feature_names, fingerprint, baseline
    except Exception as e:
        logging.warning(f"Could not load cached ML model: {e}")
        return None, None, None, None


def _save_cached_ml_model(cache_file: Path, model: Any, feature_names: list[str], fingerprint: str, baseline: dict[str, float] | None = None) -> None:
    """Save ML model with fingerprint and baseline to cache.
    
    Args:
        baseline: Baseline probabilities {"draw": float, "home": float, "away": float} 
                  computed from training data. Should match class order: [Draw=0, Home=1, Away=2]
    """
    if not XGBOOST_AVAILABLE:
        return
    
    try:
        model_file = cache_file.parent / f"{cache_file.stem}_model.pkl"
        model_bytes = pickle.dumps(model)
        model_file.write_bytes(model_bytes)
        
        cache_data = {
            "fingerprint": fingerprint,
            "feature_names": feature_names,
            "model_file": model_file.name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "class_order": CLASS_ORDER,  # Store class order for consistency
        }
        
        # Store baseline if provided (computed from training data)
        if baseline is not None:
            cache_data["baseline"] = baseline
            cache_data["baseline_source"] = "training_data"  # Mark where baseline comes from
        
        cache_file.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")
    except Exception as e:
        logging.warning(f"Could not save ML model cache: {e}")


def fit_ml_model(historical_data: dict, league: str, use_all_leagues: bool = True) -> tuple[Any, list[str] | None, dict[str, float] | None]:
    """
    Fit or load cached XGBoost model for a league.
    
    If use_all_leagues=True, trains on all available leagues' historical data combined.
    Otherwise, trains on the specified league's data only.
    
    Uses caching to avoid redundant retraining when historical data hasn't changed.
    
    Returns:
        Tuple of (model, feature_names, baseline) where baseline is {"draw": float, "home": float, "away": float}
        Baseline is computed from training data predictions (not from SHAP's buggy expected_value).
    """
    if not XGBOOST_AVAILABLE:
        return None, None, None
    
    # If training on all leagues, load all historical data
    if use_all_leagues:
        from scrapers.league_mapping import LEAGUE_CODE_TO_PATH
        
        # Combine historical data from all leagues in a consistent order
        # Sort league codes to ensure consistent dictionary key ordering
        all_league_codes = sorted(LEAGUE_CODE_TO_PATH.keys())
        combined_historical = {}
        
        # Always combine in sorted order, regardless of which league called this function
        for league_code in all_league_codes:
            try:
                if league_code == league:
                    # Use the provided historical_data for current league
                    for match_id, match_data in historical_data.items():
                        combined_historical[f"{league_code}_{match_id}"] = match_data
                else:
                    # Load from file for other leagues
                    hist_path = historical_path(league_code)
                    other_hist = load_json(hist_path, default={}) or {}
                    if isinstance(other_hist, dict):
                        # Prefix match IDs with league code to avoid collisions
                        for match_id, match_data in other_hist.items():
                            combined_historical[f"{league_code}_{match_id}"] = match_data
            except Exception as e:
                logging.debug(f"Could not load historical data for {league_code}: {e}")
                continue
        
        training_data = combined_historical
        cache_league_key = "all_leagues"  # Use single cache for all-leagues model
    else:
        training_data = historical_data
        cache_league_key = league
    
    # Create fingerprint from combined data
    data_fingerprint = _create_historical_data_fingerprint(training_data)
    
    # Check cache (using cache_league_key)
    cache_file = data_dir() / f"ml_model_cache_{cache_league_key}.json"
    cached_model, cached_features, cached_fp, cached_baseline = _load_cached_ml_model(cache_file)
    
    if cached_model and cached_features and cached_fp == data_fingerprint:
        logging.debug(f"Using cached ML model for {cache_league_key} (fingerprint: {cached_fp[:16]}...)")
        if cached_baseline:
            logging.info(f"Using training-data baseline for SHAP explanations: "
                        f"Draw={cached_baseline.get('draw', 0):.4f}, "
                        f"Home={cached_baseline.get('home', 0):.4f}, "
                        f"Away={cached_baseline.get('away', 0):.4f}")
        return cached_model, cached_features, cached_baseline
    
    # Need to train - log this only when actually training
    if use_all_leagues:
        logging.info(f"Training ML model on combined data from all leagues ({len(training_data)} total matches)")
    
    # Train new model
    model, feature_names, baseline = train_ml_model(training_data, cache_league_key)
    if model and feature_names and baseline:
        _save_cached_ml_model(cache_file, model, feature_names, data_fingerprint, baseline)
        logging.info(f"Trained and cached ML model for {cache_league_key} (fingerprint: {data_fingerprint[:16]}...)")
        logging.info(f"Using training-data baseline for SHAP explanations: "
                    f"Draw={baseline.get('draw', 0):.4f}, "
                    f"Home={baseline.get('home', 0):.4f}, "
                    f"Away={baseline.get('away', 0):.4f}")
        return model, feature_names, baseline
    
    return None, None, None


def predict_with_ml_model(
    home_name: str,
    away_name: str,
    home_form: list[MatchData],
    away_form: list[MatchData],
    team_h2h_index: dict | None,
    model: Any,
    feature_names: list[str] | None,
    reference_date: datetime | None = None,
    league: str | None = None,
) -> tuple[float, float, float] | None:
    """
    Predict H/D/A probabilities using trained ML model.
    
    Returns:
        Tuple of (p_home, p_draw, p_away) or None if prediction fails
    """
    if not XGBOOST_AVAILABLE or model is None or feature_names is None:
        return None
    
    if reference_date is None:
        reference_date = datetime.now()
    
    try:
        # Extract features
        features = extract_features_for_match(
            home_name, away_name,
            home_form, away_form,
            team_h2h_index, reference_date,
            league=league
        )
        
        # Convert to feature vector (same order as training)
        X = np.array([[features.get(f, 0.0) for f in feature_names]], dtype=np.float32)
        
        # Predict probabilities
        probs = model.predict_proba(X)[0]
        
        # Map to H/D/A
        # LabelEncoder maps labels alphabetically: A=0, D=1, H=2
        # XGBoost's model.classes_ contains the encoded integers [0, 1, 2]
        # predict_proba returns probabilities in encoded class order: [P(0), P(1), P(2)]
        # So: probs[0] = P(A=0=Away), probs[1] = P(D=1=Draw), probs[2] = P(H=2=Home)
        if len(probs) == 3:
            p_away = float(probs[0])  # Class 0 = A (Away)
            p_draw = float(probs[1])  # Class 1 = D (Draw)
            p_home = float(probs[2])  # Class 2 = H (Home)
        else:
            # Fallback if unexpected number of classes
            logging.warning(f"Unexpected number of classes in ML model: {len(probs)}")
            p_home = p_draw = p_away = 0.0
        
        return p_home, p_draw, p_away
    except Exception as e:
        logging.warning(f"ML model prediction failed: {e}")
        return None


def get_shap_explanations(
    model: Any,
    feature_names: list[str],
    features: dict[str, float],
    reference_date: datetime | None = None,
    baseline: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """
    Compute SHAP values for a single prediction to explain feature contributions.

    Args:
        baseline: Optional baseline probabilities {"draw": float, "home": float, "away": float}
                  computed from training data. If provided, this will be used instead of SHAP's
                  buggy expected_value (which is incorrect for multiclass XGBoost 3.1+).
                  Should match class order: [Draw=0, Home=1, Away=2]

    Returns:
        Dict with SHAP values per outcome class, or None if unavailable
    """
    if not XGBOOST_AVAILABLE or not SHAP_AVAILABLE or model is None or feature_names is None:
        return None
    
    try:
        # Convert feature dict to array (same order as training)
        X = np.array([[features.get(f, 0.0) for f in feature_names]], dtype=np.float32)
        
        # Fix base_score if it's stored as a string/list (known issue with XGBoost >= 3.1)
        # SHAP's TreeExplainer expects numeric base_score, not string/list representation
        # This is a workaround for: https://github.com/shap/shap/issues/4184
        # Try multiple approaches to fix base_score before TreeExplainer initialization
        base_score_fixed = False
        try:
            booster = model.get_booster()
            
            # Method 1: Try to access base_score from booster attributes directly
            try:
                attrs = booster.attributes
                if hasattr(attrs, 'get'):
                    base_score_attr = attrs.get('base_score', None)
                    if isinstance(base_score_attr, (list, str)):
                        logging.debug(f"base_score from attributes is {type(base_score_attr)}: {base_score_attr}")
            except Exception:
                pass
            
            # Method 2: Access via config JSON (more reliable path)
            config = booster.save_config()
            import json
            config_dict = json.loads(config)
            
            # Navigate to learner_model_param to check base_score
            learner = config_dict.get("learner", {})
            learner_model_param = learner.get("learner_model_param", {})
            base_score_val = learner_model_param.get("base_score", None)
            
            # Fix base_score if it's a string representation of a list/array
            if base_score_val and isinstance(base_score_val, str) and (base_score_val.startswith('[') or ',' in base_score_val):
                # Parse string like '[3.0560273E-1,2.5976232E-1,4.3463498E-1]' or '0.3,0.3,0.4'
                import re
                cleaned = base_score_val.strip('[]')
                try:
                    values = [float(x.strip()) for x in re.split(r',\s*', cleaned)]
                    # For multi-class, XGBoost uses base_score as a list, but TreeExplainer expects a single float
                    # Use average as a reasonable approximation (or first value)
                    fixed_base_score = str(sum(values) / len(values)) if len(values) > 0 else "0.0"
                    
                    # Try to set via attributes first (may not work but worth trying)
                    try:
                        booster.set_attr(base_score=fixed_base_score)
                        base_score_fixed = True
                    except Exception:
                        pass
                    
                    # Also update the config and reload
                    learner_model_param["base_score"] = fixed_base_score
                    learner["learner_model_param"] = learner_model_param
                    config_dict["learner"] = learner
                    booster.load_config(json.dumps(config_dict))
                    base_score_fixed = True
                    logging.debug(f"Fixed base_score from '{base_score_val}' to '{fixed_base_score}'")
                except (ValueError, TypeError) as parse_err:
                    logging.debug(f"Could not parse base_score string '{base_score_val}': {parse_err}")
        except Exception as fix_err:
            # If fixing fails, continue and let TreeExplainer try (it might work, or fallback will handle it)
            logging.debug(f"Could not fix base_score in model: {fix_err}")
        
        # Verify model is producing valid predictions for this single sample
        # Note: We can't check variance across samples here (only one sample)
        # But we can verify the model produces non-zero probabilities
        try:
            test_probs = model.predict_proba(X)
            if len(test_probs.shape) == 2 and test_probs.shape[0] > 0:
                probs = test_probs[0]  # Single sample
                prob_sum = np.sum(probs)
                if prob_sum < 0.99 or prob_sum > 1.01:
                    logging.debug(f"Model probabilities don't sum to 1: {prob_sum}")
                # Check if probabilities are all very similar (might indicate constant model)
                prob_range = np.max(probs) - np.min(probs)
                if prob_range < 0.01:
                    logging.debug(f"Model probabilities are very similar (range={prob_range:.4f})")
        except Exception as pred_check_err:
            logging.debug(f"Could not verify model predictions: {pred_check_err}")
        
        # Diagnostic: Check model feature importance and tree usage
        try:
            # Check if model is actually using features (XGBoost specific)
            if hasattr(model, 'get_booster'):
                booster = model.get_booster()
                # Get feature importance scores
                importance = booster.get_score(importance_type='gain')
                num_used_features = len(importance)
                total_gain = sum(importance.values()) if importance else 0.0
                if num_used_features == 0 or total_gain < 1e-10:
                    logging.warning(f"[SHAP Diagnostic] Model appears to have no feature importance (used features: {num_used_features}, total gain: {total_gain})")
                else:
                    logging.info(f"[SHAP Diagnostic] Model uses {num_used_features} features with total gain: {total_gain:.2f}")
        except Exception as imp_err:
            logging.warning(f"[SHAP Diagnostic] Could not check feature importance: {imp_err}")
        
        # Try TreeExplainer (more efficient for XGBoost)
        # For XGBoost multi-class, TreeExplainer should auto-detect probability output
        # Note: SHAP 0.49.1 (latest for Python 3.10) has a known bug with XGBoost 3.1+ base_score
        # The fix is in SHAP 0.50.0+ which requires Python 3.11+
        # If TreeExplainer fails, we gracefully return None (SHAP explanations disabled)
        explainer_type = "TreeExplainer"
        try:
            # For multi-class XGBoost, TreeExplainer should handle probabilities correctly
            # Try without explicit model_output first (TreeExplainer should auto-detect)
            explainer = shap.TreeExplainer(model)
            
            # Diagnostic: Check explainer's expected_value (baseline)
            try:
                ev = explainer.expected_value
                if hasattr(ev, '__len__'):
                    logging.info(f"[SHAP Diagnostic] TreeExplainer initialized successfully. Expected value (baseline): {ev}")
                else:
                    logging.info(f"[SHAP Diagnostic] TreeExplainer initialized successfully. Expected value (baseline): {ev}")
            except Exception as ev_err:
                logging.warning(f"[SHAP Diagnostic] Could not access TreeExplainer expected_value: {ev_err}")
                
        except (ValueError, TypeError) as e:
            error_str = str(e)
            logging.warning(f"[SHAP Diagnostic] TreeExplainer failed: {e}")
            if "base_score" in error_str or "could not convert string to float" in error_str:
                # TreeExplainer failed due to base_score issue
                # The fallback Explainer won't work well without background data
                # For now, return None - SHAP explanations are not available
                logging.warning(f"[SHAP Diagnostic] TreeExplainer failed due to base_score format. "
                              f"Fallback Explainer requires background data which we don't have. "
                              f"SHAP explanations will be disabled for this prediction.")
                return None
            else:
                # Some other error - re-raise it
                raise
        
        # Compute SHAP values (returns list of arrays: one per class for multi-class)
        # Each array is shape (1, n_features) - one sample, n_features contributions
        shap_values = explainer.shap_values(X)
        
        # Diagnostic: Log what shap_values actually is
        logging.info(f"[SHAP Diagnostic] shap_values type: {type(shap_values)}, len: {len(shap_values) if hasattr(shap_values, '__len__') else 'N/A'}")
        if hasattr(shap_values, 'shape'):
            logging.info(f"[SHAP Diagnostic] shap_values shape: {shap_values.shape}")
        
        # Debug: Log which explainer was used and check if values are non-zero
        if explainer_type == "Explainer (fallback)":
            logging.info(f"[SHAP Diagnostic] Using Explainer fallback - SHAP computation may not work correctly without background data")
        
        # Handle different SHAP output formats
        # TreeExplainer for multi-class: shap_values is a list [class0_shap, class1_shap, class2_shap]
        # Explainer wrapper: shap_values might be a single array with different shapes
        if isinstance(shap_values, np.ndarray):
            # Handle array format from Explainer wrapper
            if len(shap_values.shape) == 3:
                # Shape is (n_samples, n_classes, n_features) or (n_samples, n_features, n_classes)
                # Check which dimension is n_classes
                if shap_values.shape[1] == 3:
                    # Format: (n_samples=1, n_classes=3, n_features)
                    shap_values = [shap_values[0, i, :] for i in range(3)]  # Extract per-class arrays
                elif shap_values.shape[2] == 3:
                    # Format: (n_samples=1, n_features, n_classes=3)
                    shap_values = [shap_values[0, :, i] for i in range(3)]
                    logging.info(f"[SHAP Diagnostic] Converted from shape (1, {shap_values[0].shape[0]}, 3) to list of 3 arrays with shapes: {[sv.shape for sv in shap_values]}")
                else:
                    logging.debug(f"SHAP array shape {shap_values.shape} - trying to infer class dimension")
                    # Try to find dimension with size 3 (number of classes)
                    for dim in range(3):
                        if shap_values.shape[dim] == 3:
                            # Found class dimension, extract accordingly
                            if dim == 1:
                                shap_values = [shap_values[0, i, :] for i in range(3)]
                            elif dim == 2:
                                shap_values = [shap_values[0, :, i] for i in range(3)]
                            break
                    else:
                        logging.warning(f"Could not infer class dimension from SHAP array shape: {shap_values.shape}")
                        return None
            elif len(shap_values.shape) == 2:
                # 2D array - could be (n_features, n_classes) or (n_samples, n_features)
                if shap_values.shape[1] == 3:
                    # Format: (n_features, n_classes=3)
                    shap_values = [shap_values[:, i] for i in range(3)]
                elif shap_values.shape[0] == 1 and shap_values.shape[1] > 3:
                    # Format: (n_samples=1, n_features) - single class, but we need 3
                    # This shouldn't happen for multi-class, but handle gracefully
                    logging.warning(f"SHAP returned 2D array with shape {shap_values.shape} - expected 3 classes")
                    return None
                else:
                    logging.warning(f"Unexpected 2D SHAP array shape: {shap_values.shape}")
                    return None
            else:
                logging.warning(f"Unexpected SHAP array format: shape {shap_values.shape}, expected 2D or 3D array")
                return None
        elif isinstance(shap_values, list) and len(shap_values) >= 3:
            # Already in correct format: [class0_shap, class1_shap, class2_shap]
            # Verify each element is a numpy array with expected shape
            for idx, sv in enumerate(shap_values):
                if not isinstance(sv, np.ndarray):
                    logging.debug(f"shap_values[{idx}] is not a numpy array: {type(sv)}")
                elif len(sv.shape) not in [1, 2]:
                    logging.debug(f"shap_values[{idx}] has unexpected shape: {sv.shape}")
            pass  # Keep as-is
        else:
            logging.warning(f"Unexpected SHAP values format: expected list of 3 arrays or array, got {type(shap_values)}, shape={getattr(shap_values, 'shape', 'N/A')}")
            return None
        
        # Diagnostic: Log raw SHAP values statistics (after format conversion)
        try:
            if isinstance(shap_values, list) and len(shap_values) >= 3:
                # Check raw values after format conversion
                # LabelEncoder encodes: A=0, D=1, H=2
                raw_away = shap_values[0]  # Class 0 = A (Away)
                raw_draw = shap_values[1]  # Class 1 = D (Draw)
                raw_home = shap_values[2]  # Class 2 = H (Home)
                max_draw = np.max(np.abs(raw_draw)) if hasattr(raw_draw, '__iter__') else abs(float(raw_draw))
                max_home = np.max(np.abs(raw_home)) if hasattr(raw_home, '__iter__') else abs(float(raw_home))
                max_away = np.max(np.abs(raw_away)) if hasattr(raw_away, '__iter__') else abs(float(raw_away))
                logging.info(f"[SHAP Diagnostic] Raw SHAP values - max abs: draw={max_draw:.6f}, home={max_home:.6f}, away={max_away:.6f}")
                if max_draw < 1e-10 and max_home < 1e-10 and max_away < 1e-10:
                    logging.warning(f"[SHAP Diagnostic] All raw SHAP values are zero - TreeExplainer computed nothing")
        except Exception as diag_err:
            logging.warning(f"[SHAP Diagnostic] Could not analyze raw SHAP values: {diag_err}")
        
        # Get baseline (expected values)
        # Use provided baseline if available (computed from training data - gold standard)
        # Otherwise fall back to SHAP's expected_value (which may be buggy for multiclass XGBoost 3.1+)
        if baseline is not None:
            # Use the provided baseline (computed from training data)
            # Baseline dict keys match CLASS_NAMES: "draw", "home", "away"
            baseline_draw = baseline.get("draw", 0.33)
            baseline_home = baseline.get("home", 0.33)
            baseline_away = baseline.get("away", 0.33)
            logging.debug(f"Using training-data baseline: Draw={baseline_draw:.4f}, "
                         f"Home={baseline_home:.4f}, Away={baseline_away:.4f}")
        else:
            # Fallback to SHAP's expected_value (may be incorrect for multiclass XGBoost 3.1+)
            # SHAP's expected_value is buggy and returns uniform values [0.3056, 0.3056, 0.3056]
            # instead of the correct base_score [0.3056, 0.2598, 0.4346]
            try:
                expected_value = explainer.expected_value
            except Exception as e:
                logging.debug(f"Error accessing explainer.expected_value: {e}")
                expected_value = None
            
            # Handle different expected_value formats
            baseline_home = baseline_draw = baseline_away = 0.33  # Default
            
            # Handle expected_value - check type first to avoid errors
            if expected_value is None:
                # Use default if expected_value is None
                pass  # Already set to defaults
            elif isinstance(expected_value, np.ndarray) and len(expected_value) >= 3:
                try:
                    # Map according to class order: [A=0, D=1, H=2] (alphabetical)
                    baseline_away = float(expected_value[0])  # Class 0 = A (Away)
                    baseline_draw = float(expected_value[1])  # Class 1 = D (Draw)
                    baseline_home = float(expected_value[2])  # Class 2 = H (Home)
                except (ValueError, TypeError, IndexError) as e:
                    logging.debug(f"Error converting numpy array expected_value to float: {e}")
            elif isinstance(expected_value, (list, tuple)) and len(expected_value) >= 3:
                # Handle list/tuple format
                try:
                    baseline_away = float(expected_value[0])
                    baseline_draw = float(expected_value[1])
                    baseline_home = float(expected_value[2])
                except (ValueError, TypeError, IndexError) as e:
                    logging.debug(f"Error converting list/tuple expected_value to float: {e}")
            elif isinstance(expected_value, str):
                # Handle string representation (e.g., '[0.3, 0.3, 0.4]' or '[3.0560273E-1,2.5976232E-1,4.3463498E-1]')
                try:
                    import re
                    # Remove brackets and split by comma
                    # Handle formats like '[3.0560273E-1,2.5976232E-1,4.3463498E-1]'
                    cleaned = expected_value.strip('[]')
                    # Split by comma, handling scientific notation
                    values = [float(x.strip()) for x in re.split(r',\s*', cleaned)]
                    if len(values) >= 3:
                        # Map according to class order: [A=0, D=1, H=2] (alphabetical)
                        baseline_away = values[0]
                        baseline_draw = values[1]
                        baseline_home = values[2]
                except (ValueError, TypeError, IndexError) as e:
                    logging.debug(f"Could not parse expected_value string '{expected_value}': {e}")
            else:
                logging.debug(f"Unexpected expected_value format: {type(expected_value)}, value: {repr(expected_value)[:100]}")
            
            logging.warning("Using SHAP's expected_value as baseline (may be incorrect for multiclass XGBoost 3.1+). "
                          "Consider providing baseline computed from training data.")
        
        # Format SHAP values per class
        # LabelEncoder encodes alphabetically: A=0, D=1, H=2
        # shap_values[0] = contributions to Away (class 0 = A)
        # shap_values[1] = contributions to Draw (class 1 = D)
        # shap_values[2] = contributions to Home (class 2 = H)
        explanations = {
            "baseline": {
                "home": baseline_home,
                "draw": baseline_draw,
                "away": baseline_away,
            },
            "home": {},
            "draw": {},
            "away": {},
        }
        
        # Extract contributions per feature for each class
        # shap_values[i] should be shape (1, n_features) or (n_features,) for a single prediction
        # Verify shapes before extraction
        if len(shap_values) < 3:
            logging.warning(f"shap_values has {len(shap_values)} elements, expected 3 classes")
            return None
        
        # Get the actual shape of the first class array to determine indexing
        # LabelEncoder encodes: A=0, D=1, H=2
        away_shape = shap_values[0].shape  # Class 0 = A (Away)
        draw_shape = shap_values[1].shape  # Class 1 = D (Draw)
        home_shape = shap_values[2].shape  # Class 2 = H (Home)
        
        # Determine number of features from the array shape
        if len(draw_shape) == 2:
            # Shape is (n_samples, n_features) - typically (1, n_features) for single prediction
            n_features = draw_shape[1]
            feature_idx_offset = 0  # Features are in dimension 1
        elif len(draw_shape) == 1:
            # Shape is (n_features,) - flat array
            n_features = draw_shape[0]
            feature_idx_offset = None  # No sample dimension
        else:
            logging.warning(f"Unexpected shap_values shape: away={away_shape}, draw={draw_shape}, home={home_shape}")
            return None
        
        # Extract feature contributions
        # Debug: Log sample values to verify SHAP is working - check first few values
        try:
            # Try to get first feature contribution from each class
            # LabelEncoder encodes: A=0, D=1, H=2
            if feature_idx_offset is None:
                # 1D array
                sample_away = float(shap_values[0][0]) if len(shap_values[0]) > 0 else 0.0  # Class 0 = A
                sample_draw = float(shap_values[1][0]) if len(shap_values[1]) > 0 else 0.0  # Class 1 = D
                sample_home = float(shap_values[2][0]) if len(shap_values[2]) > 0 else 0.0  # Class 2 = H
            else:
                # 2D array
                sample_away = float(shap_values[0][0][0]) if shap_values[0].shape[0] > 0 and shap_values[0].shape[1] > 0 else 0.0  # Class 0 = A
                sample_draw = float(shap_values[1][0][0]) if shap_values[1].shape[0] > 0 and shap_values[1].shape[1] > 0 else 0.0  # Class 1 = D
                sample_home = float(shap_values[2][0][0]) if shap_values[2].shape[0] > 0 and shap_values[2].shape[1] > 0 else 0.0  # Class 2 = H
            
            # Check if all values are essentially zero
            if abs(sample_draw) < 1e-10 and abs(sample_home) < 1e-10 and abs(sample_away) < 1e-10:
                logging.warning(f"All SHAP values appear to be zero (sample: draw={sample_draw}, home={sample_home}, away={sample_away}). "
                              f"Shapes: draw={draw_shape}, home={home_shape}, away={away_shape}, n_features={n_features}, "
                              f"feature_names_count={len(feature_names)}, feature_idx_offset={feature_idx_offset}")
        except Exception as debug_err:
            logging.debug(f"Error checking sample SHAP values: {debug_err}")
        
        num_features_extracted = 0
        for i, feature in enumerate(feature_names):
            if i >= n_features:
                break  # Skip if we've exhausted the SHAP array
            
            # Extract SHAP value based on array shape
            # LabelEncoder encodes: A=0, D=1, H=2
            try:
                if feature_idx_offset is None:
                    # Flat array: shap_values[class][feature_idx]
                    away_val = shap_values[0][i]  # Class 0 = A (Away)
                    draw_val = shap_values[1][i]  # Class 1 = D (Draw)
                    home_val = shap_values[2][i]  # Class 2 = H (Home)
                else:
                    # 2D array: shap_values[class][sample_idx][feature_idx]
                    away_val = shap_values[0][0][i]  # Class 0 = A (Away)
                    draw_val = shap_values[1][0][i]  # Class 1 = D (Draw)
                    home_val = shap_values[2][0][i]  # Class 2 = H (Home)
                
                explanations["away"][feature] = float(away_val)
                explanations["draw"][feature] = float(draw_val)
                explanations["home"][feature] = float(home_val)
                num_features_extracted += 1
            except (IndexError, TypeError) as e:
                logging.debug(f"Error extracting SHAP value for feature {i} ({feature}): {e}, shapes: draw={draw_shape}, home={home_shape}, away={away_shape}")
                continue
        
        logging.info(f"[SHAP Diagnostic] Extracted {num_features_extracted} features from SHAP values (n_features={n_features}, num_feature_names={len(feature_names)})")
        if num_features_extracted == 0:
            logging.warning(f"No features extracted from SHAP values. Shapes: draw={draw_shape}, home={home_shape}, away={away_shape}, n_features={n_features}, num_feature_names={len(feature_names)}")
            return None
        
        # Verify we extracted non-zero values (if all are zero, something is wrong)
        max_abs_draw = max(abs(v) for v in explanations["draw"].values()) if explanations["draw"] else 0.0
        max_abs_home = max(abs(v) for v in explanations["home"].values()) if explanations["home"] else 0.0
        max_abs_away = max(abs(v) for v in explanations["away"].values()) if explanations["away"] else 0.0
        
        if max_abs_draw < 1e-10 and max_abs_home < 1e-10 and max_abs_away < 1e-10:
            logging.warning(f"All SHAP values are zero after extraction. This may indicate: "
                          f"1) Model makes constant predictions, 2) base_score fix didn't work, "
                          f"3) Wrong explainer configuration. Max abs values: draw={max_abs_draw}, home={max_abs_home}, away={max_abs_away}")
            # Return None instead of empty explanations - this signals the issue to the caller
            return None
        
        # Compute top contributing features (by absolute value) for each outcome
        def get_top_features(class_name: str, n: int = 10) -> list[dict[str, Any]]:
            class_contribs = explanations[class_name]
            sorted_features = sorted(
                class_contribs.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:n]
            return [
                {
                    "feature": feat,
                    "contribution": round(float(contrib), 4),
                    "direction": "positive" if contrib > 0 else "negative",
                }
                for feat, contrib in sorted_features
            ]
        
        explanations["top_features"] = {
            "home": get_top_features("home"),
            "draw": get_top_features("draw"),
            "away": get_top_features("away"),
        }
        
        logging.info(f"[SHAP Diagnostic] Successfully created SHAP explanations with {len(explanations.get('home', {}))} home features, {len(explanations.get('draw', {}))} draw features, {len(explanations.get('away', {}))} away features")
        return explanations
    except Exception as e:
        # Log more details to help diagnose issues
        import traceback
        error_msg = str(e)
        # Always show traceback for float conversion errors to diagnose the issue
        if "could not convert string to float" in error_msg:
            logging.warning(f"SHAP explanation computation failed: {e}")
            logging.warning(f"SHAP error traceback:\n{traceback.format_exc()}")
        else:
            logging.warning(f"SHAP explanation computation failed: {e}")
            logging.debug(f"SHAP error traceback: {traceback.format_exc()}")
        return None

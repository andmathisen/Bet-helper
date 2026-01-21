"""
Utility script for offline model evaluation.

Usage:
    poetry run python -m bet_helper.predict.eval --league PL --splits 5
    poetry run python -m bet_helper.predict.eval --all --splits 5 --grid medium

It builds time-aware features (only past matches) via prepare_training_data_from_historical,
then runs time-series cross validation with a small XGBoost hyperparameter grid and reports
log loss, Brier score, and accuracy per candidate.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
except Exception as e:  # pragma: no cover
    raise SystemExit(f"XGBoost is required for evaluation: {e}")

from bet_helper.predict.ml_model import prepare_training_data_from_historical, CLASS_ORDER
from bet_helper.storage import historical_path, load_json, save_json

try:
    from scrapers.league_mapping import LEAGUE_CODE_TO_PATH
except Exception:
    LEAGUE_CODE_TO_PATH = {}


def _brier_multi(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Multi-class Brier score (mean squared error on probability vectors)."""
    y_onehot = np.zeros_like(prob)
    y_onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((prob - y_onehot) ** 2, axis=1)))


def _load_historical(leagues: Iterable[str]) -> tuple[dict, list[str]]:
    combined: dict = {}
    missing: list[str] = []
    for lg in leagues:
        path = historical_path(lg)
        data = load_json(path, default={}) or {}
        if not data:
            missing.append(lg)
            continue
        for match_id, match_data in data.items():
            combined[f"{lg}_{match_id}"] = match_data
    return combined, missing


def _build_matrix(features_list: list[dict[str, float]]):
    feature_names = sorted(features_list[0].keys())
    X = np.array([[f.get(name, 0.0) for name in feature_names] for f in features_list], dtype=np.float32)
    return X, feature_names


def _param_grid(name: str):
    """Return a small hyperparameter grid."""
    grids = {
        "tiny": [
            {"name": "baseline", "n_estimators": 120, "max_depth": 4, "learning_rate": 0.1, 
             "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "gamma": 0.1,
             "reg_alpha": 1.0, "reg_lambda": 1.0},  # Regularization to match production
        ],
        "medium": [
            {"name": "baseline", "n_estimators": 120, "max_depth": 4, "learning_rate": 0.1, 
             "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "gamma": 0.1,
             "reg_alpha": 1.0, "reg_lambda": 1.0},
            {"name": "deeper", "n_estimators": 160, "max_depth": 5, "learning_rate": 0.08, 
             "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 4, "gamma": 0.15,
             "reg_alpha": 1.0, "reg_lambda": 1.0},
            {"name": "shallow_fast", "n_estimators": 200, "max_depth": 3, "learning_rate": 0.06, 
             "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 2, "gamma": 0.0,
             "reg_alpha": 1.0, "reg_lambda": 1.0},
            {"name": "strong_reg", "n_estimators": 120, "max_depth": 4, "learning_rate": 0.1,
             "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "gamma": 0.1,
             "reg_alpha": 2.0, "reg_lambda": 2.0},  # Test stronger regularization
        ],
    }
    return grids.get(name, grids["tiny"])


def evaluate(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    splits: int,
    random_state: int = 42,
) -> dict[str, float]:
    """Run time-series CV for one hyperparameter set."""
    logging.info(f"  Starting time-series CV with {splits} splits...")
    tscv = TimeSeriesSplit(n_splits=splits)
    log_losses: list[float] = []
    briers: list[float] = []
    accs: list[float] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        logging.info(f"  Training fold {fold}/{splits} (train={len(train_idx)}, test={len(test_idx)})...")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=random_state,
            **{k: v for k, v in params.items() if k != "name"},
        )
        logging.debug(f"    Training model with params: {', '.join(f'{k}={v}' for k, v in params.items() if k != 'name')}")
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        prob = model.predict_proba(X_test)
        log_losses.append(log_loss(y_test, prob, labels=[0, 1, 2]))
        briers.append(_brier_multi(y_test, prob))
        accs.append(accuracy_score(y_test, prob.argmax(axis=1)))

        logging.info(
            f"  Fold {fold}: logloss={log_losses[-1]:.4f} "
            f"brier={briers[-1]:.4f} acc={accs[-1]:.3f} "
            f"(train={len(train_idx)}, test={len(test_idx)})"
        )

    return {
        "logloss": float(np.mean(log_losses)),
        "brier": float(np.mean(briers)),
        "accuracy": float(np.mean(accs)),
        "folds": len(log_losses),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Bet Helper ML model with time-series CV.")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--league", action="append", dest="leagues", help="League code(s) to evaluate (can repeat).")
    grp.add_argument("--all", action="store_true", help="Use all leagues with available historical data.")
    parser.add_argument("--splits", type=int, default=5, help="Number of time-series CV splits (default: 5).")
    parser.add_argument("--grid", choices=["tiny", "medium"], default="medium", help="Hyperparameter grid preset.")
    parser.add_argument("--min-matches", type=int, default=150, help="Minimum matches required to run evaluation.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG).")
    parser.add_argument("--output", type=str, help="Path to save evaluation results JSON (optional).")
    args = parser.parse_args()

    # Configure logging - ensure it outputs to stderr/stdout
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True  # Force reconfiguration in case logging was already set up
    )
    
    logging.info("="*80)
    logging.info("Bet Helper ML Model Evaluation")
    logging.info("="*80)

    # Resolve league list
    if args.all:
        leagues = sorted(LEAGUE_CODE_TO_PATH.keys()) if LEAGUE_CODE_TO_PATH else []
        if not leagues:
            logging.error("No leagues found in league_mapping; provide --league codes instead.")
            raise SystemExit(1)
    else:
        leagues = args.leagues or []

    if not leagues:
        logging.error("No leagues specified.")
        raise SystemExit(1)
        
    logging.info(f"Evaluating leagues: {', '.join(leagues)}")

    logging.info("Loading historical data...")
    historical, missing = _load_historical(leagues)
    if missing:
        logging.warning(f"No historical data found for: {', '.join(missing)}")
    if not historical:
        raise SystemExit("No historical data available. Run the scraper to populate data/historical_matches_<league>.json")
    
    logging.info(f"Loaded {len(historical)} historical matches from {len(leagues) - len(missing)} leagues")

    logging.info("Preparing training data from historical matches...")
    features_list, labels_list = prepare_training_data_from_historical(
        historical, min_matches=args.min_matches, min_team_matches=5
    )
    if len(features_list) < args.min_matches:
        raise SystemExit(f"Not enough samples after preprocessing ({len(features_list)} < {args.min_matches}).")

    X, feature_names = _build_matrix(features_list)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels_list)

    if list(label_encoder.classes_) != CLASS_ORDER:
        logging.warning(f"Label order unexpected: {label_encoder.classes_}. Expected {CLASS_ORDER}.")

    logging.info(f"Dataset ready: X={X.shape}, labels={np.bincount(y).tolist()} (order A,D,H).")
    logging.info(f"Using {len(feature_names)} features. First 5: {feature_names[:5]}")

    candidates = _param_grid(args.grid)
    logging.info(f"\n=== Starting hyperparameter evaluation ===")
    logging.info(f"Testing {len(candidates)} parameter configuration(s) with {args.splits}-fold time-series CV")
    
    results = []
    for idx, params in enumerate(candidates, start=1):
        name = params.get("name", "candidate")
        logging.info(f"\n[{idx}/{len(candidates)}] === Evaluating '{name}' params ===")
        metrics = evaluate(X, y, params, splits=args.splits)
        metrics["name"] = name
        results.append(metrics)
        logging.info(f"  ✓ Completed '{name}': logloss={metrics['logloss']:.4f}, brier={metrics['brier']:.4f}, acc={metrics['accuracy']:.3f}")

    # Sort results by logloss (best first)
    sorted_results = sorted(results, key=lambda r: r["logloss"])
    
    # Print summary table
    logging.info("\n" + "="*80)
    logging.info("EVALUATION SUMMARY (lower logloss/brier is better, higher accuracy is better)")
    logging.info("="*80)
    for m in sorted_results:
        logging.info(f"  {m['name']:15} logloss={m['logloss']:.4f}  brier={m['brier']:.4f}  acc={m['accuracy']:.3f}  (folds={m['folds']})")
    logging.info("="*80)
    
    # Save results to file if requested
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "leagues": leagues,
            "splits": args.splits,
            "grid": args.grid,
            "min_matches": args.min_matches,
            "dataset_size": len(features_list),
            "feature_count": len(feature_names),
            "label_distribution": np.bincount(y).tolist(),
            "results": sorted_results,
        }
        save_json(output_path, output_data)
        logging.info(f"\n✓ Evaluation results saved to: {output_path}")
    else:
        # Default: save to data/eval_results.json with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_output = Path("data") / f"eval_results_{timestamp}.json"
        default_output.parent.mkdir(exist_ok=True)
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "leagues": leagues,
            "splits": args.splits,
            "grid": args.grid,
            "min_matches": args.min_matches,
            "dataset_size": len(features_list),
            "feature_count": len(feature_names),
            "label_distribution": np.bincount(y).tolist(),
            "results": sorted_results,
        }
        save_json(default_output, output_data)
        logging.info(f"\n✓ Evaluation results saved to: {default_output}")


if __name__ == "__main__":
    main()

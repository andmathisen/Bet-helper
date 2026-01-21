"""
Evaluation script for trained RL betting agent.

Usage:
    python -m bet_helper.rl.eval --model-path data/rl_models/betting_agent_20260120_120000
    python -m bet_helper.rl.eval --model-path data/rl_models/betting_agent --n-episodes 50
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

try:
    from stable_baselines3 import PPO
    import numpy as np
    STABLE_BASELINES3_AVAILABLE = True
except ImportError:
    STABLE_BASELINES3_AVAILABLE = False
    PPO = None

from bet_helper.rl.data import load_match_dates, load_historical_data
from bet_helper.rl.env import BettingEnv
from bet_helper.storage import data_dir, save_json


def evaluate_agent(
    model_path: Path,
    leagues: List[str],
    historical_data: Dict,
    n_episodes: int = 20,
    initial_tokens: float = 1000.0,
    max_matches_per_bet: int = 5,
    use_ml_predictions: bool = True,
) -> Dict:
    """
    Evaluate trained agent on test data.
    
    Returns:
        Dictionary with evaluation metrics
    """
    if not STABLE_BASELINES3_AVAILABLE:
        raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3")
    
    # Load model
    logging.info(f"Loading model from {model_path}...")
    model = PPO.load(str(model_path))
    
    # Load test data (use all available dates for evaluation)
    match_dates = load_match_dates(
        leagues,
        use_predictions=True,
        historical_data=historical_data,
    )
    
    if not match_dates:
        raise ValueError(f"No match dates found for leagues: {leagues}")
    
    logging.info(f"Evaluating on {len(match_dates)} match dates")
    
    # Create environment
    env = BettingEnv(
        match_dates=match_dates,
        historical_data=historical_data,
        initial_tokens=initial_tokens,
        max_matches_per_bet=max_matches_per_bet,
        use_ml_predictions=use_ml_predictions,
    )
    
    # Run evaluation episodes
    episode_results = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_rewards = []
        episode_info = {
            "episode": episode,
            "bets": [],
            "initial_tokens": initial_tokens,
            "final_tokens": initial_tokens,
            "total_profit": 0.0,
            "total_bets": 0,
            "wins": 0,
            "dates_processed": 0,
        }
        
        while not done:
            # Get action from trained agent
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_rewards.append(reward)
            episode_info["total_profit"] += reward * initial_tokens  # Denormalize
            
            # Track bet if action was a bet
            if info.get("action") in ["single", "accumulator"]:
                bet_info = {
                    "date_idx": info.get("date_idx", 0),
                    "current_date": info.get("current_date"),
                    "action": info.get("action"),
                    "result": info.get("result"),
                    "tokens_after": info.get("tokens", initial_tokens),
                }
                if "winnings" in info:
                    bet_info["winnings"] = info["winnings"]
                if "profit" in info:
                    bet_info["profit"] = info["profit"]
                if "loss" in info:
                    bet_info["loss"] = info["loss"]
                
                episode_info["bets"].append(bet_info)
                episode_info["total_bets"] += 1
                if info.get("result") == "win":
                    episode_info["wins"] += 1
            
            episode_info["dates_processed"] = info.get("date_idx", 0)
            episode_info["final_tokens"] = info.get("tokens", initial_tokens)
        
        episode_info["total_return"] = sum(episode_rewards)
        episode_info["avg_reward"] = np.mean(episode_rewards) if episode_rewards else 0.0
        episode_info["win_rate"] = episode_info["wins"] / max(episode_info["total_bets"], 1)
        episode_info["roi"] = (episode_info["final_tokens"] - initial_tokens) / initial_tokens
        
        episode_results.append(episode_info)
        
        if (episode + 1) % 5 == 0:
            logging.info(f"Completed episode {episode + 1}/{n_episodes}")
    
    # Aggregate statistics
    final_tokens_list = [r["final_tokens"] for r in episode_results]
    profits_list = [r["total_profit"] for r in episode_results]
    win_rates_list = [r["win_rate"] for r in episode_results]
    rois_list = [r["roi"] for r in episode_results]
    total_bets_list = [r["total_bets"] for r in episode_results]
    
    metrics = {
        "model_path": str(model_path),
        "leagues": leagues,
        "n_episodes": n_episodes,
        "initial_tokens": initial_tokens,
        "use_ml_predictions": use_ml_predictions,
        "n_match_dates": len(match_dates),
        "aggregate_stats": {
            "mean_final_tokens": float(np.mean(final_tokens_list)),
            "std_final_tokens": float(np.std(final_tokens_list)),
            "min_final_tokens": float(np.min(final_tokens_list)),
            "max_final_tokens": float(np.max(final_tokens_list)),
            "mean_total_profit": float(np.mean(profits_list)),
            "mean_win_rate": float(np.mean(win_rates_list)),
            "mean_roi": float(np.mean(rois_list)),
            "mean_total_bets": float(np.mean(total_bets_list)),
            "total_wins": sum(r["wins"] for r in episode_results),
            "total_bets": sum(r["total_bets"] for r in episode_results),
            "overall_win_rate": sum(r["wins"] for r in episode_results) / max(sum(r["total_bets"] for r in episode_results), 1),
        },
        "episode_results": episode_results,
    }
    
    return metrics


def print_summary(metrics: Dict) -> None:
    """Print evaluation summary."""
    stats = metrics["aggregate_stats"]
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Model: {metrics['model_path']}")
    print(f"Leagues: {', '.join(metrics['leagues'])}")
    print(f"Episodes: {metrics['n_episodes']}")
    print(f"Match dates: {metrics['n_match_dates']}")
    print(f"Initial tokens: {metrics['initial_tokens']:.2f}")
    print()
    print("Performance Metrics:")
    print(f"  Mean final tokens: {stats['mean_final_tokens']:.2f} (±{stats['std_final_tokens']:.2f})")
    print(f"  Range: [{stats['min_final_tokens']:.2f}, {stats['max_final_tokens']:.2f}]")
    print(f"  Mean total profit: {stats['mean_total_profit']:.2f}")
    print(f"  Mean ROI: {stats['mean_roi']*100:.2f}%")
    print(f"  Mean win rate: {stats['mean_win_rate']*100:.2f}%")
    print(f"  Overall win rate: {stats['overall_win_rate']*100:.2f}% ({stats['total_wins']}/{stats['total_bets']} bets)")
    print(f"  Mean bets per episode: {stats['mean_total_bets']:.1f}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL betting agent")
    
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    
    # League selection (should match training, but can be different for testing)
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--leagues", nargs="+", help="League codes to evaluate on")
    grp.add_argument("--all", action="store_true", help="Use all available leagues")
    
    parser.add_argument("--n-episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--initial-tokens", type=float, default=1000.0, help="Starting token balance")
    parser.add_argument("--max-matches-per-bet", type=int, default=5, help="Max matches in accumulator")
    parser.add_argument("--no-ml-predictions", action="store_true", help="Disable ML predictions")
    parser.add_argument("--output", type=str, help="Path to save evaluation results JSON")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Resolve leagues
    if args.all:
        try:
            from scrapers.league_mapping import LEAGUE_CODE_TO_PATH
            leagues = sorted(LEAGUE_CODE_TO_PATH.keys())
        except Exception:
            logging.error("Could not load league mapping. Specify --leagues instead.")
            return 1
    else:
        leagues = args.leagues
    
    # Load historical data
    logging.info("Loading historical data...")
    historical_data = load_historical_data(leagues)
    
    # Evaluate
    logging.info(f"Evaluating agent on {len(leagues)} leagues...")
    metrics = evaluate_agent(
        model_path=Path(args.model_path),
        leagues=leagues,
        historical_data=historical_data,
        n_episodes=args.n_episodes,
        initial_tokens=args.initial_tokens,
        max_matches_per_bet=args.max_matches_per_bet,
        use_ml_predictions=not args.no_ml_predictions,
    )
    
    # Print summary
    print_summary(metrics)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        save_json(output_path, metrics)
        logging.info(f"✓ Evaluation results saved to {output_path}")
    else:
        # Auto-save to data/rl_eval_results_TIMESTAMP.json
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = data_dir() / f"rl_eval_results_{timestamp}.json"
        save_json(output_path, metrics)
        logging.info(f"✓ Evaluation results saved to {output_path}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

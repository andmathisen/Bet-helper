"""
Training script for RL betting agent using Stable-Baselines3 PPO.

Usage:
    python -m bet_helper.rl.train --leagues PL SerieA --initial-tokens 1000 --total-timesteps 100000
    python -m bet_helper.rl.train --all --total-timesteps 500000 --save-path models/betting_agent
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import (
        CheckpointCallback,
        EvalCallback,
        CallbackList,
    )
    from stable_baselines3.common.monitor import Monitor
    STABLE_BASELINES3_AVAILABLE = True
except ImportError:
    STABLE_BASELINES3_AVAILABLE = False
    PPO = None

from bet_helper.rl.data import load_match_dates, load_historical_data
from bet_helper.rl.env import BettingEnv
from bet_helper.storage import data_dir


def create_training_env(
    leagues: list[str],
    historical_data: dict,
    initial_tokens: float = 1000.0,
    max_matches_per_bet: int = 5,
    use_ml_predictions: bool = True,
    split_ratio: float = 0.8,
) -> tuple[BettingEnv, BettingEnv]:
    """
    Create training and validation environments.
    
    Args:
        leagues: List of league codes
        historical_data: Historical match data
        initial_tokens: Starting token balance
        max_matches_per_bet: Max matches in accumulator
        use_ml_predictions: Whether to use ML model predictions
        split_ratio: Ratio of dates for training (rest for validation)
    
    Returns:
        Tuple of (train_env, eval_env)
    """
    # Load match dates from historical data (has dates and outcomes for training)
    match_dates = load_match_dates(
        leagues,
        use_historical=True,  # Use historical matches for training (has dates and outcomes)
        use_predictions=False,
        historical_data=historical_data,
    )
    
    if not match_dates:
        raise ValueError(f"No match dates found for leagues: {leagues}")
    
    # Split into train/eval
    split_idx = int(len(match_dates) * split_ratio)
    train_dates = match_dates[:split_idx]
    eval_dates = match_dates[split_idx:]
    
    logging.info(f"Split data: {len(train_dates)} dates for training, {len(eval_dates)} for evaluation")
    
    # Create environments
    train_env = BettingEnv(
        match_dates=train_dates,
        historical_data=historical_data,
        initial_tokens=initial_tokens,
        max_matches_per_bet=max_matches_per_bet,
        use_ml_predictions=use_ml_predictions,
    )
    
    eval_env = BettingEnv(
        match_dates=eval_dates,
        historical_data=historical_data,
        initial_tokens=initial_tokens,
        max_matches_per_bet=max_matches_per_bet,
        use_ml_predictions=use_ml_predictions,
    )
    
    return train_env, eval_env


def train_agent(
    train_env: BettingEnv,
    eval_env: BettingEnv,
    total_timesteps: int = 100000,
    save_path: Optional[Path] = None,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    tensorboard_log: Optional[Path] = None,
    checkpoint_freq: int = 10000,
    eval_freq: int = 5000,
    n_eval_episodes: int = 10,
) -> PPO:
    """
    Train PPO agent on betting environment.
    
    Args:
        train_env: Training environment
        eval_env: Evaluation environment
        total_timesteps: Total training timesteps
        save_path: Path to save final model
        learning_rate: Learning rate
        n_steps: Steps per update
        batch_size: Batch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Max gradient norm for clipping
        tensorboard_log: Path for tensorboard logs
        checkpoint_freq: Frequency to save checkpoints
        eval_freq: Frequency to evaluate
        n_eval_episodes: Number of episodes per evaluation
    
    Returns:
        Trained PPO model
    """
    if not STABLE_BASELINES3_AVAILABLE:
        raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3")
    
    # Wrap env in Monitor for stats
    train_env = Monitor(train_env)
    eval_env = Monitor(eval_env)
    
    # Create PPO model
    logging.info(f"Creating PPO agent with learning_rate={learning_rate}, n_steps={n_steps}")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,
        tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
    )
    
    # Create callbacks
    callbacks = []
    
    # Checkpoint callback
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=save_path.parent / "checkpoints",
            name_prefix="betting_agent",
        )
        callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path.parent / "best_model" if save_path else None,
        log_path=save_path.parent / "eval_logs" if save_path else None,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)
    
    callback_list = CallbackList(callbacks) if callbacks else None
    
    # Train
    logging.info(f"Starting training for {total_timesteps} timesteps...")
    train_dates = getattr(getattr(train_env, "unwrapped", train_env), "match_dates", None)
    eval_dates = getattr(getattr(eval_env, "unwrapped", eval_env), "match_dates", None)
    logging.info(
        f"  Training dates: {len(train_dates) if train_dates is not None else 'unknown'}"
    )
    logging.info(
        f"  Evaluation dates: {len(eval_dates) if eval_dates is not None else 'unknown'}"
    )
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        progress_bar=True,
    )
    
    # Save final model
    if save_path:
        model.save(str(save_path))
        logging.info(f"✓ Model saved to {save_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train RL betting agent with PPO")
    
    # League selection
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--leagues", nargs="+", help="League codes to use (e.g., PL SerieA)")
    grp.add_argument("--all", action="store_true", help="Use all available leagues")
    
    # Training parameters
    parser.add_argument("--total-timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--initial-tokens", type=float, default=1000.0, help="Starting token balance")
    parser.add_argument("--max-matches-per-bet", type=int, default=5, help="Max matches in accumulator")
    parser.add_argument("--no-ml-predictions", action="store_true", help="Disable ML model predictions (use raw features only)")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train/eval split ratio")
    
    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per update")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="Epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    
    # Paths
    parser.add_argument("--save-path", type=str, help="Path to save model (default: data/rl_models/betting_agent_TIMESTAMP)")
    parser.add_argument("--tensorboard-log", type=str, help="Path for tensorboard logs (default: data/rl_logs)")
    
    # Callback frequencies
    parser.add_argument("--checkpoint-freq", type=int, default=10000, help="Checkpoint frequency")
    parser.add_argument("--eval-freq", type=int, default=5000, help="Evaluation frequency")
    parser.add_argument("--n-eval-episodes", type=int, default=10, help="Episodes per evaluation")
    
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
    
    logging.info(f"Training on leagues: {', '.join(leagues)}")
    
    # Load historical data
    logging.info("Loading historical data...")
    historical_data = load_historical_data(leagues)
    logging.info(f"Loaded {len(historical_data)} historical matches")
    
    # Create save path
    if not args.save_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = data_dir() / "rl_models" / f"betting_agent_{timestamp}"
    else:
        save_path = Path(args.save_path)
    
    # Create tensorboard log path
    if not args.tensorboard_log:
        tensorboard_log = data_dir() / "rl_logs"
    else:
        tensorboard_log = Path(args.tensorboard_log)
    
    # Create environments
    logging.info("Creating training and evaluation environments...")
    train_env, eval_env = create_training_env(
        leagues=leagues,
        historical_data=historical_data,
        initial_tokens=args.initial_tokens,
        max_matches_per_bet=args.max_matches_per_bet,
        use_ml_predictions=not args.no_ml_predictions,
        split_ratio=args.split_ratio,
    )
    
    # Train
    model = train_agent(
        train_env=train_env,
        eval_env=eval_env,
        total_timesteps=args.total_timesteps,
        save_path=save_path,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        tensorboard_log=tensorboard_log,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
    )
    
    logging.info("✓ Training completed successfully!")
    logging.info(f"  Model saved to: {save_path}")
    if tensorboard_log:
        logging.info(f"  Tensorboard logs: {tensorboard_log}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

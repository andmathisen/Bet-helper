from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed


def _scrape_league_worker(payload: tuple[str, bool, bool]):
    """
    Worker function for parallel scraping.
    Returns (league, report_or_none, error_str_or_none).
    """
    league, no_upcoming, no_h2h = payload
    try:
        from bet_helper.scrape.service import scrape_league

        report = scrape_league(
            league,
            update_upcoming=(not no_upcoming),
            include_h2h=(not no_h2h),
        )
        return league, report, None
    except Exception as e:
        return league, None, str(e)


def main():
    parser = argparse.ArgumentParser(prog="bet-helper", description="Bet Helper CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scrape = sub.add_parser("scrape", help="Scrape fixtures and update caches/historical files")
    p_scrape.add_argument("--league", help="League key (e.g. LaLiga). Required unless --all is used.")
    p_scrape.add_argument("--all", action="store_true", help="Scrape all leagues from league mapping")
    p_scrape.add_argument("--no-upcoming", action="store_true", help="Do not refresh upcoming fixtures cache")
    p_scrape.add_argument("--no-h2h", action="store_true", help="Skip H2H detail scraping (much faster)")
    p_scrape.add_argument("--workers", type=int, default=1, help="Parallel processes for scraping when using --all")

    p_predict = sub.add_parser("predict", help="Generate predictions from cached data (no scraping)")
    p_predict.add_argument("--league", help="League key (e.g. LaLiga). Required unless --all is used.")
    p_predict.add_argument("--all", action="store_true", help="Generate predictions for all leagues")
    
    p_rl = sub.add_parser("rl", help="Reinforcement learning betting agent commands")
    rl_sub = p_rl.add_subparsers(dest="rl_cmd", required=True)
    
    # RL train command
    p_rl_train = rl_sub.add_parser("train", help="Train RL betting agent")
    rl_grp = p_rl_train.add_mutually_exclusive_group(required=True)
    rl_grp.add_argument("--leagues", nargs="+", help="League codes (e.g., PL SerieA)")
    rl_grp.add_argument("--all", action="store_true", help="Use all available leagues")
    p_rl_train.add_argument("--total-timesteps", type=int, default=100000, help="Total training timesteps")
    p_rl_train.add_argument("--initial-tokens", type=float, default=1000.0, help="Starting token balance")
    p_rl_train.add_argument("--max-matches-per-bet", type=int, default=5, help="Max matches in accumulator")
    p_rl_train.add_argument("--no-ml-predictions", action="store_true", help="Disable ML model predictions")
    p_rl_train.add_argument("--save-path", type=str, help="Path to save model")
    p_rl_train.add_argument("--tensorboard-log", type=str, help="Path for tensorboard logs")
    p_rl_train.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    p_rl_train.add_argument("--log-level", default="INFO", help="Logging level")
    
    # RL eval command
    p_rl_eval = rl_sub.add_parser("eval", help="Evaluate trained RL agent")
    p_rl_eval.add_argument("--model-path", required=True, help="Path to trained model")
    rl_eval_grp = p_rl_eval.add_mutually_exclusive_group(required=True)
    rl_eval_grp.add_argument("--leagues", nargs="+", help="League codes")
    rl_eval_grp.add_argument("--all", action="store_true", help="Use all available leagues")
    p_rl_eval.add_argument("--n-episodes", type=int, default=20, help="Number of evaluation episodes")
    p_rl_eval.add_argument("--initial-tokens", type=float, default=1000.0, help="Starting token balance")
    p_rl_eval.add_argument("--output", type=str, help="Path to save evaluation results")
    p_rl_eval.add_argument("--log-level", default="INFO", help="Logging level")
    
    # RL play command (live simulation)
    p_rl_play = rl_sub.add_parser("play", help="Run agent in live simulation mode")
    p_rl_play.add_argument("--model-path", required=True, help="Path to trained model")
    rl_play_grp = p_rl_play.add_mutually_exclusive_group(required=True)
    rl_play_grp.add_argument("--leagues", nargs="+", help="League codes")
    rl_play_grp.add_argument("--all", action="store_true", help="Use all available leagues")
    p_rl_play.add_argument("--initial-tokens", type=float, default=1000.0, help="Starting token balance")
    p_rl_play.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.cmd == "scrape":
        from bet_helper.scrape.service import scrape_league
        from scrapers.league_mapping import LEAGUE_CODE_TO_PATH
        
        if args.all:
            leagues = list(LEAGUE_CODE_TO_PATH.keys())
            logging.info(f"Scraping all {len(leagues)} leagues: {', '.join(leagues)}")
            reports = []
            workers = max(1, int(args.workers or 1))
            if workers == 1:
                for league in leagues:
                    try:
                        report = scrape_league(
                            league,
                            update_upcoming=(not args.no_upcoming),
                            include_h2h=(not args.no_h2h),
                        )
                        reports.append(report)
                        print(f"\n{report}\n")
                    except Exception as e:
                        logging.error(f"Error scraping {league}: {e}")
                        reports.append(None)
            else:
                payloads = [(lg, args.no_upcoming, args.no_h2h) for lg in leagues]
                with ProcessPoolExecutor(max_workers=workers) as exe:
                    future_map = {exe.submit(_scrape_league_worker, payload): payload[0] for payload in payloads}
                    for fut in as_completed(future_map):
                        lg = future_map[fut]
                        try:
                            league, report, err = fut.result()
                            if err:
                                logging.error(f"[{league}] scrape failed: {err}")
                                reports.append(None)
                            else:
                                reports.append(report)
                                print(f"\n{report}\n")
                        except Exception as e:
                            logging.error(f"[{lg}] scrape crashed: {e}")
                            reports.append(None)
            return
        
        if not args.league:
            parser.error("--league is required unless --all is used")
        
        report = scrape_league(
            args.league,
            update_upcoming=(not args.no_upcoming),
            include_h2h=(not args.no_h2h),
        )
        print(report)
        return

    if args.cmd == "predict":
        from bet_helper.predict.service import generate_predictions
        from scrapers.league_mapping import LEAGUE_CODE_TO_PATH
        
        if args.all:
            leagues = list(LEAGUE_CODE_TO_PATH.keys())
            logging.info(f"Generating predictions for all {len(leagues)} leagues...")
            reports = []
            for league in leagues:
                try:
                    report = generate_predictions(league)
                    reports.append(report)
                    print(f"\n{report}\n")
                except Exception as e:
                    logging.error(f"Error generating predictions for {league}: {e}")
                    reports.append(None)
            return
        
        if not args.league:
            parser.error("--league is required unless --all is used")
        
        report = generate_predictions(args.league)
        print(report)
        return
    
    if args.cmd == "rl":
        if args.rl_cmd == "train":
            # Import and call train.py main
            from bet_helper.rl.train import main as rl_train_main
            import sys
            # Build arguments for train.py
            train_args = ["train.py"]
            if args.all:
                train_args.append("--all")
            else:
                train_args.extend(["--leagues"] + args.leagues)
            train_args.extend([
                "--total-timesteps", str(args.total_timesteps),
                "--initial-tokens", str(args.initial_tokens),
                "--max-matches-per-bet", str(args.max_matches_per_bet),
                "--learning-rate", str(args.learning_rate),
                "--log-level", args.log_level,
            ])
            if args.no_ml_predictions:
                train_args.append("--no-ml-predictions")
            if args.save_path:
                train_args.extend(["--save-path", args.save_path])
            if args.tensorboard_log:
                train_args.extend(["--tensorboard-log", args.tensorboard_log])
            
            # Temporarily replace sys.argv
            old_argv = sys.argv
            sys.argv = train_args
            try:
                return rl_train_main()
            finally:
                sys.argv = old_argv
        
        elif args.rl_cmd == "eval":
            from bet_helper.rl.eval import main as rl_eval_main
            import sys
            eval_args = ["eval.py", "--model-path", args.model_path]
            if args.all:
                eval_args.append("--all")
            else:
                eval_args.extend(["--leagues"] + args.leagues)
            eval_args.extend([
                "--n-episodes", str(args.n_episodes),
                "--initial-tokens", str(args.initial_tokens),
                "--log-level", args.log_level,
            ])
            if args.output:
                eval_args.extend(["--output", args.output])
            
            old_argv = sys.argv
            sys.argv = eval_args
            try:
                return rl_eval_main()
            finally:
                sys.argv = old_argv
        
        elif args.rl_cmd == "play":
            # Live simulation mode
            from bet_helper.rl.data import load_match_dates, load_historical_data
            from bet_helper.rl.env import BettingEnv
            from stable_baselines3 import PPO
            
            # Resolve leagues
            if args.all:
                from scrapers.league_mapping import LEAGUE_CODE_TO_PATH
                leagues = sorted(LEAGUE_CODE_TO_PATH.keys())
            else:
                leagues = args.leagues
            
            logging.info(f"Loading model from {args.model_path}...")
            model = PPO.load(args.model_path)
            
            logging.info("Loading match dates...")
            historical = load_historical_data(leagues)
            match_dates = load_match_dates(leagues, use_predictions=True, historical_data=historical)
            
            env = BettingEnv(
                match_dates=match_dates,
                historical_data=historical,
                initial_tokens=args.initial_tokens,
                use_ml_predictions=True,
            )
            
            obs, info = env.reset()
            done = False
            
            print(f"\n🎮 Starting live simulation with {args.initial_tokens:.0f} tokens")
            print("="*60)
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                if info.get("action") in ["single", "accumulator"]:
                    date = info.get("current_date", "N/A")
                    action_type = info.get("action", "unknown")
                    result = info.get("result", "unknown")
                    tokens = info.get("tokens", 0)
                    
                    result_emoji = "✅" if result == "win" else "❌"
                    print(f"{result_emoji} [{date}] {action_type}: {result} | Tokens: {tokens:.2f}")
            
            print("="*60)
            print(f"🏁 Simulation complete!")
            print(f"   Final tokens: {info.get('tokens', 0):.2f}")
            print(f"   Total profit: {info.get('total_profit', 0):.2f}")
            print(f"   Win rate: {info.get('win_rate', 0)*100:.1f}%")
            print(f"   Total bets: {info.get('total_bets', 0)}")
            
            return 0

    logging.error(f"Unknown command: {args.cmd}")
    parser.print_help()
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

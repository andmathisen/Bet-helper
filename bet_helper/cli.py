from __future__ import annotations

import argparse
import logging


def main():
    parser = argparse.ArgumentParser(prog="bet-helper", description="Bet Helper CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scrape = sub.add_parser("scrape", help="Scrape fixtures and update caches/historical files")
    p_scrape.add_argument("--league", help="League key (e.g. LaLiga). Required unless --all is used.")
    p_scrape.add_argument("--all", action="store_true", help="Scrape all leagues from league mapping")
    p_scrape.add_argument("--no-upcoming", action="store_true", help="Do not refresh upcoming fixtures cache")

    p_predict = sub.add_parser("predict", help="Generate predictions from cached data (no scraping)")
    p_predict.add_argument("--league", help="League key (e.g. LaLiga). Required unless --all is used.")
    p_predict.add_argument("--all", action="store_true", help="Generate predictions for all leagues")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.cmd == "scrape":
        from bet_helper.scrape.service import scrape_league
        from scrapers.league_mapping import LEAGUE_CODE_TO_PATH
        
        if args.all:
            leagues = list(LEAGUE_CODE_TO_PATH.keys())
            logging.info(f"Scraping all {len(leagues)} leagues: {', '.join(leagues)}")
            reports = []
            for league in leagues:
                try:
                    report = scrape_league(league, update_upcoming=(not args.no_upcoming))
                    reports.append(report)
                    print(f"\n{report}\n")
                except Exception as e:
                    logging.error(f"Error scraping {league}: {e}")
                    reports.append(None)
            return
        
        if not args.league:
            parser.error("--league is required unless --all is used")
        
        report = scrape_league(args.league, update_upcoming=(not args.no_upcoming))
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


if __name__ == "__main__":
    main()


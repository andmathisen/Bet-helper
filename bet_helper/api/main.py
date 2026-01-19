from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from bet_helper.storage import data_dir, historical_path, load_json, predictions_path
from bet_helper.predict.service import generate_predictions
from scrapers.league_mapping import LEAGUE_CODE_TO_PATH


app = FastAPI(title="Bet Helper API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/leagues")
def get_leagues():
    """
    Discover available leagues.
    - Prefer known leagues from our FootyStats mapping (so users can scrape before predictions exist)
    - Also include leagues that already have data files present in data/
    """
    base: Path = data_dir()
    leagues: set[str] = set(LEAGUE_CODE_TO_PATH.keys())

    # Also include leagues inferred from cached files (predictions/historical/upcoming)
    for glob_pat, prefix in [
        ("predictions_*.json", "predictions_"),
        ("historical_matches_*.json", "historical_matches_"),
        ("upcoming_*.json", "upcoming_"),
    ]:
        for p in base.glob(glob_pat):
            stem = p.stem
            if stem.startswith(prefix):
                league = stem[len(prefix) :]
                if league:
                    leagues.add(league)

    out = sorted(leagues)
    return {"count": len(out), "leagues": out}


@app.get("/api/historical/summary")
def historical_summary(league: str):
    path = historical_path(league)
    data = load_json(path, default=None)
    if data is None:
        raise HTTPException(status_code=404, detail=f"No historical file found for league={league}. Run scrape first.")

    # historical matches are currently stored as a dict keyed by "<date>_<home>-<away>"
    items = []
    if isinstance(data, dict):
        items = list(data.values())
        count = len(data)
    elif isinstance(data, list):
        items = data
        count = len(data)
    else:
        items = []
        count = 0

    dates: list[datetime] = []
    for md in items:
        ds = (md or {}).get("Date")
        if not ds:
            continue
        try:
            dates.append(datetime.strptime(ds, "%d %b %Y"))
        except Exception:
            continue

    first_date = min(dates).strftime("%d %b %Y") if dates else None
    last_date = max(dates).strftime("%d %b %Y") if dates else None

    last_updated = None
    try:
        st = path.stat()
        last_updated = datetime.utcfromtimestamp(st.st_mtime).isoformat() + "Z"
    except Exception:
        last_updated = None

    return {
        "league": league,
        "count": count,
        "first_match_date": first_date,
        "last_match_date": last_date,
        "last_updated_utc": last_updated,
        "path": str(path),
    }


@app.post("/api/scrape")
def run_scrape(league: str | None = None, upcoming: bool = True, all_leagues: bool = False):
    # Lazy import so API can run without Selenium unless this endpoint is called.
    try:
        from bet_helper.scrape.service import scrape_league

        if all_leagues:
            reports = []
            for league_code in sorted(LEAGUE_CODE_TO_PATH.keys()):
                try:
                    report = scrape_league(league_code, update_upcoming=upcoming)
                    reports.append(report.__dict__)
                except Exception as e:
                    reports.append({"league": league_code, "error": str(e)})
            return {"count": len(reports), "reports": reports}
        
        if not league:
            raise HTTPException(status_code=400, detail="league parameter is required unless all_leagues=true")
        
        report = scrape_league(league, update_upcoming=upcoming)
        return report.__dict__
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions")
def get_predictions(league: str):
    path = predictions_path(league)
    data = load_json(path, default=None)
    if data is None:
        raise HTTPException(status_code=404, detail=f"No predictions found for league={league}. Run predict first.")
    return {"league": league, "count": len(data), "predictions": data}


@app.post("/api/predict")
def run_predict(league: str):
    # Runs prediction generation (no scraping). This can take a moment but is safe.
    try:
        report = generate_predictions(league)
        return report.__dict__
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


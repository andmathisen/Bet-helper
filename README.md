# Bet Helper

Backend (FastAPI + scraping + predictions) is managed with Poetry; the frontend lives in `frontend/`.

## Setup
1) Install Python 3.10+ and Poetry (running on Python 3.9 triggers Pydantic `str | None` type-eval errors).
2) Create the environment and install deps:
```
poetry env use python3.11  # or another 3.10+ interpreter
poetry install
```
3) Copy `.env.example` to `.env` and add your FootyStats credentials (needed for scraping H2H stats).
4) Ensure Google Chrome is installed; Selenium will auto-manage the driver via `webdriver-manager`.

## Run the API
```
poetry run uvicorn bet_helper.api.main:app --host 127.0.0.1 --port 8000 --reload
```

## CLI helpers
```
poetry run bet-helper scrape --league PL       # scrape fixtures + caches
poetry run bet-helper predict --league PL      # generate predictions
```
Add `--all` to process every league.
Add `--no-h2h` to skip per-match H2H scraping (faster) and `--workers 4` to parallelize leagues when using `--all`.

## Model evaluation (offline)
```
# Requires historical_matches_<league>.json in data/
poetry run python -m bet_helper.predict.eval --league PL --splits 5 --grid medium
```
The script runs time-series cross validation with several XGBoost parameter presets and reports log loss, Brier score, and accuracy so you can pick the best setup before updating `bet_helper/predict/ml_model.py`.

## Frontend (optional)
From `frontend/`:
```
npm install
npm run dev
```

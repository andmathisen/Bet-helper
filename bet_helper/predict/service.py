from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from bet_helper.storage import historical_path, upcoming_path, predictions_path, load_json, save_json, migrate_legacy_historical_files

from bet_helper.models import TeamData, MatchData, calculate_team_form
from bet_helper.predict import core
from bet_helper.predict import ml_model


@dataclass
class PredictReport:
    league: str
    matches: int
    predictions_path: str
    timestamp: str


def _build_teams_from_historical(historical_data: dict) -> dict[str, TeamData]:
    """
    Build TeamData objects + recent form from historical matches file.
    """
    teams: dict[str, TeamData] = {}

    def get_or_create(name: str) -> TeamData:
        if name not in teams:
            teams[name] = TeamData(name=name)
        return teams[name]

    # Sort by date descending so form uses most recent games first
    items = list((historical_data or {}).items())
    def _dt(md: dict) -> datetime:
        d = core._parse_dd_mmm_yyyy_season_aware((md or {}).get("Date", ""), reference_date=datetime.now())
        return d or datetime(1970, 1, 1)
    items.sort(key=lambda kv: _dt(kv[1]), reverse=True)

    for _, md in items:
        try:
            match_str = (md or {}).get("Match", "")
            if "-" not in match_str:
                continue
            home_name, away_name = [s.strip() for s in match_str.split("-", 1)]
            score = (md or {}).get("Score", "")
            if ":" not in score:
                continue
            hg_s, ag_s = score.split(":", 1)
            hg = int(hg_s.strip())
            ag = int(ag_s.strip())
            date = (md or {}).get("Date", "")

            home = get_or_create(home_name)
            away = get_or_create(away_name)
            m = MatchData(date=date, home_team=home_name, away_team=away_name, home_goals=hg, away_goals=ag)
            home.add_match(m)
            away.add_match(m)
        except Exception:
            continue

    return teams


def _team_form_summary(team: TeamData, n: int = 5) -> dict[str, Any]:
    """
    Summarize a team's recent form from the last n matches in TeamData.form.
    TeamData.form is expected to be ordered newest -> oldest.
    """
    def _summarize(matches: list[MatchData], team_name: str) -> dict[str, Any]:
        seq: list[str] = []
        w = d = l = 0
        pts = 0
        gf = ga = 0
        for m in matches:
            try:
                hg, ag = m.get_score()
                if m.home_team == team_name:
                    f, a = hg, ag
                else:
                    f, a = ag, hg
                gf += int(f)
                ga += int(a)
                if f > a:
                    seq.append("W")
                    w += 1
                    pts += 3
                elif f == a:
                    seq.append("D")
                    d += 1
                    pts += 1
                else:
                    seq.append("L")
                    l += 1
            except Exception:
                continue
        n_eff = len(seq)
        return {
            "n": n_eff,
            "sequence": "".join(seq),
            "record": {"W": w, "D": d, "L": l},
            "points": pts,
            "ppg": None if n_eff <= 0 else round(pts / n_eff, 2),
            "gf": gf,
            "ga": ga,
        }

    try:
        n = int(n)
    except Exception:
        n = 5
    n = max(0, min(n, 50))

    recent = list(team.form or [])[:n]
    as_home = [m for m in recent if m.home_team == team.name]
    as_away = [m for m in recent if m.away_team == team.name]

    out = _summarize(recent, team.name)
    out["as_home"] = _summarize(as_home, team.name)
    out["as_away"] = _summarize(as_away, team.name)
    return out


def generate_predictions(league: str) -> PredictReport:
    migrate_legacy_historical_files()
    hist = load_json(historical_path(league), default={}) or {}
    upcoming = load_json(upcoming_path(league), default=[]) or []

    if not hist:
        raise RuntimeError(f"No historical data found for {league} at {historical_path(league)}")
    if not upcoming:
        raise RuntimeError(
            f"No upcoming fixtures cache found for {league} at {upcoming_path(league)}. Run scrape first."
        )

    teams = _build_teams_from_historical(hist)
    team_h2h_index = core._build_team_h2h_index(hist, reference_date=datetime.now())
    league_nb_sizes = core._compute_league_nb_sizes(hist)
    team_nb_sizes = core._compute_team_nb_sizes(hist)
    league_dc_rho = core._tune_dixon_coles_rho_from_historical(hist, max_goals=10)
    
    # Fit isotonic calibration curves from historical predictions
    isotonic_cal = core.fit_isotonic_calibration(hist, league)
    
    # Train/load ML model (XGBoost)
    ml_model_obj, ml_feature_names, ml_baseline = ml_model.fit_ml_model(hist, league)

    # Load existing predictions to preserve finished matches
    existing_predictions = load_json(predictions_path(league), default=[]) or []
    if not isinstance(existing_predictions, list):
        existing_predictions = []
    
    # Build a map of existing predictions by match (home-away key)
    # We'll preserve predictions for matches that have already been played (in historical_data)
    existing_by_match: dict[str, dict[str, Any]] = {}
    for pred in existing_predictions:
        match_info = pred.get("match", {})
        home_name = (match_info.get("home") or "").strip()
        away_name = (match_info.get("away") or "").strip()
        if home_name and away_name:
            match_key = f"{home_name}|{away_name}"
            existing_by_match[match_key] = pred
    
    # Check which matches have been played (exist in historical data)
    # These predictions should be preserved and never updated
    played_matches: set[str] = set()
    for _, hist_match in hist.items():
        match_str = hist_match.get("Match", "")
        if "-" not in match_str:
            continue
        hist_home, hist_away = [s.strip() for s in match_str.split("-", 1)]
        if hist_home and hist_away:
            played_matches.add(f"{hist_home}|{hist_away}")
    
    # Build set of upcoming matches for quick lookup
    upcoming_matches: set[str] = set()
    for fx in upcoming:
        home_name = (fx.get("home") or "").strip()
        away_name = (fx.get("away") or "").strip()
        if home_name and away_name:
            upcoming_matches.add(f"{home_name}|{away_name}")

    out = []
    
    # First, preserve predictions for finished matches (in historical data but not in upcoming)
    for match_key, pred in existing_by_match.items():
        if match_key in played_matches:
            # This match has been played - preserve its prediction
            out.append(pred)
    
    # Then, generate/update predictions for upcoming matches
    for fx in upcoming:
        home_name = (fx.get("home") or "").strip()
        away_name = (fx.get("away") or "").strip()
        if not home_name or not away_name:
            continue

        # Match Team objects using the existing robust matcher
        hk = core._find_team_in_dict(home_name, teams)
        ak = core._find_team_in_dict(away_name, teams)
        if hk is None or ak is None:
            logging.warning(f"[predict] Team mapping failed for {league}: {home_name} vs {away_name}")
            continue

        home = teams[hk]
        away = teams[ak]

        odds = fx.get("odds") or {}
        h_odds = float(odds.get("home") or 0)
        d_odds = float(odds.get("draw") or 0)
        a_odds = float(odds.get("away") or 0)

        ph, pd, pa, expected_result, dist = core.predict_match_poisson_with_weighted_h2h(
            home_name, away_name, home.form, away.form, team_h2h_index=team_h2h_index, dc_rho=league_dc_rho, isotonic_calibration=isotonic_cal
        )
        
        # Generate ML prediction (XGBoost)
        ml_probs = ml_model.predict_with_ml_model(
            home_name, away_name, home.form, away.form,
            team_h2h_index, ml_model_obj, ml_feature_names,
            reference_date=datetime.now(),
            league=league
        )
        
        # Compute SHAP explanations if ML prediction succeeded
        shap_explanations = None
        if ml_probs and ml_model_obj and ml_feature_names:
            try:
                # Extract features for SHAP (same extraction as in predict_with_ml_model)
                ml_features = ml_model.extract_features_for_match(
                    home_name, away_name,
                    home.form, away.form,
                    team_h2h_index,
                    datetime.now(),
                    league=league
                )
                shap_explanations = ml_model.get_shap_explanations(
                    ml_model_obj, ml_feature_names, ml_features, datetime.now(), baseline=ml_baseline
                )
                logging.info(f"[SHAP] Got shap_explanations for {home_name} vs {away_name}: {shap_explanations is not None}")
            except Exception as e:
                # Log at info level so we can see if SHAP is failing
                logging.warning(f"SHAP explanation computation failed for {home_name} vs {away_name}: {e}")
                import traceback
                logging.debug(f"SHAP error traceback:\n{traceback.format_exc()}")

        market = core._implied_probs_from_decimal_odds(h_odds, d_odds, a_odds)
        blended = None
        if market:
            w_model = 0.75
            blended = (
                (w_model * ph) + ((1 - w_model) * market[0]),
                (w_model * pd) + ((1 - w_model) * market[1]),
                (w_model * pa) + ((1 - w_model) * market[2]),
            )
            s = sum(blended) or 1.0
            blended = (blended[0] / s, blended[1] / s, blended[2] / s)

        ev_home = core.decimal_ev(ph, h_odds)
        ev_draw = core.decimal_ev(pd, d_odds)
        ev_away = core.decimal_ev(pa, a_odds)

        # Calculate ML EV if ML probabilities are available
        ml_ev_home = ml_ev_draw = ml_ev_away = None
        if ml_probs:
            ml_ev_home = core.decimal_ev(ml_probs[0], h_odds)
            ml_ev_draw = core.decimal_ev(ml_probs[1], d_odds)
            ml_ev_away = core.decimal_ev(ml_probs[2], a_odds)

        best = None
        for outcome, ev in (("Home", ev_home), ("Draw", ev_draw), ("Away", ev_away)):
            if ev is None:
                continue
            if best is None or ev > best["ev"]:
                best = {"outcome": outcome, "ev": float(ev)}

        recommended = {"outcome": "No bet", "ev": None, "kelly_25pct": None}
        if best and best["ev"] > 0:
            if best["outcome"] == "Home":
                k = core.fractional_kelly(ph, h_odds, frac=0.25)
            elif best["outcome"] == "Draw":
                k = core.fractional_kelly(pd, d_odds, frac=0.25)
            else:
                k = core.fractional_kelly(pa, a_odds, frac=0.25)
            recommended = {"outcome": best["outcome"], "ev": round(best["ev"], 3), "kelly_25pct": None if k is None else round(float(k), 3)}

        extra_stats = {}
        for sp in ("corners", "cards", "shots", "sot"):
            r_match, r_source = core._match_nb_r(sp, home_name, away_name, team_nb_sizes, league_nb_sizes)
            s = core._expected_stat_for_match(
                home_name,
                away_name,
                team_h2h_index,
                sp,
                half_life_days=30.0,
                nb_sizes={sp: r_match} if r_match else None,
            )
            if s:
                probs = {k: round(float(v), 3) for k, v in s.items() if isinstance(k, str) and k.startswith("p_over_")}
                extra_stats[sp] = {
                    "model": s.get("dist"),
                    "nb_size_r": None if s.get("nb_size_r") is None else round(float(s["nb_size_r"]), 3),
                    "nb_r_source": r_source,
                    "expected": f'{s["expected_home"]:.1f}-{s["expected_away"]:.1f} (total {s["expected_total"]:.1f})',
                    "total_median": s.get("total_median"),
                    "total_q25_q75": s.get("total_q25_q75"),
                    **probs,
                }

        # Team form (last 5 matches) - overall + split by venue from the team's perspective.
        form = {"n": 5, "home": _team_form_summary(home, n=5), "away": _team_form_summary(away, n=5)}

        # Generate new prediction for this upcoming match
        # If a prediction already exists for this match, we update it (since odds/conditions may have changed)
        match_key = f"{home_name}|{away_name}"
        
        # Format ML probabilities
        ml_pred = None
        if ml_probs:
            ml_pred = {"home": round(ml_probs[0], 4), "draw": round(ml_probs[1], 4), "away": round(ml_probs[2], 4)}
        
        # Debug: Log shap_explanations before creating new_pred
        logging.info(f"[SHAP] Creating prediction dict for {home_name} vs {away_name}, shap_explanations is None: {shap_explanations is None}")
        
        new_pred = {
            "league": league,
            "match": {"home": home_name, "away": away_name},
            "odds": {"home": h_odds, "draw": d_odds, "away": a_odds},
            "probs": {"home": round(ph, 4), "draw": round(pd, 4), "away": round(pa, 4)},
            "ml_probs": ml_pred,
            "shap_explanations": shap_explanations,
            "market_implied": None if not market else {"home": round(market[0], 4), "draw": round(market[1], 4), "away": round(market[2], 4)},
            "blended": None if not blended else {"home": round(blended[0], 4), "draw": round(blended[1], 4), "away": round(blended[2], 4)},
            "ev": {
                "home": None if ev_home is None else round(float(ev_home), 4),
                "draw": None if ev_draw is None else round(float(ev_draw), 4),
                "away": None if ev_away is None else round(float(ev_away), 4),
            },
            "ml_ev": None if not ml_probs else {
                "home": None if ml_ev_home is None else round(float(ml_ev_home), 4),
                "draw": None if ml_ev_draw is None else round(float(ml_ev_draw), 4),
                "away": None if ml_ev_away is None else round(float(ml_ev_away), 4),
            },
            "recommended": recommended,
            "expected_result": expected_result,
            "distribution": dist,
            "extra_stats": extra_stats,
            "form": form,
        }
        out.append(new_pred)

    save_json(predictions_path(league), out)
    return PredictReport(league=league, matches=len(out), predictions_path=str(predictions_path(league)), timestamp=datetime.utcnow().isoformat() + "Z")


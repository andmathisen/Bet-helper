from __future__ import annotations

import math
import re
import logging
from datetime import datetime


def _normalize_club_name_for_stats(name: str) -> str:
    import unicodedata
    if not name:
        return ""
    n = unicodedata.normalize("NFD", name.lower().strip())
    n = "".join(c for c in n if unicodedata.category(c) != "Mn")
    n = re.sub(r"\s+", " ", n).strip()
    for prefix in [
        "real club deportivo ",
        "real club deportiu ",
        "reial club deportiu ",
        "real club ",
        "club deportivo ",
        "club deportiu ",
        "club de futbol ",
        "club de fútbol ",
        "fc ",
        "cf ",
        "cd ",
        "rcd ",
        "ud ",
        "ac ",
        "sc ",
        "ca ",
    ]:
        if n.startswith(prefix):
            n = n[len(prefix):].strip()
            break
    n = re.sub(r"\b\d+\b", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    for token in [
        " cf", " fc", " ud", " ac", " sc", " cd", " rc",
        " club de futbol", " club de fútbol",
        " de futbol", " de fútbol",
        " balompie", " balompie",
    ]:
        if n.endswith(token):
            n = n[: -len(token)].strip()
            break
    n = re.sub(r"[^\w\s]", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def _safe_float(v) -> float:
    try:
        if v is None:
            return 0.0
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if not s:
            return 0.0
        s = s.replace("%", "")
        return float(s)
    except Exception:
        return 0.0


def decimal_ev(p: float, odds: float) -> float | None:
    try:
        p = float(p)
        odds = float(odds)
        if odds <= 1e-9 or p <= 0:
            return None
        return (p * odds) - 1.0
    except Exception:
        return None


def kelly_fraction(p: float, odds: float) -> float | None:
    try:
        p = float(p)
        odds = float(odds)
        if odds <= 1.0 or p <= 0 or p >= 1:
            return 0.0
        b = odds - 1.0
        q = 1.0 - p
        f = (b * p - q) / b
        return max(0.0, float(f))
    except Exception:
        return None


def fractional_kelly(p: float, odds: float, frac: float = 0.25) -> float | None:
    try:
        f = kelly_fraction(p, odds)
        if f is None:
            return None
        return max(0.0, float(frac) * float(f))
    except Exception:
        return None


def _parse_dd_mmm_yyyy_season_aware(date_str: str, reference_date: datetime | None = None) -> datetime | None:
    if not date_str:
        return None
    if reference_date is None:
        reference_date = datetime.now()
    try:
        d = datetime.strptime(date_str, "%d %b %Y")
    except Exception:
        return None
    if (d - reference_date).days > 30:
        try:
            d2 = d.replace(year=d.year - 1)
            if (reference_date - d2).days >= 0:
                return d2
        except Exception:
            pass
    return d


def poisson_probability(goals: int, expected_goals: float) -> float:
    if expected_goals <= 0:
        return 0.0
    try:
        return (expected_goals ** goals) * math.exp(-expected_goals) / math.factorial(goals)
    except Exception:
        return 0.0


def _dixon_coles_tau(hg: int, ag: int, lam: float, mu: float, rho: float) -> float:
    try:
        hg = int(hg)
        ag = int(ag)
        lam = float(lam)
        mu = float(mu)
        rho = float(rho)
        if hg == 0 and ag == 0:
            return 1.0 - (lam * mu * rho)
        if hg == 0 and ag == 1:
            return 1.0 + (lam * rho)
        if hg == 1 and ag == 0:
            return 1.0 + (mu * rho)
        if hg == 1 and ag == 1:
            return 1.0 - rho
        return 1.0
    except Exception:
        return 1.0


def _temperature_scale_probs(p_home: float, p_draw: float, p_away: float, temperature: float = 1.1) -> tuple[float, float, float]:
    try:
        T = float(temperature)
        if T <= 0:
            return p_home, p_draw, p_away
        a = max(float(p_home), 0.0) ** (1.0 / T)
        b = max(float(p_draw), 0.0) ** (1.0 / T)
        c = max(float(p_away), 0.0) ** (1.0 / T)
        s = a + b + c
        if s <= 0:
            return p_home, p_draw, p_away
        return a / s, b / s, c / s
    except Exception:
        return p_home, p_draw, p_away


def _asymmetric_calibration(p_home: float, p_draw: float, p_away: float, 
                           underdog_threshold: float = 0.30,
                           correction_factor: float = 0.85) -> tuple[float, float, float]:
    """
    Apply calibration that reduces probabilities for underdogs more aggressively.
    This addresses overconfidence in rare events (big underdogs).
    
    Args:
        underdog_threshold: Probabilities below this are considered underdogs
        correction_factor: Multiplier for underdog probabilities (0.85 = 15% reduction)
    """
    try:
        p_home = max(0.0, min(1.0, float(p_home)))
        p_draw = max(0.0, min(1.0, float(p_draw)))
        p_away = max(0.0, min(1.0, float(p_away)))
        
        # Apply correction to underdogs
        if p_home < underdog_threshold:
            p_home = p_home * correction_factor
        if p_away < underdog_threshold:
            p_away = p_away * correction_factor
        
        # Draws are typically not underdogs, but if very low, apply small correction
        if p_draw < 0.15:  # Very low draw probability
            p_draw = p_draw * 0.90
        
        # Renormalize to ensure probabilities sum to 1
        total = p_home + p_draw + p_away
        if total > 0:
            p_home = max(0.0, min(1.0, p_home / total))
            p_draw = max(0.0, min(1.0, p_draw / total))
            p_away = max(0.0, min(1.0, p_away / total))
        
        return p_home, p_draw, p_away
    except Exception:
        return p_home, p_draw, p_away


def _implied_probs_from_decimal_odds(h_odds: float, d_odds: float, a_odds: float) -> tuple[float, float, float] | None:
    try:
        h = float(h_odds)
        d = float(d_odds)
        a = float(a_odds)
        if h <= 1.0 or d <= 1.0 or a <= 1.0:
            return None
        ph = 1.0 / h
        pd = 1.0 / d
        pa = 1.0 / a
        s = ph + pd + pa
        if s <= 0:
            return None
        return ph / s, pd / s, pa / s
    except Exception:
        return None


def _fit_nb_size_from_samples(values: list[float], min_n: int = 20) -> float | None:
    try:
        xs = [float(v) for v in values if v is not None]
        if len(xs) < int(min_n):
            return None
        mu = sum(xs) / len(xs)
        if mu <= 0:
            return None
        var = sum((x - mu) ** 2 for x in xs) / max(1, (len(xs) - 1))
        if var <= mu * 1.05:
            return None
        r = (mu * mu) / max(1e-9, (var - mu))
        return float(max(0.05, min(r, 1e6)))
    except Exception:
        return None


def _nb_pmf(k: int, mu: float, r: float) -> float:
    try:
        k = int(k)
        if k < 0:
            return 0.0
        mu = float(mu)
        r = float(r)
        if mu < 0 or r <= 0:
            return 0.0
        if mu == 0:
            return 1.0 if k == 0 else 0.0
        p = r / (r + mu)
        log_coeff = math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
        log_pmf = log_coeff + (r * math.log(p)) + (k * math.log(1.0 - p))
        return float(math.exp(log_pmf))
    except Exception:
        return 0.0


def _nb_tail_prob_over(line: float, mu: float, r: float, max_k: int = 120) -> float:
    try:
        mu = max(float(mu), 0.0)
        r = float(r)
        if r <= 0:
            return 0.0
        k0 = int(math.floor(float(line))) + 1
        cdf = 0.0
        for k in range(0, min(k0, max_k + 1)):
            cdf += _nb_pmf(k, mu, r)
        return max(0.0, min(1.0, 1.0 - cdf))
    except Exception:
        return 0.0


def _nb_quantile(q: float, mu: float, r: float, max_k: int = 120) -> int:
    try:
        q = float(q)
        mu = max(float(mu), 0.0)
        r = float(r)
        if r <= 0:
            return 0
        cdf = 0.0
        for k in range(0, max_k + 1):
            cdf += _nb_pmf(k, mu, r)
            if cdf >= q:
                return k
        return max_k
    except Exception:
        return 0


def _weighted_stat_profile(entries: list, for_key: str, against_key: str, half_life_days: float = 30.0, home_only: bool | None = None) -> dict | None:
    if not entries:
        return None
    total_w = 0.0
    w_for = 0.0
    w_against = 0.0
    for e in entries:
        try:
            if home_only is not None and bool(e.get("is_home")) != home_only:
                continue
            days_ago = float(e.get("days_ago", 0))
            w = math.pow(0.5, days_ago / float(half_life_days))
            total_w += w
            w_for += float(e.get(for_key, 0.0)) * w
            w_against += float(e.get(against_key, 0.0)) * w
        except Exception:
            continue
    if total_w <= 0:
        return None
    return {"for": w_for / total_w, "against": w_against / total_w}


def _build_team_h2h_index(historical_data: dict, reference_date: datetime) -> dict:
    idx = {}
    for _, match_data in (historical_data or {}).items():
        try:
            date_str = match_data.get("Date")
            if not date_str:
                continue
            match_date = _parse_dd_mmm_yyyy_season_aware(date_str, reference_date=reference_date)
            if not match_date:
                continue
            days_ago = (reference_date - match_date).days
            if days_ago < 0:
                continue

            match_str = match_data.get("Match", "")
            if "-" not in match_str:
                continue
            home_raw, away_raw = match_str.split("-", 1)
            home_raw = home_raw.strip()
            away_raw = away_raw.strip()
            home_key = _normalize_club_name_for_stats(home_raw)
            away_key = _normalize_club_name_for_stats(away_raw)
            if not home_key or not away_key:
                continue

            h2h = match_data.get("h2h_stats") or {}
            if not isinstance(h2h, dict):
                continue

            def _side_pair(stat_name: str) -> tuple[float, float]:
                dct = h2h.get(stat_name)
                if isinstance(dct, dict):
                    return _safe_float(dct.get("home")), _safe_float(dct.get("away"))
                return 0.0, 0.0

            xg_home, xg_away = _side_pair("xg")
            corners_home, corners_away = _side_pair("corners")
            cards_home, cards_away = _side_pair("cards")
            shots_home, shots_away = _side_pair("shots")
            sot_home, sot_away = _side_pair("shots_on_target")

            if (xg_home + xg_away + corners_home + corners_away + cards_home + cards_away + shots_home + shots_away + sot_home + sot_away) <= 0:
                continue

            idx.setdefault(home_key, []).append(
                {
                    "days_ago": days_ago,
                    "is_home": True,
                    "xg_for": xg_home,
                    "xg_against": xg_away,
                    "corners_for": corners_home,
                    "corners_against": corners_away,
                    "cards_for": cards_home,
                    "cards_against": cards_away,
                    "shots_for": shots_home,
                    "shots_against": shots_away,
                    "sot_for": sot_home,
                    "sot_against": sot_away,
                }
            )
            idx.setdefault(away_key, []).append(
                {
                    "days_ago": days_ago,
                    "is_home": False,
                    "xg_for": xg_away,
                    "xg_against": xg_home,
                    "corners_for": corners_away,
                    "corners_against": corners_home,
                    "cards_for": cards_away,
                    "cards_against": cards_home,
                    "shots_for": shots_away,
                    "shots_against": shots_home,
                    "sot_for": sot_away,
                    "sot_against": sot_home,
                }
            )
        except Exception:
            continue
    return idx


def _compute_league_nb_sizes(historical_data: dict) -> dict:
    sizes: dict = {}
    corners_totals: list[float] = []
    cards_totals: list[float] = []
    for _, match_data in (historical_data or {}).items():
        try:
            h2h = match_data.get("h2h_stats") or {}
            if not isinstance(h2h, dict):
                continue
            corners = h2h.get("corners")
            cards = h2h.get("cards")
            if isinstance(corners, dict):
                c = _safe_float(corners.get("home")) + _safe_float(corners.get("away"))
                if c > 0:
                    corners_totals.append(c)
            if isinstance(cards, dict):
                c = _safe_float(cards.get("home")) + _safe_float(cards.get("away"))
                if c > 0:
                    cards_totals.append(c)
        except Exception:
            continue
    rc = _fit_nb_size_from_samples(corners_totals, min_n=20)
    rd = _fit_nb_size_from_samples(cards_totals, min_n=20)
    if rc:
        sizes["corners"] = rc
    if rd:
        sizes["cards"] = rd
    return sizes


def _compute_team_nb_sizes(historical_data: dict) -> dict:
    team_values: dict = {}
    for _, match_data in (historical_data or {}).items():
        try:
            match_str = match_data.get("Match", "")
            if "-" not in match_str:
                continue
            home_team_raw, away_team_raw = match_str.split("-", 1)
            home_key = _normalize_club_name_for_stats(home_team_raw.strip())
            away_key = _normalize_club_name_for_stats(away_team_raw.strip())
            if not home_key or not away_key:
                continue
            h2h = match_data.get("h2h_stats") or {}
            if not isinstance(h2h, dict):
                continue
            corners = h2h.get("corners")
            cards = h2h.get("cards")
            corners_total = None
            cards_total = None
            if isinstance(corners, dict):
                corners_total = _safe_float(corners.get("home")) + _safe_float(corners.get("away"))
            if isinstance(cards, dict):
                cards_total = _safe_float(cards.get("home")) + _safe_float(cards.get("away"))

            def _push(team_key: str, role: str, stat: str, v: float | None):
                if v is None or v <= 0:
                    return
                team_values.setdefault(team_key, {}).setdefault(stat, {}).setdefault(role, []).append(float(v))
                team_values.setdefault(team_key, {}).setdefault(stat, {}).setdefault("all", []).append(float(v))

            _push(home_key, "home", "corners", corners_total)
            _push(away_key, "away", "corners", corners_total)
            _push(home_key, "home", "cards", cards_total)
            _push(away_key, "away", "cards", cards_total)
        except Exception:
            continue

    team_sizes: dict = {}
    for team_key, stats in team_values.items():
        for stat, roles in stats.items():
            for role, values in roles.items():
                min_n = 8 if role in ("home", "away") else 12
                r = _fit_nb_size_from_samples(values, min_n=min_n)
                if r:
                    team_sizes.setdefault(team_key, {}).setdefault(stat, {})[role] = r
    return team_sizes


def _match_nb_r(stat_prefix: str, home_team_name: str, away_team_name: str, team_nb_sizes: dict | None, league_nb_sizes: dict | None) -> tuple[float | None, str]:
    try:
        if stat_prefix not in ("corners", "cards"):
            return None, "poisson"
        hk = _normalize_club_name_for_stats(home_team_name)
        ak = _normalize_club_name_for_stats(away_team_name)
        if team_nb_sizes:
            hr = (((team_nb_sizes.get(hk, {}) or {}).get(stat_prefix, {}) or {}).get("home"))
            ar = (((team_nb_sizes.get(ak, {}) or {}).get(stat_prefix, {}) or {}).get("away"))
            if hr and ar:
                return (float(hr) + float(ar)) / 2.0, "team_role_avg"
            hr_all = (((team_nb_sizes.get(hk, {}) or {}).get(stat_prefix, {}) or {}).get("all"))
            ar_all = (((team_nb_sizes.get(ak, {}) or {}).get(stat_prefix, {}) or {}).get("all"))
            if hr_all and ar_all:
                return (float(hr_all) + float(ar_all)) / 2.0, "team_all_avg"
            if hr or ar:
                return float(hr or ar), "team_role_single"
            if hr_all or ar_all:
                return float(hr_all or ar_all), "team_all_single"
        if league_nb_sizes and league_nb_sizes.get(stat_prefix):
            return float(league_nb_sizes.get(stat_prefix)), "league"
        return None, "poisson"
    except Exception:
        return None, "poisson"


def _expected_stat_for_match(home_team_name: str, away_team_name: str, team_h2h_index: dict | None, stat_prefix: str, half_life_days: float = 30.0, nb_sizes: dict | None = None, total_lines: list[float] | None = None) -> dict | None:
    if not team_h2h_index:
        return None
    home_key = _normalize_club_name_for_stats(home_team_name)
    away_key = _normalize_club_name_for_stats(away_team_name)
    home_entries = team_h2h_index.get(home_key, [])
    away_entries = team_h2h_index.get(away_key, [])
    if not home_entries or not away_entries:
        return None

    for_key = f"{stat_prefix}_for"
    against_key = f"{stat_prefix}_against"
    home_prof_home = _weighted_stat_profile(home_entries, for_key, against_key, half_life_days=half_life_days, home_only=True) or \
                     _weighted_stat_profile(home_entries, for_key, against_key, half_life_days=half_life_days, home_only=None)
    away_prof_away = _weighted_stat_profile(away_entries, for_key, against_key, half_life_days=half_life_days, home_only=False) or \
                     _weighted_stat_profile(away_entries, for_key, against_key, half_life_days=half_life_days, home_only=None)
    if not home_prof_home or not away_prof_away:
        return None

    expected_home = (float(home_prof_home["for"]) + float(away_prof_away["against"])) / 2.0
    expected_away = (float(away_prof_away["for"]) + float(home_prof_home["against"])) / 2.0
    expected_home = max(expected_home, 0.0)
    expected_away = max(expected_away, 0.0)
    total = expected_home + expected_away

    dist = "poisson"
    r = None
    if nb_sizes and stat_prefix in ("corners", "cards"):
        r = nb_sizes.get(stat_prefix)
        if r:
            dist = "neg_binom"

    if total_lines is None:
        if stat_prefix == "corners":
            total_lines = [7.5, 8.5, 9.5, 10.5]
        elif stat_prefix == "cards":
            total_lines = [3.5, 4.5, 5.5]
        else:
            def _round_half(x: float) -> float:
                return round(float(x) * 2.0) / 2.0
            base = float(total)
            if stat_prefix == "shots":
                candidates = [base - 3.5, base - 1.5, base + 0.5, base + 2.5]
            elif stat_prefix == "sot":
                candidates = [base - 1.5, base - 0.5, base + 0.5, base + 1.5]
            else:
                candidates = [base - 1.5, base - 0.5, base + 0.5, base + 1.5]
            total_lines = []
            for x in candidates:
                line = _round_half(max(0.5, x))
                if line not in total_lines:
                    total_lines.append(line)

    probs = {}
    for line in total_lines:
        if dist == "neg_binom" and r:
            probs[f"p_over_{line}_total"] = _nb_tail_prob_over(line, total, float(r))
        else:
            # Poisson tail
            k0 = int(math.floor(float(line))) + 1
            cdf = 0.0
            for k in range(0, k0):
                cdf += poisson_probability(k, total)
            probs[f"p_over_{line}_total"] = max(0.0, min(1.0, 1.0 - cdf))

    if dist == "neg_binom" and r:
        med = _nb_quantile(0.5, total, float(r))
        q25 = _nb_quantile(0.25, total, float(r))
        q75 = _nb_quantile(0.75, total, float(r))
    else:
        # Poisson quantiles
        def _pq(q: float, lam: float, max_k: int = 120) -> int:
            c = 0.0
            for k in range(0, max_k + 1):
                c += poisson_probability(k, lam)
                if c >= q:
                    return k
            return max_k
        med = _pq(0.5, total)
        q25 = _pq(0.25, total)
        q75 = _pq(0.75, total)

    return {
        "expected_home": expected_home,
        "expected_away": expected_away,
        "expected_total": total,
        "dist": dist,
        "nb_size_r": None if r is None else float(r),
        "total_median": int(med),
        "total_q25_q75": f"{int(q25)}-{int(q75)}",
        **probs,
    }


def _dc_score_prob(hg: int, ag: int, lam: float, mu: float, rho: float, max_goals: int = 10) -> float:
    try:
        hg = int(hg)
        ag = int(ag)
        lam = max(float(lam), 0.0)
        mu = max(float(mu), 0.0)
        rho = float(rho)
        if hg < 0 or ag < 0 or hg > max_goals or ag > max_goals:
            return 1e-15
        s = 0.0
        p_hg_ag = 0.0
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                base = poisson_probability(i, lam) * poisson_probability(j, mu)
                tau = _dixon_coles_tau(i, j, lam, mu, rho)
                p = base * tau
                s += p
                if i == hg and j == ag:
                    p_hg_ag = p
        if s <= 0:
            return 1e-15
        return max(1e-15, min(1.0, p_hg_ag / s))
    except Exception:
        return 1e-15


def _tune_dixon_coles_rho_from_historical(historical_data: dict, max_goals: int = 10) -> float:
    samples: list[tuple[int, int, float, float]] = []
    for _, md in (historical_data or {}).items():
        try:
            score = (md or {}).get("Score", "")
            if ":" not in score:
                continue
            hg_s, ag_s = score.split(":", 1)
            hg = int(hg_s.strip())
            ag = int(ag_s.strip())
            if hg < 0 or ag < 0 or hg > max_goals or ag > max_goals:
                continue
            h2h = (md or {}).get("h2h_stats") or {}
            if not isinstance(h2h, dict):
                continue
            xg = h2h.get("xg")
            if not isinstance(xg, dict):
                continue
            lam = _safe_float(xg.get("home"))
            mu = _safe_float(xg.get("away"))
            if lam <= 0 or mu <= 0:
                continue
            samples.append((hg, ag, lam, mu))
        except Exception:
            continue

    if len(samples) < 30:
        return -0.10

    best_rho = -0.10
    best_ll = -1e100
    for rho in [x / 100.0 for x in range(-30, 31)]:
        ll = 0.0
        for hg, ag, lam, mu in samples:
            ll += math.log(_dc_score_prob(hg, ag, lam, mu, rho, max_goals=max_goals))
        if ll > best_ll:
            best_ll = ll
            best_rho = rho
    return float(best_rho)


def _fit_isotonic_calibration_from_historical(historical_data: dict, predictions_data: dict) -> dict[str, list[tuple[float, float]]]:
    """
    Fit isotonic regression calibration curves from historical predictions vs actual outcomes.
    
    Returns:
        Dictionary mapping outcome type to list of (predicted_prob, calibrated_prob) tuples
    """
    from collections import defaultdict
    
    # Collect predictions and outcomes
    predictions_by_outcome = {
        "home": [],
        "draw": [],
        "away": []
    }
    
    # Match predictions with actual outcomes from historical data
    for pred_id, pred_data in predictions_data.items():
        # Find corresponding historical match
        match_info = pred_data.get("match", {})
        home_name = match_info.get("home", "")
        away_name = match_info.get("away", "")
        pred_probs = pred_data.get("probs", {})
        
        if not home_name or not away_name:
            continue
        
        # Find historical match
        for hist_id, hist_data in historical_data.items():
            match_str = hist_data.get("Match", "")
            if "-" not in match_str:
                continue
            hist_home, hist_away = [s.strip() for s in match_str.split("-", 1)]
            
            if hist_home == home_name and hist_away == away_name:
                # Found match - get actual outcome
                score = hist_data.get("Score", "")
                if ":" not in score:
                    break
                hg_s, ag_s = score.split(":", 1)
                try:
                    hg = int(hg_s.strip())
                    ag = int(ag_s.strip())
                    
                    # Record prediction and actual outcome
                    p_h = pred_probs.get("home", 0.0)
                    p_d = pred_probs.get("draw", 0.0)
                    p_a = pred_probs.get("away", 0.0)
                    
                    actual_home = 1.0 if hg > ag else 0.0
                    actual_draw = 1.0 if hg == ag else 0.0
                    actual_away = 1.0 if hg < ag else 0.0
                    
                    predictions_by_outcome["home"].append((p_h, actual_home))
                    predictions_by_outcome["draw"].append((p_d, actual_draw))
                    predictions_by_outcome["away"].append((p_a, actual_away))
                except (ValueError, TypeError):
                    pass
                break
    
    # Fit isotonic regression for each outcome type
    calibration_curves = {}
    
    for outcome_type, pred_actual_pairs in predictions_by_outcome.items():
        if len(pred_actual_pairs) < 30:  # Need minimum samples
            continue
        
        # Sort by predicted probability
        pred_actual_pairs.sort(key=lambda x: x[0])
        
        # Group into bins and compute actual frequencies
        bins = {}
        bin_size = max(10, len(pred_actual_pairs) // 20)  # ~20 bins
        
        for i in range(0, len(pred_actual_pairs), bin_size):
            chunk = pred_actual_pairs[i:i + bin_size]
            if not chunk:
                continue
            
            avg_pred = sum(p for p, _ in chunk) / len(chunk)
            avg_actual = sum(a for _, a in chunk) / len(chunk)
            bins[avg_pred] = avg_actual
        
        # Apply isotonic regression (monotonic increasing)
        sorted_bins = sorted(bins.items())
        calibrated = []
        current_max = 0.0
        
        for pred_prob, actual_freq in sorted_bins:
            if actual_freq > current_max:
                current_max = actual_freq
            calibrated.append((pred_prob, current_max))
        
        if calibrated:
            calibration_curves[outcome_type] = calibrated
    
    return calibration_curves


def _apply_isotonic_calibration(p: float, calibration_curve: list[tuple[float, float]] | None) -> float:
    """
    Apply isotonic calibration curve to a probability.
    
    Args:
        p: Original probability
        calibration_curve: List of (predicted, calibrated) tuples from isotonic regression
    """
    if calibration_curve is None or len(calibration_curve) == 0:
        return p
    
    p = max(0.0, min(1.0, float(p)))
    
    # Find the two points to interpolate between
    if p <= calibration_curve[0][0]:
        return calibration_curve[0][1]
    if p >= calibration_curve[-1][0]:
        return calibration_curve[-1][1]
    
    # Linear interpolation
    for i in range(len(calibration_curve) - 1):
        p1, c1 = calibration_curve[i]
        p2, c2 = calibration_curve[i + 1]
        
        if p1 <= p <= p2:
            if p2 == p1:
                return c1
            t = (p - p1) / (p2 - p1)
            return c1 + t * (c2 - c1)
    
    return p


def _create_historical_data_fingerprint(historical_data: dict) -> str:
    """
    Create a fingerprint of historical data to detect changes.
    Uses match identifiers and dates (not scores) to detect when data is added/updated.
    """
    import hashlib
    import json
    
    # Create a stable representation: (match_id, match, date) tuples sorted
    match_keys = []
    for match_id, match_data in historical_data.items():
        match_str = match_data.get("Match", "")
        date_str = match_data.get("Date", "")
        # Use match and date, not score (score changes don't affect calibration fitting)
        match_keys.append((str(match_id), match_str, date_str))
    
    # Sort for consistent hashing
    match_keys.sort()
    
    # Create hash
    data_str = json.dumps(match_keys, sort_keys=True)
    return hashlib.sha256(data_str.encode()).hexdigest()[:16]


def _load_cached_calibration(cache_file) -> dict | None:
    """Load cached calibration curves from file."""
    try:
        if not cache_file.exists():
            return None
        import json
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return data
    except Exception:
        return None


def _save_cached_calibration(cache_file, calibration_curves: dict, fingerprint: str) -> None:
    """Save calibration curves with fingerprint to cache file."""
    try:
        import json
        cache_data = {
            "fingerprint": fingerprint,
            "calibration_curves": calibration_curves,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        cache_file.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")
    except Exception as e:
        logging.warning(f"Could not save calibration cache: {e}")


def fit_isotonic_calibration(historical_data: dict, league: str) -> dict[str, list[tuple[float, float]]]:
    """
    Fit isotonic calibration curves for a league using historical data simulations.
    
    Simulates predictions for historical matches using only data available before each match.
    This provides immediate calibration without waiting for future predictions.
    
    Uses caching to avoid redundant refitting when historical data hasn't changed.
    """
    from bet_helper.storage import data_dir
    from pathlib import Path
    
    # Create fingerprint of historical data
    data_fingerprint = _create_historical_data_fingerprint(historical_data)
    
    # Check for cached calibration
    cache_file = data_dir() / f"calibration_cache_{league}.json"
    cached = _load_cached_calibration(cache_file)
    
    if cached and cached.get("fingerprint") == data_fingerprint:
        # Data unchanged - reuse cached calibration
        calibration_curves = cached.get("calibration_curves", {})
        if calibration_curves:
            logging.debug(f"Using cached isotonic calibration for {league} (fingerprint: {data_fingerprint})")
            return calibration_curves
    
    # Data changed or no cache - fit new calibration
    try:
        # First try historical simulations (better - immediate calibration)
        cal = fit_isotonic_calibration_from_historical_simulations(historical_data, league)
        if cal:
            # Cache the calibration with fingerprint
            _save_cached_calibration(cache_file, cal, data_fingerprint)
            logging.info(f"Fitted and cached isotonic calibration for {league} (fingerprint: {data_fingerprint})")
            return cal
        
        # Fallback: try using saved predictions if available
        from bet_helper.storage import predictions_path, load_json
        pred_path = predictions_path(league)
        predictions_data = load_json(pred_path) or {}
        
        # Convert list to dict for matching
        predictions_dict = {}
        if isinstance(predictions_data, list):
            for idx, pred in enumerate(predictions_data):
                predictions_dict[str(idx)] = pred
        else:
            predictions_dict = predictions_data
        
        fallback_cal = _fit_isotonic_calibration_from_historical(historical_data, predictions_dict)
        if fallback_cal:
            # Cache the fallback calibration too
            _save_cached_calibration(cache_file, fallback_cal, data_fingerprint)
        return fallback_cal
    except Exception as e:
        logging.warning(f"Could not fit isotonic calibration for {league}: {e}")
        return {}


def fit_isotonic_calibration_from_historical_simulations(
    historical_data: dict,
    league: str,
    min_matches: int = 50,
    reference_date: datetime | None = None,
) -> dict[str, list[tuple[float, float]]]:
    """
    Fit isotonic calibration by simulating predictions for historical matches.
    
    For each historical match:
    1. Filter historical data to only include matches BEFORE this match's date
    2. Build teams/form/H2H stats using only past data (time-aware)
    3. Generate prediction using the model
    4. Compare with actual outcome
    5. Collect all prediction-outcome pairs for calibration
    
    Args:
        historical_data: All historical matches
        league: League identifier
        min_matches: Minimum number of matches needed before starting simulations
        reference_date: Reference date for date parsing (defaults to now)
    
    Returns:
        Dictionary mapping outcome type to list of (predicted_prob, calibrated_prob) tuples
    """
    from bet_helper.models import TeamData, MatchData
    
    if reference_date is None:
        reference_date = datetime.now()
    
    # Parse and sort historical matches chronologically
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
    
    # Sort chronologically (oldest first)
    matches_with_dates.sort(key=lambda x: x[0])
    
    if len(matches_with_dates) < min_matches:
        logging.debug(f"Not enough historical matches ({len(matches_with_dates)} < {min_matches}) for calibration")
        return {}
    
    # Collect prediction-outcome pairs
    predictions_by_outcome = {
        "home": [],
        "draw": [],
        "away": []
    }
    
    # Process each match, using only data available before that match
    for idx, (match_date, match_id, match_data) in enumerate(matches_with_dates):
        if idx < min_matches:
            # Skip early matches - need minimum history for reliable predictions
            continue
        
        try:
            # Filter historical data to only include matches BEFORE this match's date
            past_data = {}
            for hist_id, hist_match in historical_data.items():
                hist_date_str = hist_match.get("Date", "")
                if not hist_date_str:
                    continue
                hist_date = _parse_dd_mmm_yyyy_season_aware(hist_date_str, reference_date=reference_date)
                if hist_date and hist_date < match_date:
                    past_data[hist_id] = hist_match
            
            if len(past_data) < min_matches:
                # Not enough past data for this match
                continue
            
            # Extract match info
            match_str = match_data.get("Match", "")
            home_name, away_name = [s.strip() for s in match_str.split("-", 1)]
            
            # Build teams/form using only past data (time-aware)
            teams = {}
            for hist_id, hist_match in past_data.items():
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
                    
                    if hist_home not in teams:
                        teams[hist_home] = TeamData(name=hist_home)
                    if hist_away not in teams:
                        teams[hist_away] = TeamData(name=hist_away)
                    
                    m = MatchData(date=hist_date_str, home_team=hist_home, away_team=hist_away, home_goals=hg, away_goals=ag)
                    teams[hist_home].add_match(m)
                    teams[hist_away].add_match(m)
                except Exception:
                    continue
            
            if home_name not in teams or away_name not in teams:
                # Teams not found in past data
                continue
            
            home = teams[home_name]
            away = teams[away_name]
            
            # Build H2H index using only past data
            team_h2h_index = _build_team_h2h_index(past_data, reference_date=match_date)
            
            # Tune Dixon-Coles rho using only past data
            dc_rho = _tune_dixon_coles_rho_from_historical(past_data, max_goals=10)
            
            # Generate prediction using time-aware data (no calibration applied)
            # We want raw predictions to calibrate them
            ph_raw, pd_raw, pa_raw, _, _ = predict_match_poisson_with_weighted_h2h(
                home_name, away_name, home.form, away.form,
                team_h2h_index=team_h2h_index,
                dc_rho=dc_rho,
                isotonic_calibration=None,  # No calibration - we're fitting it!
            )
            
            # Apply temperature and asymmetric calibration (but not isotonic yet)
            ph, pd, pa = _temperature_scale_probs(ph_raw, pd_raw, pa_raw, temperature=0.95)
            ph, pd, pa = _asymmetric_calibration(ph, pd, pa)
            
            # Extract actual outcome
            score = match_data.get("Score", "")
            if ":" not in score:
                continue
            hg_s, ag_s = score.split(":", 1)
            try:
                hg = int(hg_s.strip())
                ag = int(ag_s.strip())
                
                actual_home = 1.0 if hg > ag else 0.0
                actual_draw = 1.0 if hg == ag else 0.0
                actual_away = 1.0 if hg < ag else 0.0
                
                # Record prediction and actual outcome
                predictions_by_outcome["home"].append((ph, actual_home))
                predictions_by_outcome["draw"].append((pd, actual_draw))
                predictions_by_outcome["away"].append((pa, actual_away))
            except (ValueError, TypeError):
                continue
            
        except Exception as e:
            logging.debug(f"Error simulating prediction for match {match_id}: {e}")
            continue
    
    # Fit isotonic regression for each outcome type
    return _fit_isotonic_curves_from_pairs(predictions_by_outcome)


def _fit_isotonic_curves_from_pairs(predictions_by_outcome: dict[str, list[tuple[float, float]]]) -> dict[str, list[tuple[float, float]]]:
    """
    Fit isotonic regression curves from prediction-outcome pairs.
    
    Args:
        predictions_by_outcome: Dictionary mapping outcome type to list of (predicted, actual) tuples
    
    Returns:
        Dictionary mapping outcome type to list of (predicted_prob, calibrated_prob) tuples
    """
    calibration_curves = {}
    
    for outcome_type, pred_actual_pairs in predictions_by_outcome.items():
        if len(pred_actual_pairs) < 30:  # Need minimum samples
            continue
        
        # Sort by predicted probability
        pred_actual_pairs.sort(key=lambda x: x[0])
        
        # Group into bins and compute actual frequencies
        bins = {}
        bin_size = max(10, len(pred_actual_pairs) // 20)  # ~20 bins
        
        for i in range(0, len(pred_actual_pairs), bin_size):
            chunk = pred_actual_pairs[i:i + bin_size]
            if not chunk:
                continue
            
            avg_pred = sum(p for p, _ in chunk) / len(chunk)
            avg_actual = sum(a for _, a in chunk) / len(chunk)
            bins[avg_pred] = avg_actual
        
        # Apply isotonic regression (monotonic increasing)
        sorted_bins = sorted(bins.items())
        calibrated = []
        current_max = 0.0
        
        for pred_prob, actual_freq in sorted_bins:
            if actual_freq > current_max:
                current_max = actual_freq
            calibrated.append((pred_prob, current_max))
        
        if calibrated:
            calibration_curves[outcome_type] = calibrated
    
    return calibration_curves


def _find_team_in_dict(team_name: str, teams_dict: dict) -> str | None:
    if team_name in teams_dict:
        return team_name
    for team_key in teams_dict.keys():
        if team_name.lower() == team_key.lower():
            return team_key
    # substring fallback using normalized strings
    def norm(s: str) -> str:
        return _normalize_club_name_for_stats(s)
    n = norm(team_name)
    for team_key in teams_dict.keys():
        nk = norm(team_key)
        if n and nk and (n in nk or nk in n):
            return team_key
    return None


def predict_match_poisson_with_weighted_h2h(
    home_name: str,
    away_name: str,
    home_form: list,
    away_form: list,
    team_h2h_index: dict | None = None,
    dc_rho: float = -0.10,
    isotonic_calibration: dict[str, list[tuple[float, float]]] | None = None,
) -> tuple[float, float, float, str, dict]:
    """
    Predict match outcome using Poisson model with Dixon-Coles adjustment.
    
    Args:
        home_name: Home team name
        away_name: Away team name
        home_form: List of MatchData objects for home team (newest first)
        away_form: List of MatchData objects for away team (newest first)
        team_h2h_index: Head-to-head statistics index
        dc_rho: Dixon-Coles rho parameter
    """
    from bet_helper.models import calculate_team_form
    
    # Form expectations
    _, _, _, hsc_home_5, _, hco_home_5, _ = calculate_team_form(home_form[:5], home_name, n=5)
    _, _, _, _, asc_away_5, _, aco_away_5 = calculate_team_form(away_form[:5], away_name, n=5)
    _, _, _, hsc_home_10, _, hco_home_10, _ = calculate_team_form(home_form[:10], home_name, n=10)
    _, _, _, _, asc_away_10, _, aco_away_10 = calculate_team_form(away_form[:10], away_name, n=10)

    if hsc_home_5 > 0 or aco_away_5 > 0:
        form_home = (hsc_home_5 + aco_away_5) / 2
    elif hsc_home_10 > 0 or aco_away_10 > 0:
        form_home = (hsc_home_10 + aco_away_10) / 2
    else:
        form_home = 1.5

    if asc_away_5 > 0 or hco_home_5 > 0:
        form_away = (asc_away_5 + hco_home_5) / 2
    elif asc_away_10 > 0 or hco_home_10 > 0:
        form_away = (asc_away_10 + hco_home_10) / 2
    else:
        form_away = 1.2

    expected_home_goals = float(form_home)
    expected_away_goals = float(form_away)

    xg_note = ""
    xg_home = None
    xg_away = None
    if team_h2h_index:
        home_key = _normalize_club_name_for_stats(home_name)
        away_key = _normalize_club_name_for_stats(away_name)
        home_entries = team_h2h_index.get(home_key, [])
        away_entries = team_h2h_index.get(away_key, [])
        home_prof_home = _weighted_stat_profile(home_entries, "xg_for", "xg_against", half_life_days=30.0, home_only=True) or \
                         _weighted_stat_profile(home_entries, "xg_for", "xg_against", half_life_days=30.0, home_only=None)
        away_prof_away = _weighted_stat_profile(away_entries, "xg_for", "xg_against", half_life_days=30.0, home_only=False) or \
                         _weighted_stat_profile(away_entries, "xg_for", "xg_against", half_life_days=30.0, home_only=None)
        if home_prof_home and away_prof_away:
            xg_home = (float(home_prof_home["for"]) + float(away_prof_away["against"])) / 2
            xg_away = (float(away_prof_away["for"]) + float(home_prof_home["against"])) / 2
            xg_weight = 0.7
            expected_home_goals = (xg_weight * xg_home) + ((1 - xg_weight) * expected_home_goals)
            expected_away_goals = (xg_weight * xg_away) + ((1 - xg_weight) * expected_away_goals)
            xg_note = f" (xG-weighted: {xg_home:.2f}-{xg_away:.2f})"

    expected_home_goals = max(float(expected_home_goals), 0.1)
    expected_away_goals = max(float(expected_away_goals), 0.1)

    max_goals = 10
    rho = float(dc_rho)
    score_p = []
    for hg in range(max_goals + 1):
        row = []
        for ag in range(max_goals + 1):
            base = poisson_probability(hg, expected_home_goals) * poisson_probability(ag, expected_away_goals)
            row.append(base * _dixon_coles_tau(hg, ag, expected_home_goals, expected_away_goals, rho))
        score_p.append(row)
    s = sum(sum(r) for r in score_p) or 1.0
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            score_p[hg][ag] /= s

    p_home = p_draw = p_away = 0.0
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = score_p[hg][ag]
            if hg > ag:
                p_home += p
            elif hg == ag:
                p_draw += p
            else:
                p_away += p

    p_home_raw, p_draw_raw, p_away_raw = p_home, p_draw, p_away
    p_home, p_draw, p_away = _temperature_scale_probs(p_home_raw, p_draw_raw, p_away_raw, temperature=0.95)
    p_home, p_draw, p_away = _asymmetric_calibration(p_home, p_draw, p_away)
    
    # Apply isotonic calibration if available
    if isotonic_calibration:
        p_home = _apply_isotonic_calibration(p_home, isotonic_calibration.get("home"))
        p_draw = _apply_isotonic_calibration(p_draw, isotonic_calibration.get("draw"))
        p_away = _apply_isotonic_calibration(p_away, isotonic_calibration.get("away"))
        
        # Renormalize
        total = p_home + p_draw + p_away
        if total > 0:
            p_home = max(0.0, min(1.0, p_home / total))
            p_draw = max(0.0, min(1.0, p_draw / total))
            p_away = max(0.0, min(1.0, p_away / total))

    total_pmf: dict[int, float] = {}
    btts = 0.0
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = score_p[hg][ag]
            total = int(hg + ag)
            total_pmf[total] = total_pmf.get(total, 0.0) + p
            if hg >= 1 and ag >= 1:
                btts += p

    def pmf_tail_over(line: float) -> float:
        k0 = int(math.floor(float(line))) + 1
        return max(0.0, min(1.0, sum(p for k, p in total_pmf.items() if int(k) >= k0)))

    def pmf_quantile(q: float) -> int:
        c = 0.0
        for k in sorted(total_pmf.keys()):
            c += total_pmf[k]
            if c >= q:
                return int(k)
        return int(sorted(total_pmf.keys())[-1])

    total_med = pmf_quantile(0.5)
    total_q25 = pmf_quantile(0.25)
    total_q75 = pmf_quantile(0.75)

    margin_pmf: dict[int, float] = {}
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            d = int(hg - ag)
            margin_pmf[d] = margin_pmf.get(d, 0.0) + score_p[hg][ag]

    def pmf_q_margin(q: float) -> int:
        c = 0.0
        for d in sorted(margin_pmf.keys()):
            c += margin_pmf[d]
            if c >= q:
                return int(d)
        return int(sorted(margin_pmf.keys())[-1])

    margin_med = pmf_q_margin(0.5)
    margin_q25 = pmf_q_margin(0.25)
    margin_q75 = pmf_q_margin(0.75)

    extras = {
        "expected_goals_home": round(expected_home_goals, 3),
        "expected_goals_away": round(expected_away_goals, 3),
        "model": "poisson_dc",
        "dc_rho": rho,
        "p_home_raw": round(float(p_home_raw), 3),
        "p_draw_raw": round(float(p_draw_raw), 3),
        "p_away_raw": round(float(p_away_raw), 3),
        "calibration_temperature": 0.95,
        "xg_blend_note": xg_note.strip(),
        "xg_home_used": None if xg_home is None else round(float(xg_home), 3),
        "xg_away_used": None if xg_away is None else round(float(xg_away), 3),
        "p_btts": round(float(btts), 3),
        "p_over_0_5": round(pmf_tail_over(0.5), 3),
        "p_over_1_5": round(pmf_tail_over(1.5), 3),
        "p_over_2_5": round(pmf_tail_over(2.5), 3),
        "p_over_3_5": round(pmf_tail_over(3.5), 3),
        "total_goals_median": int(total_med),
        "total_goals_q25_q75": f"{int(total_q25)}-{int(total_q75)}",
        "margin_median": int(margin_med),
        "margin_q25_q75": f"{int(margin_q25)}-{int(margin_q75)}",
    }

    expected_result = f"Expected result: {expected_home_goals:.1f}-{expected_away_goals:.1f} (Poisson+DC, rho={rho:+.2f}){xg_note}"
    return p_home, p_draw, p_away, expected_result, extras


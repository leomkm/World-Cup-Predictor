# simulate.py — Elo + Poisson blended match outcome sampling
# Uses Elo-derived result probabilities blended with Poisson-based probabilities
import csv
import os
import numpy as np
import pandas as pd
from math import exp, factorial

from venue import is_neutral_match
from blend import compute_elo_weight


def _default_elo_path():
    base = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(base, '..', 'data', 'elos.csv'))


def load_elos(path=None):
    elos = {}
    p = path or _default_elo_path()
    try:
        with open(p, newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    elos[row['team'].strip()] = float(row['elo'])
                except Exception:
                    continue
    except FileNotFoundError:
        return {}
    return elos


def elo_expected_score(h_elo, a_elo):
    d = h_elo - a_elo
    return 1.0 / (1.0 + 10 ** (-d / 400.0))


def match_probs_from_elos(h_elo, a_elo, base_draw=0.23, draw_slope=0.0009):
    S = elo_expected_score(h_elo, a_elo)
    diff = abs(h_elo - a_elo)
    p_draw = max(0.05, base_draw - draw_slope * diff)
    p_home = S * (1 - p_draw)
    p_away = (1 - S) * (1 - p_draw)
    s = p_home + p_draw + p_away
    return p_home / s, p_draw / s, p_away / s


def _poisson_pmf(k, mu):
    # safe Poisson pmf for scalar k, mu
    try:
        return exp(-mu) * (mu ** k) / factorial(k)
    except Exception:
        return 0.0


def match_result_probs(mu_home, mu_away, max_goals=10):
    """Compute P(home win), P(draw), P(away win) from independent Poisson goals up to max_goals."""
    # build pmfs
    pmf_home = [_poisson_pmf(k, mu_home) for k in range(max_goals + 1)]
    pmf_away = [_poisson_pmf(k, mu_away) for k in range(max_goals + 1)]
    # joint and sums
    p_home_win = 0.0
    p_draw = 0.0
    p_away_win = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = pmf_home[i] * pmf_away[j]
            if i > j:
                p_home_win += p
            elif i == j:
                p_draw += p
            else:
                p_away_win += p
    s = p_home_win + p_draw + p_away_win
    if s <= 0:
        return 1/3, 1/3, 1/3
    return p_home_win / s, p_draw / s, p_away_win / s


def predict_score(
    home_model,
    away_model,
    home_elo,
    away_elo,
    home_form,
    away_form,
    neutral=True
):
    match = pd.DataFrame({
        "home_elo": [home_elo],
        "away_elo": [away_elo],
        "home_recent_goals_for": [home_form["goals_for"]],
        "home_recent_goals_against": [home_form["goals_against"]],
        "home_recent_points": [home_form["points"]],
        "away_recent_goals_for": [away_form["goals_for"]],
        "away_recent_goals_against": [away_form["goals_against"]],
        "away_recent_points": [away_form["points"]],
        "home_attack": [home_form["attack"]],
        "away_attack": [away_form["attack"]],
        "home_defense": [home_form["defense"]],
        "away_defense": [away_form["defense"]],
        "neutral": [int(neutral)]
    })

    # MODEL EXPECTED GOALS
    model_home = float(home_model.predict(match)[0])
    model_away = float(away_model.predict(match)[0])

    # REGRESS TO GLOBAL AVERAGE
    league_home = 1.35
    league_away = 1.10

    expected_home = model_home * 0.65 + league_home * 0.35
    expected_away = model_away * 0.65 + league_away * 0.35

    # ELO ADVANTAGE (kept for goals magnitude)
    elo_diff = home_elo - away_elo
    elo_multiplier = 10 ** (elo_diff / 2200)
    elo_multiplier = np.clip(elo_multiplier, 0.88, 1.14)
    expected_home *= elo_multiplier
    expected_away /= elo_multiplier

    # SMALL ATTACK/DEFENCE EFFECT
    attack_home = home_form["attack"] / max(away_form["defense"], 0.8)
    attack_away = away_form["attack"] / max(home_form["defense"], 0.8)

    expected_home *= (0.90 + 0.10 * np.clip(attack_home, 0.7, 1.5))
    expected_away *= (0.90 + 0.10 * np.clip(attack_away, 0.7, 1.5))

    # FINAL CAPS
    expected_home = np.clip(expected_home, 0.25, 3.0)
    expected_away = np.clip(expected_away, 0.20, 3.0)

    return expected_home, expected_away

def simulate_match(
    home,
    away,
    home_elo,
    away_elo,
    home_model,
    away_model,
    home_form,
    away_form,
    elos=None,
    elo_weight=None,
    pair_cache=None,
    neutral=None,
    max_goals=10
):
    # If a pair_cache is provided and contains this ordered pair, use it
    if pair_cache is not None and (home, away) in pair_cache:
        entry = pair_cache[(home, away)]
        expected_home = entry["exp_home"]
        expected_away = entry["exp_away"]
        p_home = entry["p_home"]
        p_draw = entry["p_draw"]
        p_away = entry["p_away"]
        h_elo = entry.get("h_elo", home_elo)
        a_elo = entry.get("a_elo", away_elo)
    else:
        # original flow: compute expected_home, expected_away and blended probs
        h_elo = elos.get(home, home_elo) if elos else home_elo
        a_elo = elos.get(away, away_elo) if elos else away_elo

        if neutral is None:
            neutral = is_neutral_match(home, away)

        expected_home, expected_away = predict_score(
            home_model, away_model, home_elo, away_elo, home_form, away_form, neutral=neutral
        )

        if elo_weight is None:
            elo_weight = compute_elo_weight(
                home_form.get("matches_played", 0),
                away_form.get("matches_played", 0)
            )

        p_e_home, p_e_draw, p_e_away = match_probs_from_elos(h_elo, a_elo)
        p_p_home, p_p_draw, p_p_away = match_result_probs(expected_home, expected_away, max_goals=max_goals)
        p_home = elo_weight * p_e_home + (1 - elo_weight) * p_p_home
        p_draw = elo_weight * p_e_draw + (1 - elo_weight) * p_p_draw
        p_away = elo_weight * p_e_away + (1 - elo_weight) * p_p_away
        s = p_home + p_draw + p_away
        if s <= 0:
            p_home = p_draw = p_away = 1/3
        else:
            p_home /= s; p_draw /= s; p_away /= s

    # sample outcome using p_home/p_draw/p_away (same logic as before)
    r = np.random.random()
    if r < p_home:
        outcome = 'home'
    elif r < p_home + p_draw:
        outcome = 'draw'
    else:
        outcome = 'away'

    # sample goals consistent with outcome using expected_home/expected_away
    if outcome == 'home':
        away_goals = np.random.poisson(expected_away)
        home_goals = np.random.poisson(expected_home)
        if home_goals <= away_goals:
            home_goals = away_goals + 1
        winner = home
    elif outcome == 'draw':
        g = np.random.poisson((expected_home + expected_away) / 2.0)
        home_goals = away_goals = g
        probability = (1 / (1 + 10 ** ((a_elo - h_elo) / 400)))
        winner = home if np.random.random() < probability else away
    else:
        home_goals = np.random.poisson(expected_home)
        away_goals = np.random.poisson(expected_away)
        if away_goals <= home_goals:
            away_goals = home_goals + 1
        winner = away

    return {"home": home, "away": away, "score": f"{home_goals}-{away_goals}", "winner": winner}
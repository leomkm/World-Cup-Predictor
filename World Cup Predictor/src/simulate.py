# Updated simulate.py: use Elo-derived match result probabilities
# and load default Elo ratings from data/elos.csv
import csv
import os
import numpy as np
import pandas as pd


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


def predict_score(
    home_model,
    away_model,
    home_elo,
    away_elo,
    home_form,
    away_form
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
        "away_defense": [away_form["defense"]]
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
    elos=None
):
    # ensure elos dict available
    if elos is None:
        elos = load_elos()

    # get elo ratings (fallback to provided numbers or 1500)
    h_elo = elos.get(home, home_elo or 1500)
    a_elo = elos.get(away, away_elo or 1500)

    # compute expected goals from existing model
    expected_home, expected_away = predict_score(
        home_model,
        away_model,
        home_elo,
        away_elo,
        home_form,
        away_form
    )

    # get match result probabilities from Elo
    p_home, p_draw, p_away = match_probs_from_elos(h_elo, a_elo)

    r = np.random.random()

    if r < p_home:
        # force a home win: sample away goals then ensure home > away
        away_goals = np.random.poisson(expected_away)
        home_goals = np.random.poisson(expected_home)
        if home_goals <= away_goals:
            home_goals = away_goals + 1
        winner = home
    elif r < p_home + p_draw:
        # force a draw: sample one value and set both equal
        g = np.random.poisson((expected_home + expected_away) / 2.0)
        home_goals = g
        away_goals = g
        winner = None  # draw
    else:
        # force away win
        home_goals = np.random.poisson(expected_home)
        away_goals = np.random.poisson(expected_away)
        if away_goals <= home_goals:
            away_goals = home_goals + 1
        winner = away

    # If this is a knockout and a draw happened, decide on penalties using Elo
    if winner is None:
        probability = (1 / (1 + 10 ** ((a_elo - h_elo) / 400)))
        if np.random.random() < probability:
            winner = home
        else:
            winner = away

    return {"home": home, "away": away, "score": f"{home_goals}-{away_goals}", "winner": winner}

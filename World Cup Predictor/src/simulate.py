import numpy as np
import pandas as pd


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

        "home_recent_goals_for": [
            home_form["goals_for"]
        ],

        "home_recent_goals_against": [
            home_form["goals_against"]
        ],

        "home_recent_points": [
            home_form["points"]
        ],

        "away_recent_goals_for": [
            away_form["goals_for"]
        ],

        "away_recent_goals_against": [
            away_form["goals_against"]
        ],

        "away_recent_points": [
            away_form["points"]
        ],

        "home_attack": [
            home_form["attack"]
        ],

        "away_attack": [
            away_form["attack"]
        ],

        "home_defense": [
            home_form["defense"]
        ],

        "away_defense": [
            away_form["defense"]
        ]

    })


    # ==========================
    # MODEL EXPECTED GOALS
    # ==========================

    model_home = float(
        home_model.predict(match)[0]
    )

    model_away = float(
        away_model.predict(match)[0]
    )


    # ==========================
    # REGRESS TO GLOBAL AVERAGE
    # stops crazy 5-0 games
    # ==========================

    league_home = 1.35
    league_away = 1.10


    expected_home = (
        model_home * 0.65 +
        league_home * 0.35
    )


    expected_away = (
        model_away * 0.65 +
        league_away * 0.35
    )


    # ==========================
    # ELO ADVANTAGE
    # ==========================

    elo_diff = home_elo - away_elo


    # Every 100 Elo ≈ 5% scoring difference
    elo_multiplier = 10 ** (
        elo_diff / 2200
    )


    elo_multiplier = np.clip(
        elo_multiplier,
        0.88,
        1.14
    )


    expected_home *= elo_multiplier
    expected_away /= elo_multiplier



    # ==========================
    # SMALL ATTACK/DEFENCE EFFECT
    # ==========================

    attack_home = (
        home_form["attack"] /
        max(away_form["defense"],0.8)
    )


    attack_away = (
        away_form["attack"] /
        max(home_form["defense"],0.8)
    )


    # only 10% influence
    expected_home *= (
        0.90 +
        0.10 * np.clip(
            attack_home,
            0.7,
            1.5
        )
    )


    expected_away *= (
        0.90 +
        0.10 * np.clip(
            attack_away,
            0.7,
            1.5
        )
    )


    # ==========================
    # FINAL CAPS
    # ==========================

    expected_home = np.clip(
        expected_home,
        0.25,
        3.0
    )


    expected_away = np.clip(
        expected_away,
        0.20,
        3.0
    )


    # ==========================
    # POISSON
    # ==========================

    home_goals = np.random.poisson(
        expected_home
    )

    away_goals = np.random.poisson(
        expected_away
    )


    return home_goals, away_goals



def simulate_match(
    home,
    away,
    home_elo,
    away_elo,
    home_model,
    away_model,
    home_form,
    away_form
):

    home_goals, away_goals = predict_score(
        home_model,
        away_model,
        home_elo,
        away_elo,
        home_form,
        away_form
    )


    if home_goals > away_goals:

        winner = home

    elif away_goals > home_goals:

        winner = away

    else:

        # Elo decides penalties
        probability = (
            1 /
            (
                1 +
                10 ** (
                    (away_elo-home_elo)
                    /
                    400
                )
            )
        )


        if np.random.random() < probability:
            winner = home
        else:
            winner = away


    return {
        "home": home,
        "away": away,
        "score": f"{home_goals}-{away_goals}",
        "winner": winner
    }
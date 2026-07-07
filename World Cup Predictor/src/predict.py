import pandas as pd


def predict_match(model, home_elo, away_elo):
    match = pd.DataFrame({
        "home_elo": [2000],
        "away_elo": [1900],

        "home_recent_goals_for": [1.8],
        "home_recent_goals_against": [0.9],
        "home_recent_points": [2.2],

        "away_recent_goals_for": [1.4],
        "away_recent_goals_against": [1.3],
        "away_recent_points": [1.6],

        "home_attack": [2.2],
        "away_attack": [1.5],

        "home_defense": [0.6],
        "away_defense": [1.1]
    })

    predicted_goals = model.predict(match)

    return predicted_goals[0]
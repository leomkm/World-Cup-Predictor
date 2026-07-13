def create_features(df):
    X = df[
        [
            "home_elo",
            "away_elo",

            "home_recent_goals_for",
            "home_recent_goals_against",
            "home_recent_points",

            "away_recent_goals_for",
            "away_recent_goals_against",
            "away_recent_points",

            "home_attack",
            "away_attack",
            "home_defense",
            "away_defense",

            "neutral"
        ]
    ]

    y_home = df["home_score"]
    y_away = df["away_score"]

    return X, y_home, y_away
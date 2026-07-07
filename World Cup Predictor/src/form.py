from collections import defaultdict, deque


def add_form_features(df):

    df = df.sort_values("date")

    history = defaultdict(lambda: deque(maxlen=5))

    home_goals_for = []
    home_goals_against = []
    home_points = []

    away_goals_for = []
    away_goals_against = []
    away_points = []

    for _, row in df.iterrows():

        home = row["home_team"]
        away = row["away_team"]

        # Get previous form BEFORE this match
        home_matches = list(history[home])
        away_matches = list(history[away])

        # Home team
        home_goals_for.append(
            sum(match["gf"] for match in home_matches) / max(len(home_matches), 1)
        )

        home_goals_against.append(
            sum(match["ga"] for match in home_matches) / max(len(home_matches), 1)
        )

        home_points.append(
            sum(match["points"] for match in home_matches) / max(len(home_matches), 1)
        )

        # Away team
        away_goals_for.append(
            sum(match["gf"] for match in away_matches) / max(len(away_matches), 1)
        )

        away_goals_against.append(
            sum(match["ga"] for match in away_matches) / max(len(away_matches), 1)
        )

        away_points.append(
            sum(match["points"] for match in away_matches) / max(len(away_matches), 1)
        )

        # Update histories AFTER the match

        if row["home_score"] > row["away_score"]:
            home_result = 3
            away_result = 0

        elif row["home_score"] < row["away_score"]:
            home_result = 0
            away_result = 3

        else:
            home_result = 1
            away_result = 1


        history[home].append({
            "gf": row["home_score"],
            "ga": row["away_score"],
            "points": home_result
        })

        history[away].append({
            "gf": row["away_score"],
            "ga": row["home_score"],
            "points": away_result
        })


    df["home_recent_goals_for"] = home_goals_for
    df["home_recent_goals_against"] = home_goals_against
    df["home_recent_points"] = home_points

    df["away_recent_goals_for"] = away_goals_for
    df["away_recent_goals_against"] = away_goals_against
    df["away_recent_points"] = away_points

    return df
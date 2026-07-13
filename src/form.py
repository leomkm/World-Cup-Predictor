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


def get_current_form(df):
    """Returns each team's rolling form (goals for/against, points) as of
    RIGHT NOW - i.e. after processing every match in the dataset.

    This is what should be used to seed World Cup simulations. Previously,
    team_profiles.py hardcoded these values to 0 for every team, meaning
    every simulated match ignored real recent form entirely.
    """

    df = df.sort_values("date")

    history = defaultdict(lambda: deque(maxlen=5))

    for _, row in df.iterrows():

        home = row["home_team"]
        away = row["away_team"]

        if row["home_score"] > row["away_score"]:
            home_result, away_result = 3, 0
        elif row["home_score"] < row["away_score"]:
            home_result, away_result = 0, 3
        else:
            home_result, away_result = 1, 1

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

    current = {}
    teams = set(df["home_team"]).union(set(df["away_team"]))

    for team in teams:
        matches = list(history[team])
        n = max(len(matches), 1)
        current[team] = {
            "goals_for": sum(m["gf"] for m in matches) / n,
            "goals_against": sum(m["ga"] for m in matches) / n,
            "points": sum(m["points"] for m in matches) / n,
        }

    return current
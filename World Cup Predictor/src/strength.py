import pandas as pd


LAST_N = 10


def get_team_games(df, team):

    home = df[df["home_team"] == team].copy()
    away = df[df["away_team"] == team].copy()


    home["goals_for"] = home["home_score"]
    home["goals_against"] = home["away_score"]
    home["opponent"] = home["away_team"]


    away["goals_for"] = away["away_score"]
    away["goals_against"] = away["home_score"]
    away["opponent"] = away["home_team"]


    games = pd.concat(
        [
            home[
                [
                    "date",
                    "goals_for",
                    "goals_against",
                    "opponent"
                ]
            ],
            away[
                [
                    "date",
                    "goals_for",
                    "goals_against",
                    "opponent"
                ]
            ]
        ]
    )


    return games.sort_values(
        "date",
        ascending=False
    ).head(LAST_N)



def calculate_basic_strength(df):

    teams = set(
        df["home_team"]
    ).union(
        set(df["away_team"])
    )


    defense = {}


    for team in teams:

        games = get_team_games(
            df,
            team
        )


        if len(games):

            defense[team] = (
                games["goals_against"]
                .mean()
            )

        else:
            defense[team] = 1.2


    return defense



def add_strength_features(df):


    team_defense = calculate_basic_strength(df)


    teams = set(
        df["home_team"]
    ).union(
        set(df["away_team"])
    )


    attack = {}
    defense = {}


    for team in teams:


        games = get_team_games(
            df,
            team
        )


        if len(games) == 0:

            attack[team] = 1.2
            defense[team] = 1.2
            continue



        adjusted_goals = []


        conceded = []


        for _, row in games.iterrows():

            opponent = row["opponent"]


            opponent_def = team_defense.get(
                opponent,
                1.2
            )


            # Strong defence opponent increases value of goals
            adjustment = (
                opponent_def / 1.2
            )


            adjusted_goals.append(
                row["goals_for"] * adjustment
            )

            opponent_attack = team_defense.get(
                opponent,
                1.2
            )

            defense_adjustment = opponent_attack / 1.2

            conceded.append(
                row["goals_against"] * defense_adjustment
            )



        weights = [
            1/(i+1)
            for i in range(len(adjusted_goals))
        ]


        attack[team] = (
            sum(
                g*w
                for g,w in zip(
                    adjusted_goals,
                    weights
                )
            )
            /
            sum(weights)
        )


        defense[team] = (
            sum(
                g*w
                for g,w in zip(
                    conceded,
                    weights
                )
            )
            /
            sum(weights)
        )

        attack[team] = (
                attack[team] * 0.7
                +
                1.2 * 0.3
        )

        defense[team] = (
                defense[team] * 0.7
                +
                1.2 * 0.3
        )



    df["home_attack"] = (
        df["home_team"]
        .map(attack)
    )

    df["away_attack"] = (
        df["away_team"]
        .map(attack)
    )


    df["home_defense"] = (
        df["home_team"]
        .map(defense)
    )

    df["away_defense"] = (
        df["away_team"]
        .map(defense)
    )


    return df
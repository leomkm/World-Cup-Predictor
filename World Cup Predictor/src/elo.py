from collections import defaultdict
import pandas as pd


K = 20


# FIFA ranking based starting strength
# Higher ranked teams start with higher Elo
FIFA_RANKINGS = {
    "Argentina": 1,
    "France": 2,
    "Spain": 3,
    "England": 4,
    "Brazil": 5,
    "Portugal": 6,
    "Netherlands": 7,
    "Belgium": 8,
    "Germany": 9,
    "Croatia": 10,
    "Morocco": 11,
    "Colombia": 12,
    "Japan": 15,
    "Mexico": 17,
    "USA": 18,
    "Switzerland": 20,
    "Senegal": 21,
    "Ecuador": 23,
    "Austria": 25,
    "Norway": 26,
    "Australia": 27,
    "South Africa": 60,
    "Canada": 40,
    "Egypt": 35,
    "Algeria": 37,
    "Ghana": 70,
    "Paraguay": 50,
    "Ivory Coast": 45,
    "Cape Verde": 65,
    "DR Congo": 75,
    "Bosnia and Herzegovina": 75
}


def fifa_to_elo(rank):

    if rank is None:
        return 1500

    # Rank 1 ≈ 2000 Elo
    # Rank 50 ≈ 1600 Elo
    return 2100 - (rank * 8)



def starting_elo(team):

    rank = FIFA_RANKINGS.get(team)

    return fifa_to_elo(rank)



def expected_score(rating_a, rating_b):

    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))



def update_elo(
        home_rating,
        away_rating,
        home_goals,
        away_goals
):

    expected_home = expected_score(
        home_rating,
        away_rating
    )


    if home_goals > away_goals:
        actual_home = 1
        actual_away = 0

    elif home_goals < away_goals:
        actual_home = 0
        actual_away = 1

    else:
        actual_home = 0.5
        actual_away = 0.5


    home_rating += K * (
        actual_home - expected_home
    )

    away_rating += K * (
        actual_away - (1 - expected_home)
    )


    return home_rating, away_rating



def add_elo_features(df):

    df = df.sort_values("date")


    ratings = defaultdict(
        lambda: 1500
    )


    # Apply FIFA starting ratings
    for team in df.home_team.unique():

        ratings[team] = starting_elo(team)



    home_elos = []
    away_elos = []


    for _, row in df.iterrows():

        home = row["home_team"]
        away = row["away_team"]


        home_rating = ratings[home]
        away_rating = ratings[away]


        home_elos.append(home_rating)
        away_elos.append(away_rating)


        new_home, new_away = update_elo(
            home_rating,
            away_rating,
            row["home_score"],
            row["away_score"]
        )


        ratings[home] = new_home
        ratings[away] = new_away


    df["home_elo"] = home_elos
    df["away_elo"] = away_elos


    return df



def get_current_ratings(df):

    df = df.sort_values("date")


    ratings = defaultdict(
        lambda: 1500
    )


    for team in df.home_team.unique():

        ratings[team] = starting_elo(team)



    for _, row in df.iterrows():

        home = row["home_team"]
        away = row["away_team"]


        new_home, new_away = update_elo(
            ratings[home],
            ratings[away],
            row["home_score"],
            row["away_score"]
        )


        ratings[home] = new_home
        ratings[away] = new_away


    return dict(ratings)
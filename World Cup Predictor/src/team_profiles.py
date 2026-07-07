def build_team_profiles(df, ratings):

    profiles = {}

    teams = set(
        df["home_team"]
    ).union(
        set(df["away_team"])
    )


    for team in teams:

        home = df[
            df["home_team"] == team
        ]

        away = df[
            df["away_team"] == team
        ]


        attacks = []

        defenses = []


        if len(home):

            attacks.extend(
                home["home_attack"]
                .tolist()
            )

            defenses.extend(
                home["home_defense"]
                .tolist()
            )


        if len(away):

            attacks.extend(
                away["away_attack"]
                .tolist()
            )

            defenses.extend(
                away["away_defense"]
                .tolist()
            )


        profiles[team] = {

            "elo": ratings.get(
                team,
                1500
            ),

            "form": {

                "attack": sum(attacks) / max(len(attacks), 1),

                "defense": sum(defenses) / max(len(defenses), 1),

                "points": 0,

                "goals_for": 0,

                "goals_against": 0

            }

        }


    return profiles
def build_team_profiles(df, ratings, current_form, current_strength):
    """Builds each team's profile for use in World Cup simulations.

    `ratings`, `current_form`, and `current_strength` should all be computed
    via their respective get_current_*() functions (elo.get_current_ratings,
    form.get_current_form, strength.get_current_strength) so that every team
    enters the simulation with its actual up-to-date strength - not an
    average blended across its entire, possibly decades-long, history, and
    not a hardcoded zero.
    """

    profiles = {}

    teams = set(
        df["home_team"]
    ).union(
        set(df["away_team"])
    )

    # total career matches on record per team - used to decide how much to
    # trust the trained model vs. Elo for this team (see blend.py)
    matches_played = (
        df["home_team"].value_counts()
        .add(df["away_team"].value_counts(), fill_value=0)
    )

    for team in teams:

        form = current_form.get(
            team,
            {"goals_for": 1.2, "goals_against": 1.2, "points": 1.0}
        )

        strength = current_strength.get(
            team,
            {"attack": 1.2, "defense": 1.2}
        )

        profiles[team] = {

            "elo": ratings.get(
                team,
                1500
            ),

            "form": {

                "attack": strength["attack"],

                "defense": strength["defense"],

                "points": form["points"],

                "goals_for": form["goals_for"],

                "goals_against": form["goals_against"],

                "matches_played": int(matches_played.get(team, 0))

            }

        }

    return profiles
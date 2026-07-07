from simulate import simulate_match



def simulate_round(
    matches,
    home_model,
    away_model,
    profiles,
    verbose=True
):

    winners = []


    for match in matches:

        home = match[0]
        away = match[1]

        home_elo = match[2]
        away_elo = match[3]


        result = simulate_match(
            home,
            away,
            home_elo,
            away_elo,
            home_model,
            away_model,
            profiles[home]["form"],
            profiles[away]["form"]
        )


        if verbose:

            print(
                f"{result['home']} "
                f"{result['score']} "
                f"{result['away']} "
                f"→ {result['winner']}"
            )


        winners.append(
            result["winner"]
        )


    return winners
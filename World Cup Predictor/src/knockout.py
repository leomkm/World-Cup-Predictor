from tournament import simulate_round


def simulate_knockout(
    round_of_32,
    ratings,
    profiles,
    home_model,
    away_model,
):
    current_matches = round_of_32

    while len(current_matches) > 1:

        winners = simulate_round(
            current_matches,
            home_model,
            away_model,
            profiles,
            verbose=False
        )

        current_matches = []

        for i in range(0, len(winners), 2):

            home = winners[i]
            away = winners[i + 1]

            current_matches.append(
                (
                    home,
                    away,
                    ratings[home],
                    ratings[away]
                )
            )

    champion = simulate_round(
        current_matches,
        home_model,
        away_model,
        profiles,
        verbose=False
    )[0]

    return champion
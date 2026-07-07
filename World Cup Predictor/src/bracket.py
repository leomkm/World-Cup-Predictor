from simulate import simulate_match


def simulate_full_bracket(
    matches,
    ratings,
    profiles,
    home_model,
    away_model,
    verbose=True,
    return_rounds=False,
    pair_cache=None,
    elo_weight=0.85
):

    current = matches

    round_names = [
        "ROUND OF 32",
        "ROUND OF 16",
        "QUARTER FINALS",
        "SEMI FINALS",
        "FINAL"
    ]

    all_rounds = []

    while len(current) > 0:

        if verbose:
            print(f"\n========== {round_names.pop(0)} ==========")

        winners = []
        round_results = []

        for match in current:
            home = match[0]
            away = match[1]

            result = simulate_match(
                home,
                away,
                ratings[home],
                ratings[away],
                home_model,
                away_model,
                profiles[home]["form"],
                profiles[away]["form"],
                elos=None,
                elo_weight=elo_weight,
                pair_cache=pair_cache
            )

            if verbose:
                print(
                    f"{result['home']} "
                    f"{result['score']} "
                    f"{result['away']} "
                    f"→ {result['winner']}"
                )

            winners.append(result["winner"])

            round_results.append({
                "home": result["home"],
                "away": result["away"],
                "winner": result["winner"],
                "score": result["score"]
            })

        all_rounds.append(round_results)

        if len(winners) == 1:
            break

        current = []

        for i in range(0, len(winners), 2):

            current.append(
                (
                    winners[i],
                    winners[i + 1],
                    ratings[winners[i]],
                    ratings[winners[i + 1]]
                )
            )

    if verbose:
        print("\n========== CHAMPION ==========")
        print(winners[0])

    if return_rounds:
        return all_rounds

    return winners[0]
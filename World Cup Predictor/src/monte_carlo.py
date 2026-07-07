from collections import Counter
from bracket import simulate_full_bracket


def run_simulations(
    matches,
    ratings,
    profiles,
    home_model,
    away_model,
    simulations=1000
):

    champions = Counter()


    for i in range(simulations):

        if i % 50 == 0:
            print(f"Simulation {i}/{simulations}")


        # Copy matches so each simulation starts fresh
        tournament = []

        for match in matches:
            tournament.append(
                (
                    match[0],
                    match[1],
                    match[2],
                    match[3]
                )
            )


        champion = simulate_full_bracket(
            tournament,
            ratings,
            profiles,
            home_model,
            away_model,
            verbose=False
        )


        champions[champion] += 1


    return champions
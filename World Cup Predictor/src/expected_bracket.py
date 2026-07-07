from simulate import simulate_match
from collections import Counter


def simulate_series(
    home,
    away,
    ratings,
    profiles,
    home_model,
    away_model,
    simulations=1000
):

    results = Counter()


    for _ in range(simulations):

        result = simulate_match(
            home,
            away,
            ratings[home],
            ratings[away],
            home_model,
            away_model,
            profiles[home]["form"],
            profiles[away]["form"]
        )


        results[result["winner"]] += 1


    winner = results.most_common(1)[0][0]

    total = sum(results.values())

    probability = (
        results[winner] / total * 100
    )


    loser = (
        away
        if winner == home
        else home
    )


    return {
        "winner": winner,
        "loser": loser,
        "probability": probability
    }



def run_expected_bracket(
    matches,
    ratings,
    profiles,
    home_model,
    away_model,
    simulations=1000
):


    rounds = [
        "ROUND OF 32",
        "ROUND OF 16",
        "QUARTER FINALS",
        "SEMI FINALS",
        "FINAL"
    ]


    current = matches



    for round_name in rounds:

        print(
            f"\n========== {round_name} =========="
        )


        winners = []


        for match in current:


            home = match[0]
            away = match[1]


            result = simulate_series(
                home,
                away,
                ratings,
                profiles,
                home_model,
                away_model,
                simulations
            )


            print(
                f"{result['winner']} def. "
                f"{result['loser']} "
                f"({result['probability']:.1f}%)"
            )


            winners.append(
                result["winner"]
            )


        if len(winners) == 1:

            print(
                "\n========== CHAMPION =========="
            )

            print(
                winners[0]
            )

            return winners[0]

        current = []

        for i in range(0, len(winners), 2):
            home = winners[i]
            away = winners[i + 1]

            current.append(
                (
                    home,
                    away,
                    ratings[home],
                    ratings[away]
                )
            )
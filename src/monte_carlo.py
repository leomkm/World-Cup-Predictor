from collections import Counter
from bracket import simulate_full_bracket
from joblib import Parallel, delayed
import multiprocessing

# We import prediction helpers to build a pairwise cache so simulation hot-loop
# avoids calling model.predict or recomputing Poisson pmfs repeatedly.
from simulate import predict_score, match_probs_from_elos, match_result_probs
from venue import is_neutral_match
from blend import compute_elo_weight

def _simulate_chunk(seed, n, matches, ratings, profiles, home_model, away_model, pair_cache, elo_weight):
    """Run n full-bracket simulations using a private RNG (seed).
    simulate_full_bracket is called with pair_cache so simulate_match uses precomputed values.
    """
    champions = Counter()
    for i in range(n):
        champ = simulate_full_bracket(
            [tuple(m) for m in matches],
            ratings,
            profiles,
            home_model,
            away_model,
            verbose=False,
            pair_cache=pair_cache,
            elo_weight=elo_weight
        )
        champions[champ] += 1
    return champions

def run_simulations(
    matches,
    ratings,
    profiles,
    home_model,
    away_model,
    simulations=10000,
    n_jobs=-1,
    elo_weight=None
):
    """Parallel Monte Carlo with precomputed pairwise expected goals & result probabilities.

    - Builds a pair_cache for all ordered team pairs appearing in `matches`.
    - Runs simulations in parallel using joblib.
    - Passes pair_cache into simulate_full_bracket so per-match model.predict and pmf
      computations are avoided inside the hot loop.

    `elo_weight`: if None (default), each pair gets its own adaptively-computed
    weight based on how much real match history each team has (see blend.py).
    Pass a fixed number here (e.g. 0.85) to force the same weight for every
    matchup, like the old behavior.
    """
    if simulations <= 0:
        return Counter()

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    # Build pairwise cache for teams appearing in the initial matches
    teams = set()
    for m in matches:
        teams.add(m[0]); teams.add(m[1])

    pair_cache = {}
    for a in teams:
        for b in teams:
            if a == b:
                continue
            try:
                neutral = is_neutral_match(a, b)

                # expected goals from the trained models
                exp_h, exp_a = predict_score(
                    home_model,
                    away_model,
                    ratings[a],
                    ratings[b],
                    profiles[a]["form"],
                    profiles[b]["form"],
                    neutral=neutral
                )

                # Elo-derived result probs and Poisson-derived result probs
                p_e_home, p_e_draw, p_e_away = match_probs_from_elos(ratings[a], ratings[b])
                p_p_home, p_p_draw, p_p_away = match_result_probs(exp_h, exp_a)

                pair_elo_weight = elo_weight if elo_weight is not None else compute_elo_weight(
                    profiles[a]["form"].get("matches_played", 0),
                    profiles[b]["form"].get("matches_played", 0)
                )

                p_home = pair_elo_weight * p_e_home + (1 - pair_elo_weight) * p_p_home
                p_draw = pair_elo_weight * p_e_draw + (1 - pair_elo_weight) * p_p_draw
                p_away = pair_elo_weight * p_e_away + (1 - pair_elo_weight) * p_p_away

                s = p_home + p_draw + p_away
                if s <= 0:
                    p_home = p_draw = p_away = 1.0 / 3.0
                else:
                    p_home /= s; p_draw /= s; p_away /= s

                pair_cache[(a, b)] = {
                    "exp_home": exp_h,
                    "exp_away": exp_a,
                    "p_home": p_home,
                    "p_draw": p_draw,
                    "p_away": p_away,
                    "h_elo": ratings[a],
                    "a_elo": ratings[b]
                }
            except Exception:
                # If anything fails, skip caching this pair; simulate_match will fall back
                continue

    # Split the simulations into chunks for parallel execution
    base = simulations // n_jobs
    counts = [base] * n_jobs
    rem = simulations - base * n_jobs
    for i in range(rem):
        counts[i] += 1

    seeds = [int(100000 + i) for i in range(len(counts))]

    results = Parallel(n_jobs=n_jobs)(
        delayed(_simulate_chunk)(seeds[i], counts[i], matches, ratings, profiles, home_model, away_model, pair_cache, elo_weight)
        for i in range(len(counts))
    )

    champions = Counter()
    for r in results:
        champions.update(r)

    return champions
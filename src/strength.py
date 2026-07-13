from collections import defaultdict, deque


LAST_N = 10
BASELINE = 1.2   # league-average goals/game, used as prior for teams with no history
BLEND = 0.7      # weight given to a team's own opponent-adjusted record vs. the baseline


def _recency_weighted_avg(records, key):
    """Weighted average of `key` across `records`, with the most recent match
    (last item appended) given the most weight. `records` is oldest -> newest.
    Returns BASELINE if there's no history yet.
    """
    if not records:
        return BASELINE

    recent_first = list(reversed(records))
    weights = [1 / (i + 1) for i in range(len(recent_first))]

    return (
        sum(r[key] * w for r, w in zip(recent_first, weights))
        / sum(weights)
    )


def _process(df):
    """Shared sequential pass. Returns per-row feature lists AND the final
    per-team raw/adjusted histories (so callers can either build training
    columns or read off each team's final, as-of-now rating).

    Two histories are kept per team:
      - raw_history: actual goals for/against, unadjusted. Bounded (0-ish to
        single digits), used ONLY to measure an opponent's real strength.
      - adj_history: opponent-adjusted goals, blended towards BASELINE. This
        is what gets exposed as the attack/defense feature.

    Using RAW (bounded) opponent strength as the adjustment multiplier -
    rather than the opponent's own already-adjusted rating - is important:
    feeding an adjusted rating back into another team's adjustment compounds
    across thousands of sequential matches and blows up numerically. Raw
    goal counts stay bounded, so the adjustment multiplier stays bounded too.
    """

    df = df.sort_values("date").reset_index(drop=True)

    raw_history = defaultdict(lambda: deque(maxlen=LAST_N))
    adj_history = defaultdict(lambda: deque(maxlen=LAST_N))

    home_attack_col, away_attack_col = [], []
    home_defense_col, away_defense_col = [], []

    for _, row in df.iterrows():

        home = row["home_team"]
        away = row["away_team"]

        home_raw = list(raw_history[home])
        away_raw = list(raw_history[away])

        # opponent's real (bounded) recent strength, known before this match
        home_raw_attack = _recency_weighted_avg(home_raw, "gf")
        home_raw_defense = _recency_weighted_avg(home_raw, "ga")
        away_raw_attack = _recency_weighted_avg(away_raw, "gf")
        away_raw_defense = _recency_weighted_avg(away_raw, "ga")

        # exposed attack/defense ratings, blended towards the baseline
        raw_home_attack_adj = _recency_weighted_avg(list(adj_history[home]), "adj_gf")
        raw_home_defense_adj = _recency_weighted_avg(list(adj_history[home]), "adj_ga")
        raw_away_attack_adj = _recency_weighted_avg(list(adj_history[away]), "adj_gf")
        raw_away_defense_adj = _recency_weighted_avg(list(adj_history[away]), "adj_ga")

        home_attack = raw_home_attack_adj * BLEND + BASELINE * (1 - BLEND)
        home_defense = raw_home_defense_adj * BLEND + BASELINE * (1 - BLEND)
        away_attack = raw_away_attack_adj * BLEND + BASELINE * (1 - BLEND)
        away_defense = raw_away_defense_adj * BLEND + BASELINE * (1 - BLEND)

        home_attack_col.append(home_attack)
        home_defense_col.append(home_defense)
        away_attack_col.append(away_attack)
        away_defense_col.append(away_defense)

        # update raw histories with this match's actual goals
        raw_history[home].append({"gf": row["home_score"], "ga": row["away_score"]})
        raw_history[away].append({"gf": row["away_score"], "ga": row["home_score"]})

        # update adjusted histories using the OPPONENT'S RAW strength
        # (bounded) as the multiplier - never the adjusted rating itself
        adj_history[home].append({
            "adj_gf": row["home_score"] * (away_raw_defense / BASELINE),
            "adj_ga": row["away_score"] * (away_raw_attack / BASELINE),
        })
        adj_history[away].append({
            "adj_gf": row["away_score"] * (home_raw_defense / BASELINE),
            "adj_ga": row["home_score"] * (home_raw_attack / BASELINE),
        })

    return df, home_attack_col, away_attack_col, home_defense_col, away_defense_col, adj_history


def add_strength_features(df):
    """Adds opponent-adjusted, recency-weighted attack/defense ratings.

    IMPORTANT: this processes matches in chronological order and, for every
    match, only uses that team's PRIOR matches to compute the features for
    that row - so a match played in 2015 is never rated using a team's form
    from 2024. (The old version rated every historical row using each team's
    most recent 10 games in the WHOLE dataset - that was look-ahead leakage.)
    """

    df, home_attack_col, away_attack_col, home_defense_col, away_defense_col, _ = _process(df)

    df["home_attack"] = home_attack_col
    df["away_attack"] = away_attack_col
    df["home_defense"] = home_defense_col
    df["away_defense"] = away_defense_col

    return df


def get_current_strength(df):
    """Returns each team's attack/defense rating as of RIGHT NOW (i.e. after
    processing every match in the dataset). Mirrors elo.get_current_ratings.

    Use this - not a historical average over the columns added by
    add_strength_features - to seed World Cup simulations, since we want
    each team's most up-to-date strength, not a value blended across
    their entire, possibly decades-long, history.
    """

    df, *_, adj_history = _process(df)

    current = {}
    teams = set(df["home_team"]).union(set(df["away_team"]))

    for team in teams:
        records = list(adj_history[team])
        attack_raw = _recency_weighted_avg(records, "adj_gf")
        defense_raw = _recency_weighted_avg(records, "adj_ga")
        current[team] = {
            "attack": attack_raw * BLEND + BASELINE * (1 - BLEND),
            "defense": defense_raw * BLEND + BASELINE * (1 - BLEND),
        }

    return current

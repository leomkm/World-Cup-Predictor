def create_features(df, time_decay_days=365*2, competition_weights=None):
    """Create features X and targets y_home, y_away from matches DataFrame.

    Also compute a sample weight per match combining time decay and competition importance.

    Returns: X, y_home, y_away, sample_weights
    """
    import pandas as pd
    import numpy as np

    # Base feature columns
    X = df[
        [
            "home_elo",
            "away_elo",

            "home_recent_goals_for",
            "home_recent_goals_against",
            "home_recent_points",

            "away_recent_goals_for",
            "away_recent_goals_against",
            "away_recent_points",

            "home_attack",
            "away_attack",
            "home_defense",
            "away_defense"
        ]
    ].copy()

    y_home = df["home_score"].copy()
    y_away = df["away_score"].copy()

    # --------------------------
    # SAMPLE WEIGHTS
    # --------------------------
    # Time decay: more recent matches have higher weight.
    # We compute days since match relative to the latest match in the dataset.
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        last = dates.max()
        days = (last - dates).dt.days.fillna(time_decay_days)
        # exponential decay
        tau = time_decay_days
        time_weight = np.exp(-days / tau)
    else:
        time_weight = np.ones(len(df))

    # Competition weighting: allow caller to pass mapping {competition: weight}
    if competition_weights is None:
        competition_weights = {
            # common names - adjust to your dataset's competition naming
            "FIFA World Cup": 1.0,
            "World Cup": 1.0,
            "World Cup Qualifier": 0.8,
            "Qualifier": 0.8,
            "UEFA Nations League": 0.7,
            "Friendly": 0.4,
            "Club Friendly": 0.3,
        }

    if "competition" in df.columns:
        comp = df["competition"].fillna("")
        comp_weight = comp.map(lambda c: competition_weights.get(c, 0.7))
    else:
        comp_weight = np.ones(len(df))

    # Final sample weight combines both factors
    sample_weight = time_weight * comp_weight

    # Normalize weights to mean=1 to avoid scaling target problems
    if len(sample_weight) > 0:
        sample_weight = sample_weight / (sample_weight.mean())

    return X, y_home, y_away, sample_weight

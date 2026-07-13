# sample_weights.py — per-match training weights.
#
# The old code trained on all ~49,500 matches back to 1872 with every match
# counted equally. That's a problem two ways:
#   1. Recency: national squads, tactics, and even which countries exist has
#      changed enormously since 1872. A match from 1890 says very little
#      about how a team will play in 2026.
#   2. Competition importance: a World Cup match is a much better signal for
#      predicting World Cup form than a mid-season friendly with a weakened
#      squad and nothing on the line.
#
# Rather than hard-cutting the data (e.g. "only use matches since 2015" or
# "drop all friendlies"), this uses soft weights so old/low-stakes matches
# still contribute a little, but recent/high-stakes matches dominate.

import pandas as pd


RECENCY_HALF_LIFE_YEARS = 10  # a match this many years old counts for half a fresh one


def _competition_weight(tournament):
    name = str(tournament).lower()

    if "world cup" in name and "qualif" not in name:
        return 2.0  # the actual competition we're trying to predict

    major_finals_keywords = [
        "euro", "copa américa", "copa america", "cup of nations",
        "asian cup", "gold cup", "nations league", "confederations cup",
    ]
    if "qualif" not in name and any(k in name for k in major_finals_keywords):
        return 1.4  # major continental championship finals

    if "world cup" in name and "qualif" in name:
        return 1.3  # World Cup qualifiers - directly relevant, competitive

    if "qualif" in name:
        return 1.1  # other qualifiers - competitive, lower stakes

    if name == "friendly":
        return 0.5  # weakest signal: no real stakes, often weakened squads

    return 1.0  # everything else (regional cups, games, etc.)


def compute_sample_weights(df):
    """Returns a per-row training weight combining recency and competition
    importance. Pass this straight into XGBRegressor.fit(sample_weight=...).
    """

    dates = pd.to_datetime(df["date"])
    latest = dates.max()
    years_ago = (latest - dates).dt.days / 365.25

    recency_weight = 0.5 ** (years_ago / RECENCY_HALF_LIFE_YEARS)
    competition_weight = df["tournament"].map(_competition_weight)

    return (recency_weight * competition_weight).to_numpy()
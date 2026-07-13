# blend.py — decides how much to trust Elo vs. the trained model for a
# given matchup's win/draw/loss probability.
#
# The old code used a single fixed constant (elo_weight = 0.85) for every
# match in the tournament, whether it was Brazil vs Argentina (thousands of
# recorded matches each, tons of signal for the model to learn from) or two
# sparsely-documented national teams with only a handful of matches on
# record. That's backwards: the trained model needs a reasonable amount of
# history per team to say anything useful, while Elo's simple update rule
# and FIFA-ranking-based starting point degrade far more gracefully with
# little data.
#
# So instead of one fixed number, the weight now depends on how much real
# match history the two teams involved actually have:
#   - Well-documented teams -> lean more on the trained model.
#   - Sparse-data teams     -> lean more on Elo.

MIN_ELO_WEIGHT = 0.55          # weight on Elo even for the best-documented teams
MAX_ELO_WEIGHT = 0.90          # weight on Elo for teams with ~no history
FULL_CONFIDENCE_MATCHES = 150  # matches played at which we trust the model fully


def compute_elo_weight(home_matches_played, away_matches_played):
    """Returns the weight to give Elo (vs. the trained model) when blending
    win/draw/loss probabilities for a matchup.

    Uses whichever of the two teams has LESS history, since the matchup as
    a whole is only as well-modeled as its least-documented participant.
    """
    n = min(home_matches_played or 0, away_matches_played or 0)
    confidence = min(max(n / FULL_CONFIDENCE_MATCHES, 0.0), 1.0)

    return MAX_ELO_WEIGHT - confidence * (MAX_ELO_WEIGHT - MIN_ELO_WEIGHT)
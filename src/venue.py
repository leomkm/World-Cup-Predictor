# venue.py — figures out whether a given fixture is played on neutral ground.
#
# Most 2026 World Cup matches are neutral for both teams. The exception is
# the three host nations, who play the bulk of their group-stage (and often
# early knockout) matches at home, with a real home-crowd advantage.
#
# This is a simplification - it doesn't track the actual match venue per
# fixture - but it's a large step up from treating every match as if it had
# the same home-field dynamics as a 2010 European qualifier.

HOST_NATIONS = {"United States", "Canada", "Mexico"}


def is_neutral_match(home_team, away_team):
    """True unless the designated home team is a 2026 host nation.

    Note "home_team" here just means "listed first in the fixture" - for
    knockout-stage pairings built from earlier winners, that's arbitrary.
    So this will occasionally call a game neutral when a host nation is
    actually playing at home but happened to be listed as the "away" side,
    or vice versa. Good enough as a first pass; a full fix would carry a
    real venue/host flag through the whole bracket structure.
    """
    return home_team not in HOST_NATIONS
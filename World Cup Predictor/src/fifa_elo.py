# FIFA-style starting strength ratings

FIFA_RATINGS = {

    "Argentina": 1885,
    "France": 1855,
    "Spain": 1845,
    "England": 1810,
    "Brazil": 1785,
    "Portugal": 1760,
    "Netherlands": 1740,
    "Belgium": 1725,
    "Germany": 1710,
    "Croatia": 1690,
    "Morocco": 1680,
    "Japan": 1660,
    "United States": 1650,
    "Mexico": 1640,
    "Colombia": 1635,
    "Switzerland": 1625,
    "Senegal": 1615,
    "Austria": 1600,
    "Ecuador": 1595,
    "Canada": 1580,
    "Australia": 1575,
    "Norway": 1570,
    "Sweden": 1565,
    "Egypt": 1555,
    "South Africa": 1545,
    "Ghana": 1540,
    "Algeria": 1535,
    "Paraguay": 1530,
    "Ivory Coast": 1525,
    "DR Congo": 1520,
    "Cape Verde": 1515,
    "Bosnia and Herzegovina": 1510
}


def fifa_to_elo(points):

    # Convert FIFA strength into Elo scale

    return 1000 + (points - 1400) * 2


def get_fifa_elos():

    ratings = {}

    for team, points in FIFA_RATINGS.items():

        ratings[team] = fifa_to_elo(points)


    return ratings
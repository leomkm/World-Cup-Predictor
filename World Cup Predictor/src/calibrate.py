from sklearn.isotonic import IsotonicRegression
import joblib
import numpy as np

from simulate import get_blended_probs, load_elos


def fit_calibrators(df, home_model, away_model, ratings, profiles, elos=None, elo_weight=0.85, val_fraction=0.2):
    """Fit isotonic calibrators for home/draw/away probabilities.

    - df: pandas DataFrame of historical matches with columns ['date','home_team','away_team','home_score','away_score']
    - returns a dict of IsotonicRegression objects for 'home','draw','away'
    """
    if elos is None:
        elos = load_elos()

    # sort by date and take last val_fraction as calibration set
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    if n == 0:
        return None
    val_start = int(n * (1 - val_fraction))
    calib_df = df.iloc[val_start:]

    # collect predicted probs and true outcomes
    ph = []
    pdraw = []
    pa = []
    y_home = []
    y_draw = []
    y_away = []

    for _, row in calib_df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        home_elo = ratings.get(home)
        away_elo = ratings.get(away)
        home_form = profiles.get(home, {}).get("form", {})
        away_form = profiles.get(away, {}).get("form", {})

        try:
            probs = get_blended_probs(
                home,
                away,
                home_elo,
                away_elo,
                home_model,
                away_model,
                home_form,
                away_form,
                elos=elos,
                elo_weight=elo_weight
            )
        except Exception:
            continue

        ph.append(probs["p_home"])
        pdraw.append(probs["p_draw"])
        pa.append(probs["p_away"])

        # true outcome
        if row["home_score"] > row["away_score"]:
            y_home.append(1); y_draw.append(0); y_away.append(0)
        elif row["home_score"] == row["away_score"]:
            y_home.append(0); y_draw.append(1); y_away.append(0)
        else:
            y_home.append(0); y_draw.append(0); y_away.append(1)

    # ensure arrays
    ph = np.array(ph)
    pdraw = np.array(pdraw)
    pa = np.array(pa)
    y_home = np.array(y_home)
    y_draw = np.array(y_draw)
    y_away = np.array(y_away)

    # If not enough data, return None
    if len(ph) < 50:
        # not enough calibration examples
        return None

    # Fit isotonic regressions (one-vs-rest)
    iso_home = IsotonicRegression(out_of_bounds="clip").fit(ph, y_home)
    iso_draw = IsotonicRegression(out_of_bounds="clip").fit(pdraw, y_draw)
    iso_away = IsotonicRegression(out_of_bounds="clip").fit(pa, y_away)

    calibrator = {"home": iso_home, "draw": iso_draw, "away": iso_away}
    return calibrator


def save_calibrator(calibrator, path):
    try:
        joblib.dump(calibrator, path)
        return True
    except Exception:
        return False


def load_calibrator(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

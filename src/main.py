from sklearn.model_selection import train_test_split
import pandas as pd

from load_data import load_matches
from elo import add_elo_features, get_current_ratings
from form import add_form_features, get_current_form
from strength import add_strength_features, get_current_strength
from features import create_features
from train import train_model
from evaluate import evaluate
from team_profiles import build_team_profiles
from monte_carlo import run_simulations


# ==========================
# LOAD DATA
# ==========================

df = load_matches("../data/results.csv")

df = df.dropna(
    subset=[
        "home_score",
        "away_score",
        "home_team",
        "away_team",
        "date"
    ]
)


# ==========================
# CREATE FEATURES
# ==========================

df = add_elo_features(df)
df = add_form_features(df)
df = add_strength_features(df)

ratings = get_current_ratings(df)
current_form = get_current_form(df)
current_strength = get_current_strength(df)


# ==========================
# TRAIN MODELS
# ==========================

# original feature creation
X, y_home, y_away = create_features(df)

# original splitting and model training
X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(
    X,
    y_home,
    y_away,
    test_size=0.2,
    random_state=42
)

home_model = train_model(
    X_train,
    y_home_train
)

away_model = train_model(
    X_train,
    y_away_train
)
print("\nHome goals model:")
evaluate(
    home_model,
    X_test,
    y_home_test
)


print("\nAway goals model:")
evaluate(
    away_model,
    X_test,
    y_away_test
)


# ==========================
# BUILD TEAM PROFILES
# ==========================

profiles = build_team_profiles(
    df,
    ratings,
    current_form,
    current_strength
)


# SAVE MODELS HERE
import joblib
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


joblib.dump(
    home_model,
    os.path.join(
        MODEL_DIR,
        "home_model.pkl"
    )
)

joblib.dump(
    away_model,
    os.path.join(
        MODEL_DIR,
        "away_model.pkl"
    )
)

joblib.dump(
    ratings,
    os.path.join(
        MODEL_DIR,
        "ratings.pkl"
    )
)

joblib.dump(
    profiles,
    os.path.join(
        MODEL_DIR,
        "profiles.pkl"
    )
)


print("Saved models successfully")

# ==========================
# WORLD CUP TEAMS
# ==========================

round_of_32 = [

    ("Germany", "Paraguay"),
    ("France", "Sweden"),

    ("South Africa", "Canada"),
    ("Netherlands", "Morocco"),

    ("Portugal", "Croatia"),
    ("Spain", "Austria"),

    ("United States", "Bosnia and Herzegovina"),
    ("Belgium", "Senegal"),

    ("Brazil", "Japan"),
    ("Ivory Coast", "Norway"),

    ("Mexico", "Ecuador"),
    ("England", "DR Congo"),

    ("Argentina", "Cape Verde"),
    ("Australia", "Egypt"),

    ("Switzerland", "Algeria"),
    ("Colombia", "Ghana"),
]

matches = []

for home, away in round_of_32:
    matches.append(
        (
            home,
            away,
            ratings[home],
            ratings[away]
        )
    )

results = run_simulations(
    matches,
    ratings,
    profiles,
    home_model,
    away_model,
    simulations=10000
)


# ==========================
# RESULTS
# ==========================

print("\nWorld Cup probabilities\n")


from math import sqrt

def wilson_interval(k, n, z=1.96):
    """Wilson score interval for k successes in n trials (returns lower, upper)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z*z / n
    center = p + z*z / (2*n)
    half = z * sqrt(max(0.0, p*(1-p)/n + z*z/(4*n*n)))
    lower = (center - half) / denom
    upper = (center + half) / denom
    return max(0.0, lower), min(1.0, upper)

# Replace your original print loop with this
alpha = 1.0  # Laplace smoothing strength (1.0 = add-one). Lower it if you have many sims.
teams = list(results.keys())
K = len(teams)
total = sum(results.values())

print("\nWorld Cup probabilities\n")
print(f"{'Rank':>4}  {'Team':<25} {'Prob':>8}  {'95% CI':<21}  {'Wins':>6}")
print("-" * 70)

sorted_teams = sorted(results.items(), key=lambda kv: kv[1], reverse=True)
for rank, (team, wins) in enumerate(sorted_teams, start=1):
    # Smoothed proportion
    wins_s = wins + alpha
    n_s = total + alpha * K
    prob = wins_s / n_s
    lower, upper = wilson_interval(wins_s, n_s)
    print(f"{rank:>4}  {team:<25} {prob*100:7.2f}%  ({lower*100:6.2f}% - {upper*100:6.2f}%)  {wins:6d}")
from expected_bracket import run_expected_bracket


print("\n\nEXPECTED TOURNAMENT BRACKET")


run_expected_bracket(
    matches,
    ratings,
    profiles,
    home_model,
    away_model,
    simulations=1000
)
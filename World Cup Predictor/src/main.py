from sklearn.model_selection import train_test_split
import pandas as pd
import json

from load_data import load_matches
from elo import add_elo_features, get_current_ratings
from form import add_form_features
from strength import add_strength_features
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


# ==========================
# TRAIN MODELS
# ==========================

X, y_home, y_away = create_features(df)


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
    ratings
)
for team in [
    "Colombia",
    "Brazil",
    "Argentina",
    "France",
    "England"
]:
    print("\n", team)
    print(profiles[team])

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
from simulate import simulate_match

for a,b in [
    ("Argentina","Brazil"),
    ("France","Brazil"),
    ("Argentina","France")
]:

    wins = {
        a:0,
        b:0
    }

    for i in range(1000):

        result = simulate_match(
            a,
            b,
            ratings[a],
            ratings[b],
            home_model,
            away_model,
            profiles[a]["form"],
            profiles[b]["form"]
        )

        wins[result["winner"]] += 1

    print(a,b,wins)
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
    simulations=1000
)


# ==========================
# RESULTS
# ==========================

print("\nWorld Cup probabilities\n")


dashboard_output = []

total_sims = sum(results.values()) if len(results) > 0 else 1

for team, wins in results.most_common():

    probability = wins / total_sims

    # print as percentage for the console
    print(
        f"{team}: {probability * 100:.1f}%"
    )

    # collect structured output for dashboard (both fraction and percentage)
    dashboard_output.append({
        "team": team,
        "wins": int(wins),
        "prob": probability,
        "prob_percent": round(probability * 100, 2)
    })

# write dashboard JSON file
DASHBOARD_PATH = os.path.join(BASE_DIR, "dashboard_results.json")
try:
    with open(DASHBOARD_PATH, "w", encoding="utf-8") as fh:
        json.dump({"results": dashboard_output}, fh, indent=2)
    print(f"Wrote dashboard results to {DASHBOARD_PATH}")
except Exception as e:
    print("Failed to write dashboard JSON:", e)

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

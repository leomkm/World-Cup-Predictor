import streamlit as st
import pandas as pd
import joblib
import os
from simulate import simulate_match


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


home_model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "home_model.pkl"
    )
)

away_model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "away_model.pkl"
    )
)

ratings = joblib.load(
    os.path.join(
        MODEL_DIR,
        "ratings.pkl"
    )
)

profiles = joblib.load(
    os.path.join(
        MODEL_DIR,
        "profiles.pkl"
    )
)
print("DASHBOARD PROFILE")
print(profiles["Egypt"])
print(profiles["Australia"])

# =========================
# DASHBOARD
# =========================

st.title("⚽ World Cup Predictor")

teams = sorted(
    ratings.keys()
)


home_team = st.selectbox(
    "Home Team",
    teams
)

away_team = st.selectbox(
    "Away Team",
    teams,
    index=1
)


if home_team != away_team:


    home_profile = profiles[home_team]
    away_profile = profiles[away_team]


    st.subheader("Team Strength")


    col1, col2 = st.columns(2)


    with col1:

        st.write(
            f"### {home_team}"
        )

        st.metric(
            "Elo",
            round(home_profile["elo"])
        )

        st.metric(
            "Attack",
            round(
                home_profile["form"]["attack"],
                2
            )
        )

        st.metric(
            "Defense",
            round(
                home_profile["form"]["defense"],
                2
            )
        )


    with col2:

        st.write(
            f"### {away_team}"
        )

        st.metric(
            "Elo",
            round(away_profile["elo"])
        )

        st.metric(
            "Attack",
            round(
                away_profile["form"]["attack"],
                2
            )
        )

        st.metric(
            "Defense",
            round(
                away_profile["form"]["defense"],
                2
            )
        )


    st.subheader("Prediction")


    if st.button("Predict Match"):


        result = simulate_match(
            home_team,
            away_team,
            ratings[home_team],
            ratings[away_team],
            home_model,
            away_model,
            home_profile["form"],
            away_profile["form"]
        )


        st.success(
            f"{result['home']} {result['score']} {result['away']}"
        )


        st.write(
            "Winner:",
            result["winner"]
        )
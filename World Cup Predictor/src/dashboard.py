import os
import streamlit as st
import joblib
from collections import Counter
from simulate import simulate_match

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

home_model = joblib.load(os.path.join(MODEL_DIR, "home_model.pkl"))
away_model = joblib.load(os.path.join(MODEL_DIR, "away_model.pkl"))
ratings = joblib.load(os.path.join(MODEL_DIR, "ratings.pkl"))
profiles = joblib.load(os.path.join(MODEL_DIR, "profiles.pkl"))

st.title("⚽ World Cup Predictor")

teams = sorted(ratings.keys())

home_team = st.selectbox("Home Team", teams)
away_team = st.selectbox("Away Team", teams, index=1)

if home_team != away_team:
    home_profile = profiles[home_team]
    away_profile = profiles[away_team]

    st.subheader("Team Strength")
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"### {home_team}")
        st.metric("Elo", round(home_profile["elo"]))
        st.metric("Attack", round(home_profile["form"]["attack"], 2))
        st.metric("Defense", round(home_profile["form"]["defense"], 2))

    with col2:
        st.write(f"### {away_team}")
        st.metric("Elo", round(away_profile["elo"]))
        st.metric("Attack", round(away_profile["form"]["attack"], 2))
        st.metric("Defense", round(away_profile["form"]["defense"], 2))

    st.subheader("Prediction")

    # choose how many sims to use for the probability estimate
    simulations = st.slider(
        "Simulations for probability estimate", min_value=100, max_value=20000, value=1000, step=100
    )

    if st.button("Predict Match"):
        # run Monte Carlo by sampling simulate_match many times and counting winners
        counts = Counter()
        progress_bar = st.progress(0)
        status_text = st.empty()

        # batch in small blocks so UI updates aren't too frequent but progress is smooth
        batch = max(1, simulations // 100)
        for i in range(simulations):
            res = simulate_match(
                home_team,
                away_team,
                ratings[home_team],
                ratings[away_team],
                home_model,
                away_model,
                home_profile["form"],
                away_profile["form"]
            )
            counts[res["winner"]] += 1

            if (i + 1) % batch == 0 or i == simulations - 1:
                progress = int((i + 1) / simulations * 100)
                progress_bar.progress(min(progress, 100))
                status_text.text(f"Simulations: {i+1}/{simulations}")

        progress_bar.empty()
        status_text.empty()

        total = sum(counts.values()) if counts else 1
        home_pct = counts.get(home_team, 0) / total * 100
        away_pct = counts.get(away_team, 0) / total * 100

        # show both teams' win probabilities and the predicted winner with percent
        st.markdown("### Win probabilities")
        st.write(f"{home_team}: {home_pct:.1f}%")
        st.write(f"{away_team}: {away_pct:.1f}%")

        if home_pct > away_pct:
            winner, winner_pct = home_team, home_pct
        elif away_pct > home_pct:
            winner, winner_pct = away_team, away_pct
        else:
            winner, winner_pct = "Tie (equal)", home_pct

        st.success(f"Predicted winner: {winner} ({winner_pct:.1f}%)")
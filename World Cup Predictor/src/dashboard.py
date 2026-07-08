import os
import streamlit as st
import joblib
import pandas as pd
from collections import Counter
from simulate import simulate_match
from head_to_head import load_head_to_head_stats, calculate_h2h_summary, format_h2h_match_display
from load_data import load_matches

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

home_model = joblib.load(os.path.join(MODEL_DIR, "home_model.pkl"))
away_model = joblib.load(os.path.join(MODEL_DIR, "away_model.pkl"))
ratings = joblib.load(os.path.join(MODEL_DIR, "ratings.pkl"))
profiles = joblib.load(os.path.join(MODEL_DIR, "profiles.pkl"))

# Load match data for head-to-head stats
df = load_matches(os.path.join(DATA_DIR, "results.csv"))

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

    # Head-to-Head Stats Section
    st.subheader("📊 Head-to-Head History")
    
    years_filter = st.slider("Years of history to show", min_value=1, max_value=20, value=5)
    
    h2h_matches = load_head_to_head_stats(df, home_team, away_team, years=years_filter)
    h2h_summary = calculate_h2h_summary(h2h_matches, home_team, away_team)
    
    if h2h_summary["total_matches"] > 0:
        # Display summary stats
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                f"{home_team} Wins",
                h2h_summary["team1_wins"],
                f"{h2h_summary['team1_win_pct']:.1f}%"
            )
        
        with col2:
            st.metric(
                f"{away_team} Wins",
                h2h_summary["team2_wins"],
                f"{h2h_summary['team2_win_pct']:.1f}%"
            )
        
        with col3:
            st.metric(
                "Draws",
                h2h_summary["team1_draws"],
                f"{h2h_summary['draw_pct']:.1f}%"
            )
        
        with col4:
            st.metric(
                f"{home_team} GF/GA",
                f"{h2h_summary['team1_goals_for']}/{h2h_summary['team1_goals_against']}"
            )
        
        with col5:
            st.metric(
                f"{away_team} GF/GA",
                f"{h2h_summary['team2_goals_for']}/{h2h_summary['team2_goals_against']}"
            )
        
        # Display recent matches
        st.write(f"#### Recent Matches ({len(h2h_matches)} total)")
        
        match_data = []
        for _, match in h2h_matches.iterrows():
            formatted = format_h2h_match_display(match, home_team)
            match_data.append({
                "Date": pd.to_datetime(formatted["date"]).strftime("%Y-%m-%d"),
                "Home": formatted["home"],
                "Away": formatted["away"],
                "Score": formatted["score"],
                "Result": formatted["result"]
            })
        
        if match_data:
            matches_df = pd.DataFrame(match_data)
            st.dataframe(matches_df, use_container_width=True)
    else:
        st.info(f"No previous matches found between {home_team} and {away_team} in the past {years_filter} years.")

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

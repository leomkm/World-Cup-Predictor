import pandas as pd
from datetime import datetime, timedelta


def load_head_to_head_stats(df, team1, team2, years=5):
    """
    Load historical head-to-head stats between two teams in the past N years.
    
    Args:
        df: Match results dataframe
        team1: First team name
        team2: Second team name
        years: Number of years to look back (default 5)
    
    Returns:
        DataFrame of matches between the two teams
    """
    
    # Calculate cutoff date (N years ago from today)
    cutoff_date = datetime.now() - timedelta(days=365 * years)
    
    # Convert date column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Filter for matches in the timeframe
    df_filtered = df[df['date'] >= cutoff_date].copy()
    
    # Find all matches between the two teams
    h2h = df_filtered[
        ((df_filtered['home_team'] == team1) & (df_filtered['away_team'] == team2)) |
        ((df_filtered['home_team'] == team2) & (df_filtered['away_team'] == team1))
    ].copy()
    
    # Sort by date descending (most recent first)
    h2h = h2h.sort_values('date', ascending=False)
    
    return h2h


def calculate_h2h_summary(h2h, team1, team2):
    """
    Calculate summary stats for head-to-head matches.
    
    Returns a dictionary with:
    - Total matches
    - Team1 wins/draws/losses
    - Team2 wins/draws/losses
    - Goals for/against each team
    - Win percentages
    """
    
    if len(h2h) == 0:
        return {
            "total_matches": 0,
            "team1_wins": 0,
            "team1_draws": 0,
            "team1_losses": 0,
            "team2_wins": 0,
            "team2_draws": 0,
            "team2_losses": 0,
            "team1_goals_for": 0,
            "team1_goals_against": 0,
            "team2_goals_for": 0,
            "team2_goals_against": 0,
            "team1_win_pct": 0,
            "team2_win_pct": 0,
            "draw_pct": 0,
        }
    
    total = len(h2h)
    
    team1_wins = 0
    team1_draws = 0
    team1_losses = 0
    team1_goals_for = 0
    team1_goals_against = 0
    
    team2_wins = 0
    team2_draws = 0
    team2_losses = 0
    team2_goals_for = 0
    team2_goals_against = 0
    
    for _, match in h2h.iterrows():
        if match['home_team'] == team1:
            # Team1 is home
            team1_goals_for += match['home_score']
            team1_goals_against += match['away_score']
            
            if match['home_score'] > match['away_score']:
                team1_wins += 1
                team2_losses += 1
            elif match['home_score'] < match['away_score']:
                team1_losses += 1
                team2_wins += 1
            else:
                team1_draws += 1
                team2_draws += 1
        else:
            # Team1 is away
            team1_goals_for += match['away_score']
            team1_goals_against += match['home_score']
            
            if match['away_score'] > match['home_score']:
                team1_wins += 1
                team2_losses += 1
            elif match['away_score'] < match['home_score']:
                team1_losses += 1
                team2_wins += 1
            else:
                team1_draws += 1
                team2_draws += 1
        
        team2_goals_for += team1_goals_against
        team2_goals_against += team1_goals_for
    
    # Recalculate team2 goals properly
    team2_goals_for = sum(
        match['away_score'] if match['home_team'] == team1 else match['home_score']
        for _, match in h2h.iterrows()
    )
    team2_goals_against = sum(
        match['home_score'] if match['home_team'] == team1 else match['away_score']
        for _, match in h2h.iterrows()
    )
    
    return {
        "total_matches": total,
        "team1_wins": team1_wins,
        "team1_draws": team1_draws,
        "team1_losses": team1_losses,
        "team2_wins": team2_wins,
        "team2_draws": team2_draws,
        "team2_losses": team2_losses,
        "team1_goals_for": team1_goals_for,
        "team1_goals_against": team1_goals_against,
        "team2_goals_for": team2_goals_for,
        "team2_goals_against": team2_goals_against,
        "team1_win_pct": (team1_wins / total * 100) if total > 0 else 0,
        "team2_win_pct": (team2_wins / total * 100) if total > 0 else 0,
        "draw_pct": (team1_draws / total * 100) if total > 0 else 0,
    }


def format_h2h_match_display(match, team1):
    """Format a single match for display."""
    
    if match['home_team'] == team1:
        home = team1
        away = match['away_team']
        home_score = match['home_score']
        away_score = match['away_score']
    else:
        home = match['home_team']
        away = team1
        home_score = match['home_score']
        away_score = match['away_score']
    
    if home_score > away_score:
        winner = home
        result = "W" if home == team1 else "L"
    elif home_score < away_score:
        winner = away
        result = "L" if home == team1 else "W"
    else:
        winner = "Draw"
        result = "D"
    
    return {
        "date": match['date'],
        "home": home,
        "away": away,
        "score": f"{home_score}-{away_score}",
        "result": result,
        "winner": winner
    }

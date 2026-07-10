import pandas as pd
from simulate import simulate_match
from collections import defaultdict, Counter


# World Cup Groups
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Switzerland", "Canada", "Bosnia and Herzegovina", "Qatar"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Australia", "Paraguay", "Turkey"],
    "E": ["Germany", "Ivory Coast", "Ecuador", "Curacao"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cabo Verde", "Uruguay", "Saudi Arabia"],
    "I": ["France", "Norway", "Senegal", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Colombia", "Portugal", "DR Congo", "Uzbekistan"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


def simulate_match_multiple(home, away, ratings, home_model, away_model, profiles, num_simulations=10):
    """
    Simulate a match multiple times and return the most likely winner and average score.
    
    Args:
        num_simulations: Number of times to simulate the match (default 10)
    
    Returns:
        Dict with winner (based on most wins) and average score
    """
    winners = Counter()
    total_home_goals = 0
    total_away_goals = 0
    
    for _ in range(num_simulations):
        result = simulate_match(
            home,
            away,
            ratings[home],
            ratings[away],
            home_model,
            away_model,
            profiles[home]["form"],
            profiles[away]["form"]
        )
        
        winners[result["winner"]] += 1
        
        # Parse score
        score_parts = result["score"].split("-")
        total_home_goals += int(score_parts[0])
        total_away_goals += int(score_parts[1])
    
    # Get most likely winner
    winner = winners.most_common(1)[0][0]
    
    # Calculate average score
    avg_home_goals = total_home_goals / num_simulations
    avg_away_goals = total_away_goals / num_simulations
    avg_score = f"{avg_home_goals:.1f}-{avg_away_goals:.1f}"
    
    return {
        "winner": winner,
        "score": avg_score,
        "home": home,
        "away": away
    }


def simulate_group_stage(ratings, home_model, away_model, profiles, num_match_sims=10, verbose=True):
    """
    Simulate all group stage matches and return qualified teams.
    
    Args:
        num_match_sims: Number of times to simulate each match (default 10)
    
    Returns:
        - group_standings: Dict with group results
        - qualified_teams: List of 16 teams advancing to knockout stage
    """
    
    group_standings = {}
    qualified_teams = []
    
    for group_name, teams in GROUPS.items():
        if verbose:
            print(f"\n========== GROUP {group_name} ==========")
        
        # Initialize standings for this group
        standings = {
            team: {
                "played": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "goals_for": 0,
                "goals_against": 0,
                "points": 0,
            }
            for team in teams
        }
        
        # Simulate round-robin (each team plays each other once)
        for i, home in enumerate(teams):
            for away in teams[i + 1:]:
                result = simulate_match_multiple(
                    home, away, ratings, home_model, away_model, profiles, 
                    num_simulations=num_match_sims
                )
                
                score_parts = result["score"].split("-")
                home_score = float(score_parts[0])
                away_score = float(score_parts[1])
                
                # Round to nearest integer for point calculation
                home_score_int = round(home_score)
                away_score_int = round(away_score)
                
                # Update home team stats
                standings[home]["played"] += 1
                standings[home]["goals_for"] += home_score
                standings[home]["goals_against"] += away_score
                
                # Update away team stats
                standings[away]["played"] += 1
                standings[away]["goals_for"] += away_score
                standings[away]["goals_against"] += home_score
                
                # Determine points based on rounded scores
                if home_score_int > away_score_int:
                    standings[home]["wins"] += 1
                    standings[home]["points"] += 3
                    standings[away]["losses"] += 1
                elif home_score_int < away_score_int:
                    standings[away]["wins"] += 1
                    standings[away]["points"] += 3
                    standings[home]["losses"] += 1
                else:
                    standings[home]["draws"] += 1
                    standings[home]["points"] += 1
                    standings[away]["draws"] += 1
                    standings[away]["points"] += 1
                
                if verbose:
                    print(f"{home} {result['score']} {away}")
        
        # Sort teams by points, then goal differential
        sorted_teams = sorted(
            standings.items(),
            key=lambda x: (
                -x[1]["points"],
                -(x[1]["goals_for"] - x[1]["goals_against"]),
                -x[1]["goals_for"]
            )
        )
        
        if verbose:
            print(f"\nStandings:")
            for idx, (team, stats) in enumerate(sorted_teams, 1):
                gd = stats["goals_for"] - stats["goals_against"]
                print(
                    f"{idx}. {team}: {stats['points']:.0f}pts "
                    f"({stats['wins']}W-{stats['draws']}D-{stats['losses']}L) "
                    f"GD: {gd:+.1f}"
                )
        
        # Top 2 teams advance
        qualified_teams.append(sorted_teams[0][0])
        qualified_teams.append(sorted_teams[1][0])
        
        group_standings[group_name] = {
            "standings": sorted_teams,
            "qualified": [sorted_teams[0][0], sorted_teams[1][0]]
        }
    
    return group_standings, qualified_teams


def create_knockout_bracket(qualified_teams, group_standings):
    """
    Create Round of 16 bracket from qualified teams.
    
    Bracket structure (standard World Cup):
    - Winner Group A vs Runner-up Group B
    - Winner Group B vs Runner-up Group A
    - Winner Group C vs Runner-up Group D
    - Winner Group D vs Runner-up Group C
    - etc.
    """
    
    groups_order = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    
    knockout_matches = []
    
    for i in range(0, len(groups_order), 2):
        group1 = groups_order[i]
        group2 = groups_order[i + 1]
        
        winner1 = group_standings[group1]["qualified"][0]
        runner2 = group_standings[group2]["qualified"][1]
        
        winner2 = group_standings[group2]["qualified"][0]
        runner1 = group_standings[group1]["qualified"][1]
        
        knockout_matches.append((winner1, runner2))
        knockout_matches.append((winner2, runner1))
    
    return knockout_matches


def simulate_knockout_round(
    matches,
    ratings,
    home_model,
    away_model,
    profiles,
    round_name="",
    num_match_sims=10,
    verbose=True
):
    """Simulate knockout round matches (determine winners)."""
    
    if verbose:
        print(f"\n========== {round_name} ==========")
    
    winners = []
    round_results = []
    
    for home, away in matches:
        result = simulate_match_multiple(
            home, away, ratings, home_model, away_model, profiles,
            num_simulations=num_match_sims
        )
        
        if verbose:
            print(f"{result['home']} {result['score']} {result['away']} → {result['winner']}")
        
        winners.append(result["winner"])
        round_results.append({
            "home": result["home"],
            "away": result["away"],
            "score": result["score"],
            "winner": result["winner"]
        })
    
    return winners, round_results


def simulate_world_cup(ratings, home_model, away_model, profiles, num_match_sims=10, verbose=True):
    """
    Simulate entire World Cup from group stage to final.
    
    Args:
        num_match_sims: Number of times to simulate each match (default 10)
    
    Returns:
        - champion: Winning team
        - group_stage_results: Group standings
        - knockout_results: Dict with each round results
    """
    
    # Simulate group stage
    group_standings, qualified_teams = simulate_group_stage(
        ratings, home_model, away_model, profiles, num_match_sims=num_match_sims, verbose=verbose
    )
    
    knockout_results = {}
    
    # Create Round of 16 bracket
    round_16_matches = create_knockout_bracket(qualified_teams, group_standings)
    winners_16, results_16 = simulate_knockout_round(
        round_16_matches,
        ratings,
        home_model,
        away_model,
        profiles,
        round_name="ROUND OF 16",
        num_match_sims=num_match_sims,
        verbose=verbose
    )
    knockout_results["Round of 16"] = results_16
    
    # Quarter Finals (pair up winners)
    qf_matches = [(winners_16[i], winners_16[i + 1]) for i in range(0, len(winners_16), 2)]
    winners_qf, results_qf = simulate_knockout_round(
        qf_matches,
        ratings,
        home_model,
        away_model,
        profiles,
        round_name="QUARTER FINALS",
        num_match_sims=num_match_sims,
        verbose=verbose
    )
    knockout_results["Quarter Finals"] = results_qf
    
    # Semi Finals
    sf_matches = [(winners_qf[i], winners_qf[i + 1]) for i in range(0, len(winners_qf), 2)]
    winners_sf, results_sf = simulate_knockout_round(
        sf_matches,
        ratings,
        home_model,
        away_model,
        profiles,
        round_name="SEMI FINALS",
        num_match_sims=num_match_sims,
        verbose=verbose
    )
    knockout_results["Semi Finals"] = results_sf
    
    # Final
    final_matches = [(winners_sf[0], winners_sf[1])]
    winners_final, results_final = simulate_knockout_round(
        final_matches,
        ratings,
        home_model,
        away_model,
        profiles,
        round_name="FINAL",
        num_match_sims=num_match_sims,
        verbose=verbose
    )
    knockout_results["Final"] = results_final
    
    champion = winners_final[0]
    
    if verbose:
        print(f"\n========== CHAMPION ==========")
        print(champion)
    
    return champion, group_standings, knockout_results

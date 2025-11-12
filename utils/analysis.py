import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
from utils.database import execute_query

def round_floats(obj):
    """Recursively round all float values to 2 decimal places"""
    if isinstance(obj, float):
        return round(obj, 2)
    elif isinstance(obj, dict):
        return {k: round_floats(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [round_floats(x) for x in obj]
    return obj

def calculate_correlations(df: pd.DataFrame) -> List[Dict]:
    """
    Calculate correlations between productivity and other factors
    (Phase 2 code!)
    """
    factors = [
        'mood', 'sleep_hours', 'stress', 
        'physical_activity_min', 'screen_time_hours'
    ]
    
    correlations = []
    
    for factor in factors:
        if factor in df.columns and df[factor].notna().sum() > 0:
            corr = df['productivity'].corr(df[factor])
            
            if not np.isnan(corr):
                # Determine strength
                abs_corr = abs(corr)
                if abs_corr >= 0.7:
                    strength = "STRONG"
                elif abs_corr >= 0.4:
                    strength = "MODERATE"
                else:
                    strength = "WEAK"
                
                correlations.append({
                    "factor": factor.replace('_', ' ').title(),
                    "correlation": round(corr, 2),
                    "is_booster": bool(corr > 0),
                    "strength": strength
                })
    
    # Sort by absolute correlation (strongest first)
    correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    return correlations

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from raw data
    (Phase 3 code!)
    """
    # Sleep Deficit
    df['Sleep_Deficit'] = 8 - df['sleep_hours']
    
    # Is Weekend
    df['log_date'] = pd.to_datetime(df['log_date'])
    df['Is_Weekend'] = df['log_date'].dt.day_name().isin(['Saturday', 'Sunday'])
    
    # Is Active
    df['Is_Active'] = df['physical_activity_min'] >= 30
    
    # High Screen Time
    df['High_Screen_Time'] = df['screen_time_hours'] > 6
    
    # High Mood
    df['High_Mood'] = df['mood'] >= 8
    
    # High Productivity
    df['High_Productivity'] = df['productivity'] >= 8
    
    # Sleep Quality Score (if categorical)
    if 'sleep_quality' in df.columns:
        sleep_quality_map = {
            'Poor': 1,
            'Fair': 2,
            'Good': 3,
            'Excellent': 4
        }
        df['Sleep_Quality_Score'] = df['sleep_quality'].map(sleep_quality_map)
    
    # Diet Quality Score (if categorical)
    if 'diet_quality' in df.columns:
        diet_quality_map = {
            'Poor': 1,
            'Average': 2,
            'Good': 3
        }
        df['Diet_Quality_Score'] = df['diet_quality'].map(diet_quality_map)
    
    return df

def get_boosters_and_drainers(correlations: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Separate boosters (positive correlations) and drainers (negative)
    """
    boosters = [c for c in correlations if c['is_booster']][:3]
    drainers = [c for c in correlations if not c['is_booster']][:3]
    
    return boosters, drainers

def get_top_recommendation(correlations: List[Dict]) -> Dict:
    """
    Get the top recommendation based on strongest correlation
    """
    if not correlations:
        return None
    
    top = correlations[0]
    
    return {
        "factor": top['factor'],
        "correlation": top['correlation'],
        "strength": top['strength'],
        "potential_gain": abs(top['correlation']) * 2,  # Potential productivity boost
        "is_booster": top['is_booster']
    }

def create_action_plan(boosters: List[Dict], drainers: List[Dict]) -> List[Dict]:
    """
    Create actionable plan based on correlations
    (Phase 4 code!)
    """
    action_plan = []
    priority = 1
    
    # Actions for boosters
    action_map = {
        "Mood": {
            "emoji": "😊",
            "actions": [
                "Morning: Write 3 gratitudes",
                "Midday: 30-min mood-boost activity",
                "Evening: Connect with 1 person"
            ],
            "success_metric": "Mood score ≥ 7 on at least 5 days"
        },
        "Sleep Hours": {
            "emoji": "😴",
            "actions": [
                "Set consistent bedtime (10 PM)",
                "No screens 1 hour before bed",
                "Morning: 10 min sunlight exposure"
            ],
            "success_metric": "Sleep 7-8 hours on at least 5 nights"
        },
        "Physical Activity Min": {
            "emoji": "🏃",
            "actions": [
                "Morning: 20-min walk or workout",
                "Take stairs instead of elevator",
                "Evening: 10-min stretching"
            ],
            "success_metric": "30+ minutes activity on 5 days"
        },
        "Stress": {
            "emoji": "🧘",
            "actions": [
                "Morning: 5-min meditation",
                "Midday: 10-min walk break",
                "Evening: Brain dump worries"
            ],
            "success_metric": "Stress score ≤ 5 on at least 5 days"
        },
        "Screen Time Hours": {
            "emoji": "📱",
            "actions": [
                "Set app time limits (2 hours max)",
                "No phone during meals",
                "Evening: Device-free hour before bed"
            ],
            "success_metric": "Screen time < 6 hours on 5 days"
        }
    }
    
    # Add top 3 boosters to action plan
    for booster in boosters:
        factor = booster['factor']
        if factor in action_map:
            action_plan.append({
                "priority": priority,
                "factor": factor,
                "emoji": action_map[factor]["emoji"],
                "correlation": booster['correlation'],
                "strength": booster['strength'],
                "daily_actions": action_map[factor]["actions"],
                "success_metric": action_map[factor]["success_metric"],
                "potential_impact": abs(booster['correlation']) * 2
            })
            priority += 1
    
    return action_plan

def calculate_population_correlations() -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate average correlations across ALL users in the database
    This becomes our baseline for comparison
    
    Returns:
        (population_avg, population_std): Dictionaries with average and std correlations
    """
    # Fetch ALL logs from ALL users who have 5+ days of data
    query = """
        SELECT user_id, COUNT(*) as log_count
        FROM daily_logs
        GROUP BY user_id
        HAVING COUNT(*) >= 5
    """
    users_with_data = execute_query(query, fetch_all=True)
    
    if not users_with_data:
        return {}, {}
    
    # Factors to analyze
    factors = {
        'mood': 'Mood',
        'sleep_hours': 'Sleep Hours',
        'stress': 'Stress',
        'physical_activity_min': 'Physical Activity Min',
        'screen_time_hours': 'Screen Time Hours'
    }
    
    # Store each user's correlations
    all_user_correlations = {}
    
    # Loop through users
    for user_row in users_with_data:
        user_id = user_row['user_id']
        
        # Fetch user's logs
        query = """
            SELECT * FROM daily_logs
            WHERE user_id = %s
            ORDER BY log_date
        """
        user_logs = execute_query(query, params=(user_id,), fetch_all=True)
        
        if not user_logs or len(user_logs) < 5:
            continue
        
        # Convert to DataFrame
        user_df = pd.DataFrame(user_logs)
        user_corrs = {}
        
        # Calculate correlations for this user
        for factor_col, factor_name in factors.items():
            if factor_col in user_df.columns:
                corr = user_df['productivity'].corr(user_df[factor_col])
                if not np.isnan(corr):
                    user_corrs[factor_name] = float(corr)
        
        if user_corrs:
            all_user_correlations[user_id] = user_corrs
    
    # Calculate population averages and standard deviations
    population_avg = {}
    population_std = {}
    
    for factor_name in factors.values():
        # Collect all users' correlations for this factor
        factor_correlations = [
            user_corrs[factor_name] 
            for user_corrs in all_user_correlations.values() 
            if factor_name in user_corrs
        ]
        
        if factor_correlations:
            population_avg[factor_name] = float(np.mean(factor_correlations))
            population_std[factor_name] = float(np.std(factor_correlations))
    
    return population_avg, population_std, len(all_user_correlations)

def compare_to_population(
    user_correlations: List[Dict],
    population_avg: Dict[str, float],
    population_std: Dict[str, float]
) -> List[Dict]:
    """
    Compare user's correlations to population averages
    
    Returns list of comparisons with insights
    """
    comparisons = []
    
    for user_corr in user_correlations:
        factor = user_corr['factor']
        user_value = user_corr['correlation']
        
        # Get population average for this factor
        pop_avg = population_avg.get(factor, 0)
        pop_std_val = population_std.get(factor, 0)
        
        # Calculate difference (using absolute values for sensitivity)
        user_abs = abs(user_value)
        pop_abs = abs(pop_avg)
        abs_difference = user_abs - pop_abs
        
        # Determine comparison category
        if abs_difference > 0.2:
            comparison = "MORE sensitive"
            emoji = "🔴"
        elif abs_difference < -0.2:
            comparison = "LESS sensitive"
            emoji = "🟢"
        else:
            comparison = "TYPICAL"
            emoji = "🟡"
        
        # Calculate z-score (how many standard deviations away from mean)
        if pop_std_val > 0:
            z_score = (user_abs - pop_abs) / pop_std_val
        else:
            z_score = 0
        
        comparisons.append({
            "factor": factor,
            "user_correlation": user_value,
            "population_avg": pop_avg,
            "difference": user_value - pop_avg,
            "absolute_difference": abs_difference,
            "comparison": comparison,
            "emoji": emoji,
            "z_score": float(z_score),
            "interpretation": get_interpretation(factor, comparison, abs_difference)
        })
    
    return comparisons

def get_interpretation(factor: str, comparison: str, abs_diff: float) -> str:
    """
    Generate human-readable interpretation of the comparison
    """
    if comparison == "TYPICAL":
        return f"Your {factor} sensitivity is similar to most users."
    
    sensitivity = "more" if comparison == "MORE sensitive" else "less"
    
    interpretations = {
        "Mood": f"You are {sensitivity} affected by mood changes than average. {abs_diff:.2f} points difference.",
        "Sleep Hours": f"Sleep impacts your productivity {sensitivity} than typical. {abs_diff:.2f} points difference.",
        "Stress": f"You are {sensitivity} sensitive to stress than most users. {abs_diff:.2f} points difference.",
        "Physical Activity Min": f"Exercise affects you {sensitivity} than average. {abs_diff:.2f} points difference.",
        "Screen Time Hours": f"Screen time impacts you {sensitivity} than typical. {abs_diff:.2f} points difference."
    }
    
    return interpretations.get(factor, f"You are {sensitivity} sensitive to {factor} than average.")

def analyze_user_data(logs: List[Dict], user_id: int) -> Dict:
    """
    Main analysis function - runs all phases INCLUDING population comparison
    
    Args:
        logs: List of user's daily logs (from database)
        user_id: User ID
    
    Returns:
        Complete analysis results with population comparison
    """
    # Convert to DataFrame
    df = pd.DataFrame(logs)
    
    if len(df) < 7:
        return {
            "error": f"Need at least 7 days of data. You have {len(df)} days.",
            "days_needed": 7 - len(df)
        }
    
    # Engineer features
    df = engineer_features(df)
    
    # Calculate user's correlations
    correlations = calculate_correlations(df)
    
    # Calculate population correlations
    population_avg, population_std, population_user_count = calculate_population_correlations()
    
    # Compare user to population
    population_comparison = compare_to_population(
        correlations, 
        population_avg, 
        population_std
    )
    
    # Get boosters and drainers
    boosters, drainers = get_boosters_and_drainers(correlations)
    
    # Top recommendation
    top_rec = get_top_recommendation(correlations)
    
    # Action plan
    action_plan = create_action_plan(boosters, drainers)
    
    # Summary stats
    summary = {
        "avg_productivity": round(float(df['productivity'].mean()), 2),
        "avg_mood": round(float(df['mood'].mean()), 2),
        "avg_sleep": round(float(df['sleep_hours'].mean()), 2),
        "avg_stress": round(float(df['stress'].mean()), 2)
    }
    
    # Time series data for charts
    time_series = df[['log_date', 'mood', 'productivity', 'stress', 'sleep_hours', 'physical_activity_min']].copy()
    time_series['log_date'] = time_series['log_date'].dt.strftime('%Y-%m-%d')
    time_series = time_series.to_dict('records')
    
    # Round all numeric values in time series
    time_series = round_floats(time_series)
    
    result = {
        "user_id": user_id,
        "days_logged": len(df),
        "date_range": {
            "start": df['log_date'].min().strftime('%Y-%m-%d'),
            "end": df['log_date'].max().strftime('%Y-%m-%d')
        },
        "summary": summary,
        "correlations": round_floats(correlations),
        "boosters": round_floats(boosters),
        "drainers": round_floats(drainers),
        "top_recommendation": round_floats(top_rec) if top_rec else None,
        "action_plan": round_floats(action_plan),
        "time_series": time_series,
        "population_comparison": round_floats(population_comparison),
        "population_stats": {
            "averages": round_floats(population_avg),
            "std_deviations": round_floats(population_std),
            "users_analyzed": population_user_count
        }
    }
    
    return result
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, date
from utils.action_library import action_library
from utils.database import execute_query
import math
import random

def round_floats(obj):
    """
    Recursively round floats to 2 decimals.
    Converts NaN/Infinity to None (null) for valid JSON.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None # JSON standard uses null, not NaN
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

def create_action_plan(correlations: List[Dict]) -> List[Dict]:
    """
    Create actionable plan based on correlations
    (Phase 4 code!)
    """
    action_plan = []
    priority = 1
    
    top_3_factors = correlations[:3]
    
    # Add top 3 factors to action plan
    for correlation in top_3_factors:
        factor = correlation['factor']
        if factor in action_library:
            strategy = random.choice(action_library[factor]["strategies"])
            action_plan.append({
                "priority": priority,
                "factor": factor,
                "emoji": action_library[factor]["emoji"],
                "correlation": correlation['correlation'],
                "strength": correlation['strength'],
                "title": strategy["title"],
                "daily_actions": strategy["actions"],
                "success_metric": strategy["metric"],
                "potential_impact": abs(correlation['correlation']) * 2
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

def calculate_general_lift(df: pd.DataFrame, factor_col, factor_name, is_booster=True):
    """
    Calculates productivity difference between High vs Low values of ANY factor.
    """
    clean_df = df.dropna(subset=[factor_col, 'productivity']).copy()
    if clean_df.empty: return None
    
    # 1. Determine Threshold
    threshold = clean_df[factor_col].median()
    
    # FIX: If median is 0 (common for exercise), default to a meaningful minimum
    if threshold == 0:
        if 'min' in factor_col: # Physical Activity
            threshold = 15.0 # Anything > 15 mins counts as "Active"
        elif 'hours' in factor_col: # Screen Time / Social
            threshold = 1.0 
            
    # 2. Split groups
    if is_booster:
        high_group = clean_df[clean_df[factor_col] >= threshold]
        low_group = clean_df[clean_df[factor_col] < threshold]
        # Dynamically describe the groups for clearer insights
        group_desc = f"{factor_name} is high (> {int(threshold)})"
    else:
        # For drainers, we compare "Good State" (Low) vs "Bad State" (High)
        high_group = clean_df[clean_df[factor_col] < threshold]
        low_group = clean_df[clean_df[factor_col] >= threshold]
        group_desc = f"{factor_name} is low (< {int(threshold)})"

    if len(high_group) < 2 or len(low_group) < 2: return None # Lowered to 2 for small datasets
    
    # 3. Calculate Lift
    avg_high = high_group['productivity'].mean()
    avg_low = low_group['productivity'].mean()
    
    if avg_low == 0: return None
    
    lift = ((avg_high - avg_low) / avg_low) * 100
    diff = avg_high - avg_low
    
    # 4. Generate Insight
    # Handle "Massive" Lift (> 100%)
    if lift >= 100:
        return f"🚀 You are **twice as productive** (+{diff:.1f} pts) when {group_desc}."
        
    # Handle "Big" Lift (> 50%)
    elif lift > 50:
        return f"🚀 You are **{int(lift)}% more productive** (+{diff:.1f} pts) when {group_desc}."
        
    # Handle "Moderate" Lift (> 5%)
    elif lift > 5:
        return f"📈 You get a **{int(lift)}% boost** (+{diff:.1f} pts) when {group_desc}."
    
    elif lift < -50:
        return f"⚠️ **Critical Drain:** Your productivity is **cut in half** (-{abs(diff):.1f} pts) when {group_desc}."
        
    # "Big Drop" (20% - 50% loss)
    elif lift < -20:
        return f"⚠️ You are **{abs(int(lift))}% less productive** (-{abs(diff):.1f} pts) when {group_desc}."
        
    # "Moderate Drop" (5% - 20% loss)
    elif lift < -5:
        return f"📉 You experience a **{abs(int(lift))}% dip** in productivity (-{abs(diff):.1f} pts) when {group_desc}."
    
    return None

def find_optimal_factor_zone(df, factor_col, factor_name):
    """
    Finds the 'Sweet Spot' for any numerical factor.
    """
    clean_df = df.dropna(subset=[factor_col, 'productivity']).copy()
    if clean_df.empty: return None
    
    # 1. Create Bins dynamically based on data range
    min_val = clean_df[factor_col].min() # 2
    max_val = clean_df[factor_col].max() # 10
    
    # Create ~4-5 bins
    if max_val - min_val < 2: return None # Range too small
    
    # Custom bin sizes based on factor type
    if 'hours' in factor_col:
        step = 1.0
    elif 'min' in factor_col: 
        step = 15.0 # 15 min chunks
    else:
        step = 2.0 # 1-10 scales
        
    clean_df['temp_bin'] = clean_df[factor_col].apply(lambda x: round(x / step) * step)
    
    # 2. Analyze Bins
    min_required = 1 if len(clean_df) < 15 else 2
    stats = clean_df.groupby('temp_bin')['productivity'].agg(['mean', 'count'])
    valid_stats = stats[stats['count'] >= min_required] # Need valid data
    
    if valid_stats.empty: return None
    
    # 3. Find Peak
    best_bin = valid_stats['mean'].idxmax()
    
    # 4. Format Output
    if 'hours' in factor_col:
        unit = "hours"
    elif 'min' in factor_col:
        unit = "minutes"
    else:
        unit = "points"
        
    # Polish the text for 0 values
    if best_bin == 0:
        if factor_name == "Stress":
            return f"Your peak productivity happens when you have **zero stress**."
        if factor_name == "Screen Time Hours":
            return f"Your peak productivity happens with **zero screen time**."
            
    return f"Your peak productivity happens at {best_bin} {unit} of {factor_name}."

def predict_today_productivity(df: pd.DataFrame):
    """
    Predicts today's productivity based on sleep and last mood
    """
    
    # 1. Prepare Data: Sort by date
    df = df.sort_values('log_date')
    
    # 2. Create Previous Day Mood
    df['previous_day_mood'] = df['mood'].shift(1)
    
    # Get the very last log entry from the DB
    if df.empty: return None
    current_log = df.iloc[-1]
    
    # Check 1: Is the last log actually from TODAY?
    # Ensure formats match (date object comparison)
    today = date.today()
    if current_log['log_date'].date() != today:
        return None  # User hasn't started a log for today yet
        
    # Check 2: Did they log Sleep Hours? (And is it not None/NaN)
    if pd.isna(current_log['sleep_hours']) or current_log['sleep_hours'] == 0:
        return None
        
    # Check 3: Do we have "Yesterday's Mood"?
    # If this is the very first day, shift(1) will be NaN
    if pd.isna(current_log['previous_day_mood']):
        return None  # Cannot predict without history
    
    # 3. Drop rows with missing data (e.g., the first day has no "yesterday")
    model_df = df[['previous_day_mood', 'sleep_hours', 'productivity']].dropna()
    
    # Need at least ~10 days of continuous data for a decent prediction
    if len(model_df) < 10:
        return "Keep logging! We need more continuous days to predict your productivity."
    
    # 4. Prepare Matrices for Linear Regression (y = mx + c)
    Y = model_df['productivity'].values
    X = model_df[['sleep_hours', 'previous_day_mood']].values
    # Add column of ones for intercept
    X = np.c_[X, np.ones(X.shape[0])]
    
    # 5. Train Model (Least Squares)
    # weights[0] = sleep weight, weights[1] = mood weight, weights[2] = bias
    weights, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    
    # 6. Make Prediction for "Today"
    prediction = weights[0] * current_log['sleep_hours'] + weights[1] * current_log['previous_day_mood'] + weights[2]
    
    predicted_score = max(1, min(10, round(prediction, 1)))
    
    # Clamp result between 1-10
    predicted_score = max(1, min(10, round(predicted_score, 1)))
    
    return f"Based on your sleep and yesterday's mood, today looks like a {predicted_score}/10 productivity day."
    
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
    action_plan = create_action_plan(correlations)
    
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
    
    smart_insights = []
    
    # Predict today's productivity
    predicted_productivity = predict_today_productivity(df)
    
    # Add prediction to insights
    if predicted_productivity and isinstance(predicted_productivity, str):
        smart_insights.append({
            "type": "prediction", 
            "message": predicted_productivity,
            "priority": 1
        })
    
    top_factors = correlations[:3]
    
    for factor in top_factors:
        col_name = factor['factor'].lower().replace(' ', '_')
        
        # Try to find optimal zone
        if factor['factor'] not in ["Mood", "Diet Quality", "Social Interaction"]:
            display_name = factor['factor']
            display_name = display_name.replace(' Hours', '').replace(' Min', '').replace(' Score', '') 
            opt_msg = find_optimal_factor_zone(df, col_name, display_name)
            if opt_msg:
                smart_insights.append({
                    "type": "optimization", 
                    "message": opt_msg,
                    "priority": 2
                })
                
        lift_msg = calculate_general_lift(df, col_name, factor['factor'], factor['is_booster'])
        if lift_msg:
            smart_insights.append({
                "type": "impact",
                "message": lift_msg,
                "priority": 3
            })

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
        },
        "smart_insights": smart_insights
    }
    
    return result
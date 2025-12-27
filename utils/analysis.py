import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, date, timedelta
from utils.action_library import action_library
from utils.database import execute_query
import math
import random
import hashlib
from utils.constants import TAG_LABELS, TAG_ADVICE

def round_floats(obj):
    """
    Recursively round floats to 2 decimals.
    Converts NaN/Infinity to None (null) for valid JSON.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None 
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
        'physical_activity_min', 'morning_workout', 
        'afternoon_workout', 'evening_workout', 'screen_time_hours'
    ]
    
    correlations = []
    
    for factor in factors:
        if factor in df.columns and df[factor].notna().sum() > 0:
            corr = df['productivity'].corr(df[factor])
            
            if not np.isnan(corr):
                abs_corr = abs(corr)
                if abs_corr >= 0.7:
                    strength = "STRONG"
                elif abs_corr >= 0.4:
                    strength = "MODERATE"
                else:
                    strength = "WEAK"
                
                clean_name = factor.replace('_', ' ').title().replace(' Hours', '').replace(' Min', '')

                correlations.append({
                    "factor": clean_name,
                    "original_col": factor, 
                    "correlation": round(corr, 2),
                    "is_booster": bool(corr > 0),
                    "strength": strength
                })
    
    correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    timing_factors = ['Morning Workout', 'Afternoon Workout', 'Evening Workout']
    selected_factors = []
    seen_workout = False

    for factor in correlations:
        if factor['factor'] in timing_factors:
            if seen_workout or not factor['is_booster']: continue
            seen_workout = True
            
        selected_factors.append(factor)
        
    return selected_factors

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from raw data
    (Phase 3 code!)
    """
    # Sleep Deficit
    df['sleep_deficit'] = 8 - df['sleep_hours']
    
    # Is Weekend
    df['log_date'] = pd.to_datetime(df['log_date'])
    df['is_weekend'] = df['log_date'].dt.day_name().isin(['Saturday', 'Sunday'])
    df['day_name'] = df['log_date'].dt.day_name()
    
    # Is Active
    df['is_active'] = df['physical_activity_min'] >= 30
    
    if 'activity_time' in df.columns:
        # 1. Morning Workout
        df['morning_workout'] = (
            (df['physical_activity_min'] >= 22) & 
            (df['activity_time'] == 'Morning')
        ).astype(int)

        # 2. Afternoon Workout
        df['afternoon_workout'] = (
            (df['physical_activity_min'] >= 22) & 
            (df['activity_time'] == 'Afternoon')
        ).astype(int)

        # 3. Evening Workout
        df['evening_workout'] = (
            (df['physical_activity_min'] >= 22) & 
            (df['activity_time'] == 'Evening')
        ).astype(int)
        
    
    # High Screen Time
    df['high_screen_time'] = df['screen_time_hours'] > 6
    
    # High Mood
    df['high_mood'] = df['mood'] >= 8
    
    # High Productivity
    df['high_productivity'] = df['productivity'] >= 8
    
    # Sleep Quality Score (if categorical)
    if 'sleep_quality' in df.columns:
        sleep_quality_map = {
            'Poor': 1,
            'Fair': 2,
            'Good': 3,
            'Excellent': 4
        }
        df['sleep_quality_score'] = df['sleep_quality'].map(sleep_quality_map)
    
    # Diet Quality Score (if categorical)
    if 'diet_quality' in df.columns:
        diet_quality_map = {
            'Poor': 1,
            'Average': 2,
            'Good': 3
        }
        df['diet_quality_score'] = df['diet_quality'].map(diet_quality_map)
    
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
    factor_name = top['factor']
    
    factor_meta = {
        "Mood": {"step": 2, "unit": "points", "direction": "Increasing"},
        "Stress": {"step": 2, "unit": "points", "direction": "Lowering"},
        "Sleep Hours": {"step": 1, "unit": "hour", "direction": "Increasing"},
        "Screen Time Hours": {"step": 1, "unit": "hour", "direction": "Reducing"},
        "Physical Activity Min": {"step": 30, "unit": "minutes", "direction": "Increasing"},
        "Diet Quality": {"step": 1, "unit": "level", "direction": "Improving"},
        "Morning Workout": {"step": 1, "unit": "session", "direction": "Prioritizing"},
        "Afternoon Workout": {"step": 1, "unit": "session", "direction": "Prioritizing"},
        "Evening Workout": {"step": 1, "unit": "session", "direction": "Prioritizing"},
    }
    
    meta = factor_meta.get(factor_name, {"step": 2, "unit": "points", "direction": "Improving"})
    
    if factor_name not in factor_meta:
        if top['correlation'] < 0:
            meta['direction'] = "Reducing"
    
    return {
        "factor": factor_name,
        "original_col": top['original_col'],
        "correlation": top['correlation'],
        "strength": top['strength'],
        "potential_gain": abs(top['correlation']) * 2,
        "is_booster": top['is_booster'],
        "action_label": meta['direction'],
        "improvement_step": meta['step'],
        "improvement_unit": meta['unit']
    }

def calculate_factor_averages(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculates the average for specific factors, handling non-numeric data gracefully.
    Returns a dictionary mapped to the "Nice Names" used in the Action Library.
    """
    averages = {}
    
    # Map DB column names to Action Library keys
    column_map = {
        'mood': 'Mood',
        'sleep_hours': 'Sleep',
        'stress': 'Stress',
        'physical_activity_min': 'Physical Activity',
        'screen_time_hours': 'Screen Time',
        'diet_quality_score': 'Diet Quality',
        'social_interaction_hours': 'Social Interaction' 
    }

    for col, nice_name in column_map.items():
        if col in df.columns:
            # force numeric, turning errors to NaN
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            mean_val = numeric_series.mean()
            
            if not math.isnan(mean_val):
                averages[nice_name] = float(mean_val)
            else:
                averages[nice_name] = 0.0
                
    return averages

def generate_smart_goal(factor: str, current_avg: float, template: str) -> str:
    """
    Calculates a realistic target based on the user's current average.
    """
    target = 0
    
    if factor == "Mood":
        # Aim for +1, max 10
        target = min(10, round(current_avg + 0.5, 1))
        # Fallback if average is already high
        if target < 7: target = 7 

    elif factor == "Sleep":
        # Aim for +1 hour, max 8
        target = min(8, round(current_avg + 1.0, 1))
        if target < 7: target = 7

    elif factor == "Stress":
        # Aim for -1, min 1
        target = max(1, round(current_avg - 1.0, 1))
        if target > 4: target = 4 # Cap "Success" at 4 or lower

    elif factor == "Physical Activity":
        # Aim for +15 mins
        target = round(current_avg + 15)
        if target < 20: target = 20
        
    elif factor == "Diet Quality":
        # Map: 1=Poor, 2=Average, 3=Good
        current_level = round(current_avg)
        
        # Goal: Aim for one level higher, maxing out at 3 (Good)
        target_level = min(3, current_level + 1)
        
        # If they are already "Good" (3), keep it "Good"
        if target_level < 2: target_level = 2 # Minimum goal "Average"
        
        # Convert number back to text for the UI
        level_map = {1: "Poor", 2: "Average", 3: "Good"}
        target_str = level_map.get(target_level, "Good")
        
        return template.format(target=target_str)

    elif factor == "Screen Time":
        # Aim for -1 hour, min 2
        target = max(2.0, round(current_avg - 1.0, 1))
        
    elif factor == "Social Interaction":
        target = 1 # Default baseline
    
    # Inject the calculated target into the template
    # Example: "Sleep {target} hours" -> "Sleep 7.5 hours"
    return template.format(target=target)

def create_action_plan(correlations: List[Dict], user_id: int, df: pd.DataFrame) -> List[Dict]:
    """
    Create actionable plan based on correlations with DYNAMIC metrics
    """
    action_plan = []
    priority = 1
    top_3_factors = correlations[:3]
    
    # 1. Calculate User's Actual Averages
    user_averages = calculate_factor_averages(df)
    
    # Get today's date string for seeding
    today_str = date.today().strftime('%Y-%m-%d')
    
    for correlation in top_3_factors:
        factor = correlation['factor']

        if factor in action_library:
            strategies = action_library[factor]["strategies"]
            
            # Deterministic Randomness (Rotates daily, stable for 24h)
            seed_str = f"{user_id}-{factor}-{today_str}"
            seed_int = int(hashlib.sha256(seed_str.encode('utf-8')).hexdigest(), 16)
            random.seed(seed_int)
            
            strategy = random.choice(strategies)
            
            # 2. Generate Dynamic Metric String
            # Get user's average for this factor (default to 0 if missing)
            current_avg = user_averages.get(factor, 0)
            
            # Inject smart numbers into the text
            # e.g. "Screen time < {target} hours" becomes "Screen time < 3.5 hours"
            dynamic_metric = generate_smart_goal(
                factor, 
                current_avg, 
                strategy["metric_template"]
            )
            
            action_plan.append({
                "priority": priority,
                "factor": factor,
                "icon": action_library[factor]["icon"],
                "correlation": correlation['correlation'],
                "strength": correlation['strength'],
                "title": strategy["title"],
                "daily_actions": strategy["actions"],
                "success_metric": dynamic_metric, 
                "potential_impact": abs(correlation['correlation']) * 2
            })
            priority += 1
    
    recent_logs = df.tail(3)
    has_low_energy = recent_logs['tags'].apply(lambda x: 'Low Energy' in x if isinstance(x, list) else False).any()
    
    if has_low_energy:
        # Force a specific action card to appear at the top
        action_plan.insert(0, {
            "priority": 0, # Top priority
            "factor": "Energy Management",
            "icon": "BatteryCharging", # Lucid icon name
            "title": "Recharge Protocol",
            "daily_actions": [
                "Go to bed 30 mins earlier tonight",
                "No caffeine after 2 PM",
                "Take a 15 min walk in sunlight"
            ],
            "success_metric": "Tag 'High Energy' tomorrow",
            "potential_impact": "High"
        })
    
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
        factor = user_corr['original_col']
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
    
    if threshold == 0:
        if 'min' in factor_col: # Physical Activity
            threshold = 15.0 
        elif 'hours' in factor_col: # Screen Time / Social
            threshold = 1.0 
        elif 'workout' in factor_name.lower(): 
            threshold = 0.5
            
    # 2. Split groups
    if is_booster:
        high_group = clean_df[clean_df[factor_col] >= threshold]
        low_group = clean_df[clean_df[factor_col] < threshold]
        
        if factor_name == "Morning Workout":
            group_desc = "you workout in the morning"
        elif factor_name == "Afternoon Workout":
            group_desc = "you workout in the afternoon"
        elif factor_name == "Evening Workout":
            group_desc = "you workout in the evening"
        else:
            group_desc = f"{factor_name} is high (> {int(threshold)})"
    else:
        # For drainers
        high_group = clean_df[clean_df[factor_col] < threshold]
        low_group = clean_df[clean_df[factor_col] >= threshold]
        
        if factor_name == "Morning Workout":
            group_desc = "you don't workout in the morning"
        elif factor_name == "Afternoon Workout":
            group_desc = "you don't workout in the afternoon"
        elif factor_name == "Evening Workout":
            group_desc = "you don't workout in the evening"
        else:
            group_desc = f"{factor_name} is low (< {int(threshold)})"

    if len(high_group) < 2 or len(low_group) < 2: return None 
    
    # 3. Calculate Lift
    avg_high = high_group['productivity'].mean()
    avg_low = low_group['productivity'].mean()
    
    if avg_low == 0: return None
    
    lift = ((avg_high - avg_low) / avg_low) * 100
    
    score_label = f"(Avg {avg_high:.1f} vs {avg_low:.1f})"
    
    # 4. Generate Insight
    # Handle "Massive" Lift (> 100%)
    if lift >= 100:
        return f"🚀 You are **twice as productive** {score_label} when {group_desc}."
        
    # Handle "Big" Lift (> 50%)
    elif lift > 50:
        return f"🚀 You are **{int(lift)}% more productive** {score_label} when {group_desc}."
        
    # Handle "Moderate" Lift (> 5%)
    elif lift > 5:
        return f"📈 You get a **{int(lift)}% boost** {score_label} when {group_desc}."
    
    elif lift < -50:
        return f"⚠️ **Critical Drain:** Your productivity is **cut in half** {score_label} when {group_desc}."
        
    # "Big Drop" (20% - 50% loss)
    elif lift < -20:
        return f"⚠️ You are **{abs(int(lift))}% less productive** {score_label} when {group_desc}."
        
    # "Moderate Drop" (5% - 20% loss)
    elif lift < -5:
        return f"📉 You experience a **{abs(int(lift))}% dip** in productivity {score_label} when {group_desc}."
    
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
    
    # 1. Ensure Dates are proper Datetime objects and Sorted
    df['log_date'] = pd.to_datetime(df['log_date'])
    df = df.sort_values('log_date')
    
    # 2. Calculate the "Time Gap" between rows
    # This creates a column representing how many days passed since the last log
    df['days_since_last_log'] = df['log_date'].diff().dt.days
    
    # 3. Create 'prev_mood' SAFELY
    df['prev_mood'] = df['mood'].shift(1)
    
    # This ensures we only use "Yesterday's Mood" if it was actually yesterday
    df.loc[df['days_since_last_log'] != 1, 'prev_mood'] = np.nan
    
    if df.empty: return None
    current_log = df.iloc[-1]
    
    # Check if current log is today
    if current_log['log_date'].date() != date.today():
        return None

    # Check for Morning Data
    if pd.isna(current_log['sleep_hours']) or pd.isna(current_log['mood']):
        return None 
    
    # It will be False if the user skipped yesterday.
    has_consecutive_history = not pd.isna(current_log['prev_mood'])
    
    if has_consecutive_history:
        features = ['sleep_hours', 'mood', 'prev_mood']
        
        # We only train on rows where 'days_since_last_log' == 1
        model_df = df[df['days_since_last_log'] == 1][features + ['productivity']].dropna()
        
        if len(model_df) < 5:
             pass 
        else:
            X = model_df[features].values
            Y = model_df['productivity'].values
            X = np.c_[X, np.ones(X.shape[0])]
            
            weights, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            
            prediction = (weights[0] * current_log['sleep_hours'] + 
                          weights[1] * current_log['mood'] + 
                          weights[2] * current_log['prev_mood'] + 
                          weights[3])
                          
            return format_prediction(prediction)
    
    features = ['sleep_hours', 'mood']
    model_df = df[features + ['productivity']].dropna()
    
    if len(model_df) < 5:
        return "Keep logging! We need more data."

    X = model_df[features].values
    Y = model_df['productivity'].values
    X = np.c_[X, np.ones(X.shape[0])]
    
    weights, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    
    prediction = (weights[0] * current_log['sleep_hours'] + 
                  weights[1] * current_log['mood'] + 
                  weights[2])
                  
    return format_prediction(prediction)

def format_prediction(val):
    score = max(1, min(10, round(val, 1)))
    return f"Based on your patterns, today looks like a {score}/10 productivity day."

def analyze_tag_impact(df: pd.DataFrame, baseline_productivity: float):
    """
    Analyzes how specific tags impact productivity compared to the baseline.
    Returns a list of Smart Insight objects.
    """
    insights = []
    
    # 1. Safety Check: Ensure 'tags' column exists and drop empty rows
    if 'tags' not in df.columns:
        return []
    
    tagged_df = df.dropna(subset=['tags'])
    tagged_df = tagged_df[tagged_df['tags'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    
    if tagged_df.empty:
        return []
    
    # 2. Explode the tags
    # This turns one row like [Date: 12-01, Tags: ['WFH', 'Tired']] 
    # into two rows: [Date: 12-01, Tag: 'WFH'] and [Date: 12-01, Tag: 'Tired']
    exploded_df = tagged_df.explode('tags')
    
    # 3. Group by Tag
    tag_stats = exploded_df.groupby('tags')['productivity'].agg(['mean', 'count'])
    valid_tags = tag_stats[tag_stats['count'] >= 3]
    
    if valid_tags.empty:
        return []
    
    # 4. Generate Insights
    for tag, row in valid_tags.iterrows():
        avg_prod = row['mean']
        
        if baseline_productivity == 0: continue
        
        lift = ((avg_prod - baseline_productivity) / baseline_productivity) * 100
        
        readable_tag = TAG_LABELS.get(tag, tag)
        
        if lift >= 15:
            # Positive Insight
            insights.append({
                "type": "optimization",
                "message": f"⚡ **Context Unlock:** You are **{int(lift)}% more productive** when {readable_tag}.",
                "priority": 2
            })
        elif lift <= -15:
            # Negative Insight
            advice = TAG_ADVICE.get(tag, "Try to identify potential distractions.")
            
            insights.append({
                "type": "impact", 
                "message": f"⚠️ **Context Alert:** Your productivity drops by **{abs(int(lift))}%** when {readable_tag}. {advice}",
                "priority": 2
            })
            
    return insights

def analyze_energy_drainers(df):
    """
    Finds out what leads to a "Low Energy" tag.
    Checks TODAY for Sleep issues.
    Checks YESTERDAY for Stress/Burnout "Hangovers".
    """
    if 'tags' not in df.columns: return None
    
    # 1. Identify "Low Energy" days
    df = df.copy()
    df['is_low_energy'] = df['tags'].apply(lambda x: 'low_energy' in x if isinstance(x, list) else False)
    
    # Get indices of low energy days
    low_energy_indices = df.index[df['is_low_energy']].tolist()
    
    if not low_energy_indices: return None
    
    insights = []
    
    # Calculate User's Baselines (Normal Days)
    normal_days = df[~df['is_low_energy']]
    if normal_days.empty: return None
    
    baseline_sleep = normal_days['sleep_hours'].mean()
    baseline_stress = normal_days['stress'].mean()
    
    for idx in low_energy_indices:
        today = df.loc[idx]
        
        # CHECK 1: TODAY'S SLEEP (The most likely culprit)
        # If I slept significantly less than my normal baseline
        if today['sleep_hours'] < (baseline_sleep - 1.0):
             insights.append({
                "type": "warning",
                "message": (
                    f"🔋 **Sleep Debt:** You tagged 'Low Energy' today. "
                    f"This corresponds to sleeping only {today['sleep_hours']}h (Your norm is {baseline_sleep:.1f}h)."
                )
            })
            
        # CHECK 2: YESTERDAY'S STRESS (The Hangover)
        # Only check previous day if it exists in the dataframe
        elif (idx - 1) in df.index:
            yesterday = df.loc[idx - 1]
            
            if yesterday['stress'] > (baseline_stress + 2.0):
                insights.append({
                    "type": "warning",
                    "message": (
                        f"🧠 **Stress Hangover:** You felt drained today despite sleeping okay. "
                        f"This was likely caused by High Stress ({yesterday['stress']}/10) the day before."
                    )
                })
                
        # CHECK 3: DIGITAL BURNOUT (Dopamine Crash)
        # Did they have huge screen time yesterday?
        elif (idx - 1) in df.index and df.loc[idx-1].get('screen_time_hours', 0) > 5.0:
             prev_screen = df.loc[idx-1]['screen_time_hours']
             insights.append({
                "type": "warning",
                "message": (
                    f"📱 **Digital Drain:** You slept well, but your battery is empty. "
                    f"This might be a 'dopamine crash' from high Screen Time ({prev_screen}h) yesterday."
                )
            })
             
        # CHECK 4: SEDENTARY INERTIA (The "Rust" Effect)
        # Did they barely move yesterday? (assuming 'physical_activity_min' exists)
        elif (idx - 1) in df.index and df.loc[idx-1].get('physical_activity_min', 0) < 15:
             insights.append({
                "type": "suggestion",
                "message": (
                    "🏃 **Inertia:** Your energy is low because you've been too still. "
                    "Data shows practically zero movement yesterday. Try a 10-minute walk to jumpstart your system."
                )
            })
                
    
    unique_insights = {i['message']: i for i in insights}.values()
    
    return list(unique_insights)[:2]

def generate_perfect_day_blueprint(df: pd.DataFrame) -> Dict:
    """
    Analyzes the top 10% of days to generate a "Recipe" for success.
    """
    if df.empty or len(df) < 3:
        return None
    
    # 1. Filter for High Performance Days (Top 10-15%)
    # We use quantile to find the cutoff score (e.g., productivity > 8.0)
    target_quantile = 0.85
    min_days_needed = 3
    while target_quantile >= 0.50:
        cutoff = df['productivity'].quantile(target_quantile)
        if cutoff < 5 and target_quantile > 0.5:
            target_quantile = 0.75
            continue
    
        best_days = df[df['productivity'] >= cutoff]
        if len(best_days) >= min_days_needed:
            break
        
        target_quantile -= 0.05
        
    if len(best_days) < 3:
        return None
    
    # 2. Calculate the "Perfect Stats"
    blueprint = {
        "avg_score": round(best_days['productivity'].mean(), 1),
        "mood": round(best_days['mood'].mean(), 1),
        "sleep": round(best_days['sleep_hours'].mean(), 1),
        "stress_limit": round(best_days['stress'].mean(), 1),
        "activity": round(best_days['physical_activity_min'].mean()),
        "screen_limit": round(best_days['screen_time_hours'].mean(), 1),
        "social_time": round(best_days['social_interaction_hours'].mean(), 1),
        "social_hours": round(best_days['social_interaction_hours'].mean(), 1)
    }
    
    # 3. Find the "Perfect Context" (Tags)
    # Count tags that appear in best days
    if 'tags' in best_days.columns:
        all_tags = best_days['tags'].explode().dropna()
        if not all_tags.empty:
            # Get the single most common tag in high-performance days
            blueprint['best_context'] = all_tags.mode()[0]
    
    # 4. Find "Perfect Workout Time"
    # Which workout time appears most often in best days?
    workout_times = best_days['activity_time'].dropna()
    if not workout_times.empty:
        blueprint['workout_time'] = workout_times.mode()[0] # e.g., "Morning"
        
    return blueprint

def count_blueprint_matches(df: pd.DataFrame, blueprint: Dict) -> int:
    """
    Counts how many days in the history matched the Perfect Day Blueprint.
    Reproduces the same logic as the frontend gamification.
    """
    if not blueprint or df.empty:
        return 0
    
    match_count = 0
    
    # We iterate through every log to check for matches
    for _, row in df.iterrows():
        daily_matches = 0
        
        # 1. SLEEP: +/- 0.75 hours
        if abs(row['sleep_hours'] - blueprint['sleep']) <= 0.75:
            daily_matches += 1
            
        # 2. ACTIVITY: >= Target - 10 mins
        # Handle NaN/None by treating as 0
        act = row['physical_activity_min'] if pd.notna(row['physical_activity_min']) else 0
        if act >= (blueprint['activity'] - 10):
            daily_matches += 1
            
        # 3. SCREEN: <= Target + 0.5 hours
        scr = row['screen_time_hours'] if pd.notna(row['screen_time_hours']) else 0
        if scr <= (blueprint['screen_limit'] + 0.5):
            daily_matches += 1
            
        # 4. SOCIAL: >= Target - 0.5 hours
        # Only check if social was part of the blueprint
        if 'social_hours' in blueprint:
            soc = row['social_interaction_hours'] if pd.notna(row['social_interaction_hours']) else 0
            if soc >= (blueprint['social_hours'] - 0.5):
                daily_matches += 1
        
        # THRESHOLD: If they hit 3 out of 4 (or 3/3), it's a match
        if daily_matches >= 3:
            match_count += 1
            
    return match_count

def calculate_badges(df: pd.DataFrame, blueprint: Dict) -> List[Dict]:
    """
    Calculates badges based on user's data and blueprint.
    Returns a list of badge objects.
    """
    if df.empty:
        return []
    
    badges = []
    
    # --- 1. Perfect Day Badge ---
    if blueprint:
        perfect_day_count = count_blueprint_matches(df, blueprint)
        if perfect_day_count > 0: 
            badges.append({
                "id": "perfect_day",
                "count": perfect_day_count, 
                "label": "Blueprint Days",
                "icon": "Medal",
                "color": "indigo" 
            })
        
    # --- 2. The Streak (Hot Streak) ---
    dates = pd.to_datetime(df['log_date']).dt.date.unique()
    dates = sorted(dates, reverse=True) # Sort newest first
    
    current_streak = 0
    today = date.today()
    
    if len(dates) > 0:
        if dates[0] == today or dates[0] == (today - timedelta(days=1)):
            current_streak = 1
            
            for i in range(1, len(dates)):
                if dates[i-1] - dates[i] == timedelta(days=1):
                    current_streak += 1
                else:
                    break
    
    # Only add badge if they have a meaningful streak (e.g., 3+)
    if current_streak >= 3:
        badges.append({
            "id": "streak",
            "count": current_streak,
            "label": "Day Streak", 
            "icon": "Flame",
            "color": "orange"
        })
            
    # --- 3. Diamond (Resilience) ---
    # Logic: High Stress AND High Productivity
    diamond_days = df[(df['stress'] >= 7) & (df['productivity'] >= 7)]
    if len(diamond_days) > 0:
        badges.append({
            "id": "diamond",
            "count": len(diamond_days),
            "label": "Pressure Performer",
            "icon": "Gem",
            "color": "cyan"
        })
        
    # --- 4. Unplugged (Digital Health) ---
    # Logic: Screen time less than 2 hours
    unplugged_days = df[(df['screen_time_hours'] < 2.0) & (df['screen_time_hours'].notna())]
    if len(unplugged_days) > 0:
        badges.append({
            "id": "unplugged",
            "count": len(unplugged_days),
            "label": "Digital Detox",
            "icon": "Sprout",
            "color": "green"
        })
        
    # 5. NEW: Flow State (Peak Performance)
    # Logic: High Mood AND High Productivity
    flow_days = df[(df['mood'] >= 8) & (df['productivity'] >= 8)]
    if len(flow_days) > 0:
        badges.append({
            "id": "flow_state",
            "count": len(flow_days),
            "label": "Flow State",
            "icon": "Zap",       
            "color": "yellow"    
        })

    # 6. NEW: Iron Body (High Activity)
    # Logic: More than 60 mins of exercise
    active_days = df[df['physical_activity_min'] >= 60]
    if len(active_days) > 0:
        badges.append({
            "id": "iron_body",
            "count": len(active_days),
            "label": "Iron Body",
            "icon": "Dumbbell",  
            "color": "red"       
        })

    # 7. NEW: Zen Master (Low Stress)
    # Logic: Stress level 3 or lower
    zen_days = df[df['stress'] <= 3]
    if len(zen_days) > 0:
        badges.append({
            "id": "zen_master",
            "count": len(zen_days),
            "label": "Zen Master",
            "icon": "Sun",       
            "color": "teal"      
        })
    
    return badges

def analyze_weekly_rhythm(df: pd.DataFrame) -> Dict:
    """
    Groups productivity by Day of the Week to find natural rhythms.
    """
    if df.empty:
        return None
    
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Calculate stats
    weekly_stats = df.groupby('day_name')['productivity'].mean()
    
    # Reindex ensures all days are present (even if empty/NaN)
    weekly_stats = weekly_stats.reindex(days_order)
    
    best_day = None
    max_score = -1
    worst_score = 11
    
    valid_scores = []
    
    for day in days_order:
        score = weekly_stats[day]
        
        # Handle NaN (days with no logs)
        if pd.isna(score):
            val = 0
        else:
            val = round(score, 1)
            valid_scores.append(val)
            
            if val > max_score:
                max_score = val
                best_day = day
            if val < worst_score:
                worst_score = val
                
    insight = "Keep logging to find your rhythm."
    percent_diff = 0
    
    if best_day and len(valid_scores) >= 3:
        avg_score = sum(valid_scores) / len(valid_scores)
        
        if avg_score > 0:
            percent_diff = int(((max_score - avg_score) / avg_score) * 100)
            insight = f"You are {percent_diff}% more productive on {best_day}s compared to your average."
    
    chart_data = [{"day": day, "score": 0 if pd.isna(weekly_stats[day]) else round(weekly_stats[day], 1)} for day in days_order]

    return {
        "chart_data": chart_data,
        "best_day": best_day,
        "max_score": max_score,
        "insight": insight,
        "percent_diff": percent_diff
    }
    
def get_second_order_insights(df, top_driver_col):
    """
    Finds what drives the Top Driver.
    e.g. If 'mood' is the top driver for productivity, what drives 'mood'?
    """
    # 1. Drop targets we don't care about (Productivity) and the driver itself
    potential_inputs = df.drop(columns=['productivity', top_driver_col], errors='ignore')
    
    # 2. Correlate inputs against the Top Driver (e.g., Mood)
    # Ensure numeric
    numeric_df = potential_inputs.select_dtypes(include=[np.number])
    correlations = numeric_df.corrwith(df[top_driver_col])
    
    # 3. Find the strongest influence
    # Sort by absolute value to find strongest positive OR negative driver
    strongest_influence = correlations.abs().sort_values(ascending=False).index[0]
    corr_value = correlations[strongest_influence]
    
    return {
        "driver": strongest_influence, # e.g., "sleep_hours"
        "target": top_driver_col,      # e.g., "mood"
        "correlation": corr_value      # e.g., 0.72
    }
    
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
    action_plan = create_action_plan(correlations, user_id, df)
    
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
        
    tag_insights = analyze_tag_impact(df, summary['avg_productivity'])
    smart_insights.extend(tag_insights)
    
    top_factors = correlations[:3]
    
    for factor in top_factors:
        col_name = factor.get('original_col', factor['factor'].lower().replace(' ', '_'))
        
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
            if "Workout" in factor['factor'] and factor['is_booster']:
                is_activity_drainer = any(d['factor'] == 'Physical Activity Min' for d in drainers)
                
                if is_activity_drainer:
                    clean_lift_msg = lift_msg.replace("🚀 ", "").replace("📈 ", "")
                    time_of_day = factor['factor'].replace(" Workout", "")
                    
                    lift_msg = (
                        f"🚀 **Timing Unlock:** While long workouts can sometimes drain you, "
                        f"**{time_of_day} Workouts** are your superpower. {clean_lift_msg}"
                    )
                    
                    drainers = [d for d in drainers if d['factor'] != 'Physical Activity Min']
            
            smart_insights.append({
                "type": "impact",
                "message": lift_msg,
                "priority": 3
            })
            
    perfect_day = generate_perfect_day_blueprint(df)
    
    badges = calculate_badges(df, perfect_day)
    
    weekly_rhythm = analyze_weekly_rhythm(df)
    
    if abs(top_rec['correlation']) > 0.6:
        root_cause = get_second_order_insights(df, top_rec['original_col'])
    
        if abs(root_cause['correlation']) > 0.4:
            driver_name = root_cause['driver'].replace('_', ' ').title()
            target_name = root_cause['target'].title()
        
            is_positive_target = top_rec['correlation'] > 0
            
            message = ""
            
            if is_positive_target:
                if root_cause['correlation'] > 0:
                    message = (
                        f"**Root Cause:** Your {target_name} is heavily influenced by **{driver_name}**. "
                        f"To boost your {target_name}, focus on improving {driver_name}."
                    )
                else:
                    message = (
                        f"**Root Cause:** Managing {driver_name} is key to keeping your {target_name} high."
                    )

            else:
                if root_cause['correlation'] > 0:
                    message = (
                        f"**Root Cause:** Your {target_name} is driven by **{driver_name}**. "
                        f"To lower your {target_name}, you must manage {driver_name}."
                    )
                else:
                    message = (
                        f"**Root Cause:** Lack of **{driver_name}** is causing your {target_name} to spike. "
                        f"Prioritize {driver_name} to keep {target_name} under control."
                    )

            smart_insights.append({
                "type": "root_cause", 
                "icon": "🔗", 
                "message": message
            })
            
        energy_drainers = analyze_energy_drainers(df)
        if energy_drainers:
            smart_insights.extend(energy_drainers)
            
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
        "perfect_day": round_floats(perfect_day),
        "gamification": badges,
        "time_series": time_series,
        "population_comparison": round_floats(population_comparison),
        "population_stats": {
            "averages": round_floats(population_avg),
            "std_deviations": round_floats(population_std),
            "users_analyzed": population_user_count
        },
        "smart_insights": smart_insights,
        "weekly_rhythm": weekly_rhythm
    }
    
    return result
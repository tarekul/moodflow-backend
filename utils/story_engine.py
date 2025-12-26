import pandas as pd
import numpy as np
from datetime import time

def generate_data_story(logs):
    if len(logs) < 7:
        return None 

    df = pd.DataFrame(logs)
    
    numeric_cols = ['mood', 'sleep_hours', 'stress', 'productivity', 
                    'screen_time_hours', 'physical_activity_min', 'social_interaction_hours']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'sleep_wake_time' in df.columns:
        def time_to_float(t):
            if pd.isna(t): return np.nan
            if isinstance(t, str):
                parts = t.split(':')
                return int(parts[0]) + int(parts[1])/60
            if isinstance(t, time):
                return t.hour + t.minute/60
            return np.nan
            
        df['wake_hour'] = df['sleep_wake_time'].apply(time_to_float)

    valid_cols = [c for c in numeric_cols if c in df.columns]
    if 'productivity' not in valid_cols:
        return None

    corrs = df[valid_cols].corr()['productivity'].drop('productivity')
    slides = []

    # =========================================================
    # STORY 1: THE DRIVER (Mood vs Sleep)
    # =========================================================
    mood_corr = corrs.get('mood', 0)
    sleep_corr = corrs.get('sleep_hours', 0)
    
    if mood_corr > 0.5 and mood_corr > (sleep_corr + 0.2):
        slides.append({
            "type": "driver",
            "title": "Your Superpower",
            "emoji": "🔮",
            "theme": "purple",
            "headline": "The Emotional Powerhouse",
            "data_highlight": f"Mood Impact: {int(mood_corr*100)}%",
            "narrative": (
                f"Your productivity is fueled by **Mood** ({mood_corr:.2f}), "
                f"not physical energy.\n\n"
                "If you wake up happy, you will crush the day—even on low sleep."
            ),
            "action": "Prioritize joy in your morning routine. Music or a fun hobby matters more than 'waking up early' for you."
        })
    elif sleep_corr > 0.5:
        slides.append({
            "type": "driver",
            "title": "Your Superpower",
            "emoji": "🔋",
            "theme": "green",
            "headline": "The Biological Battery",
            "data_highlight": f"Sleep Impact: {int(sleep_corr*100)}%",
            "narrative": "You are physically driven. Your output scales almost perfectly with your sleep hours.",
            "action": "Protect your 7+ hours of sleep like your career depends on it—because it does."
        })

    # =========================================================
    # STORY 2: THE CHRONOTYPE (Wake Time)
    # =========================================================
    if 'wake_hour' in df.columns:
        clean_wake = df[(df['wake_hour'] > 4) & (df['wake_hour'] < 12)]
        if len(clean_wake) > 5:
            wake_corr = clean_wake['wake_hour'].corr(clean_wake['productivity'])
            
            if wake_corr < -0.4: 
                slides.append({
                    "type": "chronotype",
                    "title": "Timing Insight",
                    "emoji": "🌅",
                    "theme": "orange", 
                    "headline": "The Early Bird",
                    "data_highlight": "Early Riser Advantage",
                    "narrative": (
                        "Your data shows a strong link between **waking up early** and high performance.\n\n"
                        "For every hour you delay waking up, your productivity score tends to drop."
                    ),
                    "action": "Try setting your alarm 30 mins earlier for a week and track the difference."
                })
            elif wake_corr > 0.4:
                slides.append({
                    "type": "chronotype",
                    "title": "Timing Insight",
                    "emoji": "🦉",
                    "theme": "indigo",
                    "headline": "The Night Owl",
                    "data_highlight": "Late Riser Advantage",
                    "narrative": (
                        "You defy the '5 AM Club' myth. Your data suggests you perform better when you sleep in slightly later.\n\n"
                        "Forcing an early start might actually be hurting your output."
                    ),
                    "action": "Don't force mornings. Shift your deep work block to later in the day."
                })

    # =========================================================
    # STORY 3: WORKOUT TIMING (Morning vs Evening)
    # =========================================================
    if 'activity_time' in df.columns and 'physical_activity_min' in df.columns:
        active_days = df[df['physical_activity_min'] > 15]
        
        if len(active_days) > 3:
            timing_stats = active_days.groupby('activity_time')['productivity'].mean()
            
            morning_prod = timing_stats.get('morning', 0)
            evening_prod = timing_stats.get('evening', 0)
            
            if morning_prod > (evening_prod + 1.0) and evening_prod > 0:
                slides.append({
                    "type": "workout_timing",
                    "title": "Optimization",
                    "emoji": "sunrise",
                    "theme": "teal",
                    "headline": "Morning Mover",
                    "data_highlight": "Morning Workouts Win",
                    "narrative": (
                        f"You are significantly more productive on days you exercise in the **Morning** (Avg {morning_prod:.1f}) "
                        f"compared to the Evening (Avg {evening_prod:.1f})."
                    ),
                    "action": "Front-load your exercise. It primes your brain for the rest of the day."
                })
                
    # =========================================================
    # STORY 4: PHYSICAL ACTIVITY (The Kinetic Engine)
    # =========================================================
    activity_corr = corrs.get('physical_activity_min', 0)
    
    if activity_corr > 0.35:
        slides.append({
            "type": "booster",
            "title": "Momentum Builder",
            "emoji": "🏃",
            "theme": "teal",
            "headline": "The Kinetic Engine",
            "data_highlight": f"+{int(activity_corr*100)}% Boost",
            "narrative": (
                f"You think better when you move. There is a strong link ({activity_corr:.2f}) between exercise and your output.\n\n"
                "Your brain needs oxygen to function at peak levels."
            ),
            "action": "Don't skip the gym when you're busy. That is exactly when you need it most."
        })

    # =========================================================
    # STORY 5: SOCIAL INTERACTION (Lone Wolf vs Social Butterfly)
    # =========================================================
    social_corr = corrs.get('social_interaction_hours', 0)
    
    if social_corr > 0.4:
        slides.append({
            "type": "social_pos",
            "title": "Energy Source",
            "emoji": "💬",
            "theme": "yellow",
            "headline": "The Social Butterfly",
            "data_highlight": "Extroverted Pattern",
            "narrative": (
                "Isolation drains you. Your data shows that **social interaction boosts your productivity**.\n\n"
                "Talking to people recharges your mental battery."
            ),
            "action": "Schedule 'collaboration hours' or coffee chats. Don't isolate yourself for too long."
        })
    elif social_corr < -0.4:
        slides.append({
            "type": "social_neg",
            "title": "Focus Style",
            "emoji": "🐺",
            "theme": "slate",
            "headline": "The Lone Wolf",
            "data_highlight": "Deep Work Mode",
            "narrative": (
                "You thrive in solitude. High levels of social interaction tend to **drain your productivity**.\n\n"
                "You pay a high 'context switching' tax when interrupted."
            ),
            "action": "Defend your calendar. Block out 'No Meeting' days to protect your focus."
        })

    # =========================================================
    # STORY 6: CONTEXT (The Office Effect)
    # =========================================================
    if 'tags' in df.columns:
        office_days = df[df['tags'].apply(lambda x: 'office' in x if isinstance(x, list) else False)]
        home_days = df[df['tags'].apply(lambda x: 'wfh' in x if isinstance(x, list) else False)]
        
        if not office_days.empty and not home_days.empty:
            avg_prod_office = office_days['productivity'].mean()
            avg_prod_home = home_days['productivity'].mean()
            
            if avg_prod_home > 0:
                diff = ((avg_prod_office - avg_prod_home) / avg_prod_home) * 100
                
                if diff > 10:
                    slides.append({
                        "type": "context",
                        "title": "Environment Check",
                        "emoji": "🏢",
                        "theme": "blue",
                        "headline": "The Office Effect",
                        "data_highlight": f"+{int(diff)}% Boost",
                        "narrative": (
                            f"You are **{int(diff)}% more productive** when you go to the Office vs Working from Home.\n\n"
                            "The structure triggers your 'Deep Work' mode much better than home."
                        ),
                        "action": "On high-stakes days, commute. Don't risk the distraction of home."
                    })

    # =========================================================
    # STORY 7: THE ANOMALY (Sleep Sweet Spot)
    # =========================================================
    if 'sleep_hours' in df.columns:
        df['sleep_bin'] = pd.cut(df['sleep_hours'], bins=[0, 6, 7, 8, 12], labels=['<6h', '6-7h', '7-8h', '8h+'])
        if not df['sleep_bin'].isnull().all():
            best_sleep = df.groupby('sleep_bin')['productivity'].mean().idxmax()
            
            if best_sleep == '<6h' or best_sleep == '6-7h':
                 slides.append({
                    "type": "anomaly",
                    "title": "The Anomaly",
                    "emoji": "🧬",
                    "theme": "indigo",
                    "headline": "The Short Sleeper",
                    "data_highlight": f"Peak at {best_sleep}",
                    "narrative": (
                        f"Data shows you perform best on **{best_sleep}** of sleep. This is unusual!\n\n"
                        "You might be running on adrenaline (High Momentum), or you simply require less sleep than average."
                    ),
                    "action": "Be careful. Short-term performance on low sleep often leads to long-term burnout."
                })

    return slides
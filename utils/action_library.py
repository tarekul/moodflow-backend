action_library = {
    "Mood": {
        "emoji": "😊",
        "strategies": [
            {
                "title": "Gratitude & Reflection",
                "actions": ["Morning: Write 3 things you're grateful for", "Midday: Take a photo of something beautiful", "Evening: Journal one win from today"],
                "metric_template": "Mood score ≥ {target} for 5 days"
            },
            {
                "title": "Environment & Joy",
                "actions": ["Start day with your favorite upbeat song", "Work from a different spot or tidy desk", "Evening: 30 mins of a hobby (no screens)"],
                "metric_template": "Mood score ≥ {target} for 5 days"
            },
            {
                "title": "Connection Boost",
                "actions": ["Send a nice text to a friend before 10am", "Eat lunch with a colleague or partner", "Evening: Call a family member"],
                "metric_template": "Mood score ≥ {target} for 5 days"
            }
        ]
    },
    "Sleep": {
        "emoji": "😴",
        "strategies": [
            {
                "title": "The Routine Reset",
                "actions": ["Set a 'Reverse Alarm' for 9:30 PM (start winding down)", "Layout clothes for tomorrow", "Bed by 10:30 PM sharp"],
                "metric_template": "Sleep {target} hours on 5 nights"
            },
            {
                "title": "Light & Dark Control",
                "actions": ["Morning: 10 mins sunlight within 30 mins of waking", "Afternoon: No caffeine after 2 PM", "Evening: Dim lights 1 hour before bed"],
                "metric_template": "Sleep {target} hours on 5 nights"
            },
            {
                "title": "Physical Relaxation",
                "actions": ["No heavy meals 3 hours before bed", "Take a warm shower/bath in the evening", "Do 5 mins of box breathing in bed"],
                "metric_template": "Sleep {target} hours on 5 nights"
            }
        ]
    },
    "Stress": {
        "emoji": "🧘",
        "strategies": [
            {
                "title": "Mindfulness Micro-Doses",
                "actions": ["Morning: 5-minute guided meditation", "Midday: Two minutes of deep breathing", "Evening: Do a 'brain dump' list for tomorrow"],
                "metric_template": "Stress score ≤ {target} on 5 days"
            },
            {
                "title": "Physical Release",
                "actions": ["Do 10 jumping jacks when feeling stuck", "Take a 15-min walk without your phone", "Progressive muscle relaxation before dinner"],
                "metric_template": "Stress score ≤ {target} on 5 days"
            },
            {
                "title": "Boundary Setting",
                "actions": ["Say 'no' to one non-essential task", "Turn off non-urgent notifications", "Take a strictly unplugged lunch break"],
                "metric_template": "Stress score ≤ {target} on 5 days"
            }
        ]
    },
    "Physical Activity": {
        "emoji": "🏃",
        "strategies": [
            {
                "title": "The Morning Charge",
                "actions": ["Put on workout clothes immediately upon waking", "Drink glass of water", "20-minute brisk walk or jog"],
                "metric_template": "{target}+ mins activity on 5 days"
            },
            {
                "title": "Deskercise Integration",
                "actions": ["Take the stairs instead of the elevator", "Stand up and stretch every hour", "Walk while taking phone calls"],
                "metric_template": "{target}+ mins activity on 5 days"
            },
            {
                "title": "Evening Wind-Down",
                "actions": ["15-minute yoga flow after work", "Go for a post-dinner stroll", "Do 5 minutes of stretching before bed"],
                "metric_template": "{target}+ mins activity on 5 days"
            }
        ]
    },
    "Screen Time": {
        "emoji": "📱",
        "strategies": [
            {
                "title": "Morning Digital Fast",
                "actions": ["No phone for first 30 mins of day", "Use an old-school alarm clock", "Read a physical book with breakfast"],
                "metric_template": "Screen time < {target} hours on 5 days"
            },
            {
                "title": "Focus Block Method",
                "actions": ["Phone in another room during deep work", "Use 'Grayscale Mode' to make phone boring", "Set app limits for social media"],
                "metric_template": "Screen time < {target} hours on 5 days"
            },
            {
                "title": "The Bedtime Ban",
                "actions": ["Charge phone outside the bedroom", "No screens 1 hour before sleep", "Replace scrolling with reading or audiobooks"],
                "metric_template": "Screen time < {target} hours on 5 days"
            }
        ]
    },
    "Social Interaction": {
        "emoji": "💬",
        "strategies": [
            {
                "title": "Reach Out",
                "actions": ["Text one person you haven't seen in a while", "Schedule a coffee date", "Call a friend during your commute"],
                "metric_template": "Socialize > {target} hour on 4 days"
            },
            {
                "title": "Deepen Connections",
                "actions": ["Ask meaningful questions ('How are you really?')", "Put phone away during conversations", "Plan a group activity for the weekend"],
                "metric_template": "Socialize > {target} hour on 4 days"
            },
            {
                "title": "Community Engagement",
                "actions": ["Work from a coffee shop or co-working space", "Smile/say hi to a stranger", "Join a club or group event"],
                "metric_template": "Socialize > {target} hour on 4 days"
            }
        ]
    },
    "Diet Quality": {
        "emoji": "🍎",
        "strategies": [
            {
                "title": "Hydration Station",
                "actions": ["Drink water immediately upon waking", "Keep a water bottle at your desk", "Drink a glass of water before every meal"],
                "metric_template": "Diet Quality '{target}' on 5 days"
            },
            {
                "title": "Green Machine",
                "actions": ["Add a fruit or veggie to breakfast", "Have a salad or green side with lunch", "Snack on nuts or fruit instead of chips"],
                "metric_template": "Diet Quality '{target}' on 5 days"
            },
            {
                "title": "Mindful Eating",
                "actions": ["Eat without screens/distractions", "Chew slowly and savor food", "Stop eating when 80% full"],
                "metric_template": "Diet Quality '{target}' on 5 days"
            }
        ]
    }
}
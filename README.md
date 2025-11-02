# MoodFlow Backend

This is the backend for the MoodFlow app, a personal productivity analytics platform.

## Setup

1. Clone the repository
2. Install dependencies
3. Run migrations
4. Run the server

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

5. Run migrations

```bash
python3 migrations/run_migrations.py
```

6. Run the server

```bash
python3 main.py
```

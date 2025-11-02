import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

# Database configuration
DB_CONFIG = {
    'dbname': 'moodflow',
    'user': 'tarek',
    'password': '',
    'host': 'localhost',
    'port': 5432
}

# Opens connection to PostgreSQL database
def get_db_connection():
    """
    Create a database connection.
    Returns a connection object.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    return conn

# Context manager for database connections (automatically closes connection)
@contextmanager
def get_db():
    """
    Context manager for database connections.
    Automatically closes connection when done.
    
    Usage:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users")
    """
    conn = get_db_connection()
    try: 
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
        
# Helper to run queries easily
def execute_query(query, params=None, fetch_one=False, fetch_all=False):
    """
    Execute a query and return results.
    
    Args:
        query: SQL query string
        params: Query parameters (tuple)
        fetch_one: Return single row
        fetch_all: Return all rows
    
    Returns:
        Query results or None
    """
    with get_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor) # Returns rows as dictionaries
        cursor.execute(query, params)
        
        if fetch_one:
            return cursor.fetchone()
        elif fetch_all:
            return cursor.fetchall()
        else:
            return None
    
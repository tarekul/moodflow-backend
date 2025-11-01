import psycopg2
from pathlib import Path

# Database configuration
DB_CONFIG = {
    'dbname': 'moodflow',
    'user': 'tarek',
    'password': '',
    'host': 'localhost',
    'port': 5432
}

def get_connection():
    """Connect to PostgreSQL database"""
    return psycopg2.connect(**DB_CONFIG)

def create_database_if_not_exists():
    """Create moodflow database if it doesn't exist"""
    # Connect to default postgres database
    conn = psycopg2.connect(
        dbname='postgres',
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        host=DB_CONFIG['host'],
        port=DB_CONFIG['port']
    )
    conn.autocommit = True
    cursor = conn.cursor()
    
    # Check if database exists
    cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'moodflow'")
    exists = cursor.fetchone()
    
    if not exists:
        cursor.execute("CREATE DATABASE moodflow")
        print("✅ Database 'moodflow' created")
    else:
        print("ℹ️  Database 'moodflow' already exists")
        
    cursor.close()
    conn.close()
    
def get_applied_migrations(cursor):
    """Get list of applied migrations"""
    try:
        cursor.execute("SELECT version FROM schema_migrations")
        return {row[0] for row in cursor.fetchall()}
    except psycopg2.Error:
        # Table doesn't exist yet
        return set()

def apply_migration(cursor, filepath):
    """Apply a single migration file"""
    with open(filepath, 'r') as f:
        sql = f.read()
        
    # Execute the SQL
    cursor.execute(sql)
    
    # Record that this migration was applied
    version = filepath.name
    cursor.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
    
    print(f"✅ Applied migration {version}")
    
def run_migrations():
    """Run all pending migrations"""
    print("🚀 Starting database migrations...")
    
    # Step 1: Create database if needed
    create_database_if_not_exists()
    
    # Step 2: Connect to moodflow database
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Step 3: Ensure the migrations table exists first
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id SERIAL PRIMARY KEY,
                version VARCHAR(255) UNIQUE NOT NULL,
                applied_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # Step 4: Get list of applied migrations
        applied = get_applied_migrations(cursor)
        
        # Step 5: Get list of all migration files
        migrations_dir = Path(__file__).parent
        migration_files = sorted(migrations_dir.glob('*.sql'))
        
        # Step 6: Apply migrations
        pending = [f for f in migration_files if f.name not in applied]
        
        if not pending:
            print("✅ Database is up to date. No migrations needed.")
            return
        
        print(f"📝 Found {len(pending)} pending migration(s)")
        
        for filepath in pending:
            apply_migration(cursor, filepath)
        
        # Commit all changes
        conn.commit()
        print("✅ All migrations completed successfully!")
    except Exception as e:
        conn.rollback()
        print(f"❌ Error applying migrations: {e}")
        raise
    finally:
        cursor.close()
        conn.close()
        
if __name__ == '__main__':
    run_migrations()

    
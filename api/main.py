from fastapi import FastAPI, HTTPException, Depends, status
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import sys
import os
from datetime import date, datetime, time, timedelta
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

from utils.email import send_password_reset_email, generate_reset_token
from utils.database import execute_query, get_db, get_user_by_email
from utils.auth import (hash_password, verify_password, create_access_token, get_current_user_email)
from utils.analysis import analyze_user_data
from utils.story_engine import generate_data_story

def get_current_user(current_user_email: str = Depends(get_current_user_email)):
    """
    Dependency to get full current user object
    """
    user = get_user_by_email(current_user_email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# Create FastAPI app
app = FastAPI(
    title="MoodFlow API",
    description="Personalized productivity analytics API",
    version="1.0.0"
)

origins = [
    "http://localhost:5174",  # local Vite dev server
    "http://192.168.1.8:5174",  # local Vite dev server
    settings.FRONTEND_URL  # deployed Netlify site
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# DATA MODELS (Request/Response Schemas)
# ============================================

class UserResponse(BaseModel):
    id: int
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: str
    tier: str
    created_at: str

class UserCreate(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    password: str

class DailyLogCreate(BaseModel):
    log_date: date
    mood: Optional[float] = None
    productivity: Optional[float] = None
    sleep_hours: float
    sleep_bed_time: Optional[time] = None 
    sleep_wake_time: Optional[time] = None
    stress: Optional[float] = None
    physical_activity_min: Optional[int] = None
    activity_time: Optional[str] = None
    screen_time_hours: Optional[float] = None
    sleep_quality: Optional[str] = None
    social_interaction_hours: Optional[float] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None

class DailyLogUpdate(BaseModel):
    """Model for updating a log - all fields optional"""
    mood: Optional[float] = None
    productivity: Optional[float] = None
    sleep_hours: Optional[float] = None
    sleep_bed_time: Optional[time] = None 
    sleep_wake_time: Optional[time] = None
    stress: Optional[float] = None
    physical_activity_min: Optional[int] = None
    activity_time: Optional[str] = None
    screen_time_hours: Optional[float] = None
    sleep_quality: Optional[str] = None
    social_interaction_hours: Optional[float] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None

class DailyLogResponse(BaseModel):
    id: int
    user_id: int
    log_date: date
    mood: Optional[float] = None
    screen_time_hours: Optional[float] = None
    productivity: Optional[float] = None
    sleep_hours: float
    sleep_bed_time: Optional[time] = None 
    sleep_wake_time: Optional[time] = None
    stress: Optional[float] = None
    physical_activity_min: Optional[int] = None
    activity_time: Optional[str] = None
    sleep_quality: Optional[str] = None
    social_interaction_hours: Optional[float] = None
    notes: Optional[str] = None
    created_at: str
    tags: Optional[List[str]] = None
    
class Token(BaseModel):
    access_token: str
    token_type: str
    

class TokenData(BaseModel):
    email: Optional[str] = None
    
class ForgotPasswordRequest(BaseModel):
    email: EmailStr
    
class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str
    
class GoogleAuthRequest(BaseModel):
    token: str
    
class TagCreate(BaseModel):
    tag_name: str
    
class TagUpdate(BaseModel):
    tag_id: int
    tag_name: str
    
class TagDelete(BaseModel):
    tag_id: int

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
def read_root():
    """
    Root endpoint - health check
    """
    return {
        "message": "Welcome to MoodFlow API!",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/test/{name}")
def test_endpoint(name: str):
    """
    Test endpoint to verify API is working
    """
    return {
        "message": f"Hello, {name}!",
        "status": "success"
    }

# ============================================
# USER ENDPOINTS
# ============================================

@app.get("/users", response_model=List[UserResponse])
def get_all_users():
    """
    Get all users from database
    """
    query = "SELECT id, email, tier, created_at::text FROM users ORDER BY id"
    users = execute_query(query, fetch_all=True)
    
    if not users:
        return []
    
    return users

@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int):
    """
    Get a specific user by ID
    """
    query = "SELECT id, email, tier, created_at::text FROM users WHERE id = %s"
    user = execute_query(query, params=(user_id,), fetch_one=True)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user

@app.post("/users", response_model=UserResponse)
def create_user(user: UserCreate):
    """
    Create a new user with hashed password
    """
    # Check if user already exists
    check_query = "SELECT id FROM users WHERE email = %s"
    existing = execute_query(check_query, params=(user.email,), fetch_one=True)
    
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash password
    hashed_password = hash_password(user.password)
    
    # Insert new user
    insert_query = """
        INSERT INTO users (first_name, last_name, email, password_hash, tier)
        VALUES (%s, %s, %s, %s, 'free')
        RETURNING id, first_name, last_name, email, tier, created_at::text
    """
    new_user = execute_query(
        insert_query,
        params=(user.first_name, user.last_name, user.email, hashed_password),
        fetch_one=True
    )
    
    return new_user

@app.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login endpoint
    
    Takes username (email) and password
    Returns JWT access token if credentials valid
    
    Usage in /docs:
    1. Click "Authorize" button at top
    2. Enter email as username, enter password
    3. Click "Authorize"
    4. Now all protected endpoints will use this token!
    """
    # Find user by email
    query = "SELECT id, email, password_hash FROM users WHERE email = %s"
    user = execute_query(query, params=(form_data.username,), fetch_one=True)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify password
    if not verify_password(form_data.password, user['password_hash']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token = create_access_token(data={"sub": user['email']})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/auth/google")
def google_auth(request: GoogleAuthRequest):
    try:
        # 1. Verify the token with Google
        idinfo = id_token.verify_oauth2_token(
            request.token, 
            google_requests.Request(), 
            settings.GOOGLE_CLIENT_ID
        )

        # 2. Extract user info
        email = idinfo['email']
        first_name = idinfo['given_name']
        last_name = idinfo['family_name']
        google_id = idinfo['sub'] # Unique Google ID

        # 3. Check if user exists in YOUR database
        query = "SELECT * FROM users WHERE email = %s"
        user = execute_query(query, params=(email,), fetch_one=True)

        if not user:
            # 4a. Create new user if they don't exist (Sign Up)
            insert_query = """
                INSERT INTO users (first_name, last_name, email, google_id, created_at) 
                VALUES (%s, %s, %s, %s, NOW()) RETURNING id
            """
            user_id = execute_query(insert_query, params=(first_name, last_name, email, google_id), fetch_one=True)['id']
            
            # Fetch the new user object
            user = {'id': user_id, 'email': email, 'google_id': google_id}
        else:
            # 4b. Optional: Update google_id if linking accounts
            if not user.get('google_id'):
                execute_query("UPDATE users SET google_id = %s WHERE id = %s", (google_id, user['id']))

        # 5. Generate YOUR JWT token (same as normal login)
        access_token = create_access_token(data={"sub": user['email']})
        
        return {
            "access_token": access_token, 
            "token_type": "bearer",
            "user": {
                "id": user['id'],
                "email": user['email'],
                "google_id": user.get('google_id'),
                "first_name": user.get('first_name'),
                "last_name": user.get('last_name')
            }
        }

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Google token")
    except Exception as e:
        print(f"Google Auth Error: {e}")
        raise HTTPException(status_code=500, detail="Authentication failed")

@app.post("/change-password")
def change_password(
    old_password: str,
    new_password: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Change password for current user
    """
    user_id = current_user['id']
    
    # Verify old password
    query = "SELECT password_hash FROM users WHERE id = %s"
    user = execute_query(query, params=(user_id,), fetch_one=True)
    
    if not user or not verify_password(old_password, user['password_hash']):
        raise HTTPException(status_code=401, detail="Invalid old password")
    
    # Hash new password
    hashed_password = hash_password(new_password)
    
    # Update password
    update_query = "UPDATE users SET password_hash = %s WHERE id = %s"
    execute_query(update_query, params=(hashed_password, user_id))
    
    return {"message": "Password changed successfully"}

@app.get("/me", response_model=UserResponse)
def get_current_user(current_user_email: str = Depends(get_current_user_email)):
    """
    Get current logged-in user's information
    
    This endpoint requires authentication (JWT token)
    """
    query = "SELECT id, first_name, last_name, email, tier, created_at::text FROM users WHERE email = %s"
    user = execute_query(query, params=(current_user_email,), fetch_one=True)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user

@app.delete("/me")
def delete_my_account(current_user: dict = Depends(get_current_user)):
    """
    Permanently delete the current user's account and all associated data.
    This action is irreversible.
    """
    user_id = current_user['id']
    
    try:
        delete_logs_query = "DELETE FROM daily_logs WHERE user_id = %s"
        execute_query(delete_logs_query, params=(user_id,))
        
        delete_user_query = "DELETE FROM users WHERE id = %s"
        execute_query(delete_user_query, params=(user_id,))
        
        return {"message": "Account and all data permanently deleted"}
        
    except Exception as e:
        print(f"Error deleting account: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Failed to delete account. Please try again."
        )

@app.get("/users/{user_id}/summary")
def get_user_summary(user_id: int):
    """
    Get summary statistics for a user
    """
    # Check if user exists
    user_check = execute_query(
        "SELECT email FROM users WHERE id = %s",
        params=(user_id,),
        fetch_one=True
    )
    
    if not user_check:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get summary stats
    query = """
        SELECT 
            COUNT(*) as total_logs,
            ROUND(AVG(mood)::numeric, 2) as avg_mood,
            ROUND(AVG(productivity)::numeric, 2) as avg_productivity,
            ROUND(AVG(sleep_hours)::numeric, 2) as avg_sleep,
            ROUND(AVG(stress)::numeric, 2) as avg_stress,
            ROUND(AVG(physical_activity_min)::numeric, 0) as avg_activity,
            MIN(log_date)::text as first_log_date,
            MAX(log_date)::text as last_log_date
        FROM daily_logs
        WHERE user_id = %s
    """
    
    summary = execute_query(query, params=(user_id,), fetch_one=True)
    
    return {
        "user_email": user_check['email'],
        "summary": summary
    }

@app.get("/users/{user_id}/logs", response_model=List[DailyLogResponse])
def get_user_logs(
    user_id: int, 
    start_date: Optional[date] = None, 
    end_date: Optional[date] = None
):
    """
    Get all logs for a specific user
    Optional filters: start_date, end_date
    
    Example: /users/1/logs?start_date=2023-06-01&end_date=2023-06-30
    """
    # First check if user exists
    user_check = execute_query(
        "SELECT id FROM users WHERE id = %s",
        params=(user_id,),
        fetch_one=True
    )
    
    if not user_check:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Build query with optional date filters
    query = """
        SELECT id, user_id, log_date::text, mood, productivity,
               sleep_hours, stress, physical_activity_min, created_at::text
        FROM daily_logs
        WHERE user_id = %s
    """
    
    params = [user_id]
    
    if start_date:
        query += " AND log_date >= %s"
        params.append(start_date)
    
    if end_date:
        query += " AND log_date <= %s"
        params.append(end_date)
    
    query += " ORDER BY log_date DESC"
    
    logs = execute_query(query, params=tuple(params), fetch_all=True)
    
    return logs if logs else []

# ============================================
# PASSWORD RESET ENDPOINTS
# ============================================

@app.post("/forgot-password")
def forgot_password(request: ForgotPasswordRequest):
    """
    Request password reset - sends email with reset token
    
    Step 1 of password reset flow
    """
    email = request.email
    
    # Check if user exists
    query = "SELECT id, email FROM users WHERE email = %s"
    user = execute_query(query, params=(email,), fetch_one=True)
    
    if not user:
        return {
            "message": "If that email exists, a reset link has been sent."
        }
    
    # Generate reset token
    reset_token = generate_reset_token()
    expires_at = datetime.now() + timedelta(hours=1)
    
    # Save reset token in database
    update_query = """
        UPDATE users 
        SET reset_token = %s, reset_token_expires_at = %s
        WHERE id = %s
    """
    execute_query(update_query, params=(reset_token, expires_at, user['id']))
    
    # Send email
    email_sent = send_password_reset_email(email, reset_token)
    
    if not email_sent:
        raise HTTPException(
            status_code=500,
            detail="Failed to send reset email. Please try again."
        )
    
    return {
        "message": "If that email exists, a reset link has been sent."
    }
    
@app.post("/reset-password")
def reset_password(request: ResetPasswordRequest):
    """
    Reset password using token from email
    
    Step 2 of password reset flow
    """
    token = request.token
    new_password = request.new_password
    
    # Find user by reset token
    query = """
        SELECT id, email, reset_token_expires_at 
        FROM users 
        WHERE reset_token = %s
    """
    user = execute_query(query, params=(token,), fetch_one=True)
    
    if not user:
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired reset token"
        )
    
    # Check if token expired
    expires_at = user['reset_token_expires_at']
    if datetime.now() > expires_at:
        raise HTTPException(
            status_code=400,
            detail="Reset token has expired. Please request a new one."
        )
    
    # Hash new password
    hashed_password = hash_password(new_password)
    
    # Update password
    update_query = "UPDATE users SET password_hash = %s WHERE id = %s"
    execute_query(update_query, params=(hashed_password, user['id']))
    
    # Update password and clear reset token
    update_query = """
        UPDATE users 
        SET password_hash = %s, 
            reset_token = NULL, 
            reset_token_expires_at = NULL
        WHERE id = %s
    """
    execute_query(
        update_query,
        params=(hashed_password, user['id'])
    )
    
    return {
        "message": "Password reset successfully. You can now log in with your new password."
    }

# ============================================
# CUSTOM TAGS ENDPOINTS
# ============================================

@app.get("/custom-tags")
def get_custom_tags(current_user: dict = Depends(get_current_user)):
    """
    Get all custom tags for a user
    """
    query = """
        SELECT id, user_id, tag_name, created_at::text
        FROM custom_tags
        WHERE user_id = %s
    """
    tags = execute_query(query, params=(current_user['id'],), fetch_all=True)
    
    return tags if tags else []

@app.post("/custom-tags")
def create_custom_tag(tag: TagCreate, current_user: dict = Depends(get_current_user)):
    """
    Create a new custom tag.
    """
    insert_query = """
        INSERT INTO custom_tags (user_id, tag_name)
        VALUES (%s, %s)
        RETURNING id, user_id, tag_name, created_at::text
    """
    
    new_tag = execute_query(
        insert_query,
        params=(current_user['id'], tag.tag_name),
        fetch_one=True
    )
    
    return new_tag

@app.put("/custom-tags")
def update_custom_tag(tag: TagUpdate, current_user: dict = Depends(get_current_user)):
    """
    Update a custom tag by ID
    """
    update_query = """
        UPDATE custom_tags 
        SET tag_name = %s 
        WHERE id = %s AND user_id = %s
        RETURNING id, user_id, tag_name, created_at::text
    """
    updated_tag = execute_query(
        update_query,
        params=(tag.tag_name, tag.tag_id, current_user['id']),
        fetch_one=True
    )
    
    return updated_tag

@app.delete("/custom-tags")
def delete_custom_tag(tag: TagDelete):
    """
    Delete a custom tag by ID
    """
    delete_query = "DELETE FROM custom_tags WHERE id = %s"
    
    execute_query(delete_query, params=(tag.tag_id,)) 
    
    return {"message": "Custom tag deleted successfully"}

# ============================================
# LOG ENDPOINTS
# ============================================

@app.get("/logs", response_model=List[DailyLogResponse])
def get_all_logs(user_id: Optional[int] = None, limit: int = 100):
    """
    Get daily logs
    Optional: filter by user_id
    """
    if user_id:
        query = """
            SELECT id, user_id, log_date::text, mood, productivity, 
                   sleep_hours, sleep_bed_time, sleep_wake_time, stress, physical_activity_min, activity_time, 
                   tags, sleep_quality, social_interaction_hours, 
                   notes, created_at::text
            FROM daily_logs 
            WHERE user_id = %s 
            ORDER BY log_date DESC 
            LIMIT %s
        """
        logs = execute_query(query, params=(user_id, limit), fetch_all=True)
    else:
        query = """
            SELECT id, user_id, log_date::text, mood, productivity, 
                   sleep_hours, sleep_bed_time, sleep_wake_time, stress, physical_activity_min, activity_time, 
                   tags, sleep_quality, social_interaction_hours, 
                   notes, created_at::text
            FROM daily_logs 
            ORDER BY log_date DESC 
            LIMIT %s
        """
        logs = execute_query(query, params=(limit,), fetch_all=True)
    
    return logs if logs else []

@app.get("/logs/{log_id}", response_model=DailyLogResponse)
def get_log(log_id: int):
    """
    Get a specific log by ID
    """
    query = """
        SELECT id, user_id, log_date::text, mood, productivity, 
               sleep_hours, sleep_bed_time, sleep_wake_time, stress, physical_activity_min, activity_time, 
               tags, sleep_quality, social_interaction_hours, 
               notes, created_at::text
        FROM daily_logs 
        WHERE id = %s
    """
    log = execute_query(query, params=(log_id,), fetch_one=True)
    
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")
    
    return log

@app.post("/logs", response_model=DailyLogResponse)
def create_log(log: DailyLogCreate, current_user: dict = Depends(get_current_user)):
    """
    Create a new daily log
    """
    user_id = current_user['id']
    
    # Check if log already exists for this user/date
    check_query = """
        SELECT id FROM daily_logs 
        WHERE user_id = %s AND log_date = %s
    """
    existing = execute_query(
        check_query, 
        params=(user_id, log.log_date), 
        fetch_one=True
    )
    
    if existing:
        raise HTTPException(
            status_code=400, 
            detail="Log already exists for this date. Use PUT /logs/{log_id} to update."
        )
    
    # Insert new log
    insert_query = """
        INSERT INTO daily_logs (
            user_id, log_date, mood, productivity, sleep_hours, sleep_bed_time, sleep_wake_time, stress,
            physical_activity_min, activity_time, screen_time_hours, sleep_quality,
            social_interaction_hours, notes, tags
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id, user_id, log_date::text, mood, productivity,
                  sleep_hours, sleep_bed_time, sleep_wake_time, stress, physical_activity_min, activity_time, screen_time_hours, tags, created_at::text
    """
    
    new_log = execute_query(
        insert_query,
        params=(
            user_id, log.log_date, log.mood, log.productivity,
            log.sleep_hours, log.sleep_bed_time, log.sleep_wake_time, log.stress, log.physical_activity_min,
            log.activity_time, log.screen_time_hours, log.sleep_quality,
            log.social_interaction_hours, log.notes, log.tags
        ),
        fetch_one=True
    )
    
    return new_log

@app.put("/logs/{log_id}", response_model=DailyLogResponse)
def update_log(
    log_id: int, 
    log: DailyLogUpdate,
    current_user: dict = Depends(get_current_user)
):
    """
    Update an existing log
    Users can only update their own logs
    """
    user_id = current_user['id']
    
    # Check if log exists AND belongs to current user
    check_query = """
        SELECT id, user_id FROM daily_logs 
        WHERE id = %s
    """
    existing_log = execute_query(check_query, params=(log_id,), fetch_one=True)
    
    if not existing_log:
        raise HTTPException(status_code=404, detail="Log not found")
    
    # Verify ownership
    if existing_log['user_id'] != user_id:
        raise HTTPException(
            status_code=403, 
            detail="You don't have permission to update this log"
        )
    
    # Build update query (same as before)
    update_fields = []
    params = []
    
    if log.mood is not None:
        update_fields.append("mood = %s")
        params.append(log.mood)
    if log.productivity is not None:
        update_fields.append("productivity = %s")
        params.append(log.productivity)
    if log.sleep_hours is not None:
        update_fields.append("sleep_hours = %s")
        params.append(log.sleep_hours)
    if log.sleep_bed_time is not None:
        update_fields.append("sleep_bed_time = %s")
        params.append(log.sleep_bed_time)
    if log.sleep_wake_time is not None:
        update_fields.append("sleep_wake_time = %s")
        params.append(log.sleep_wake_time)
    if log.stress is not None:
        update_fields.append("stress = %s")
        params.append(log.stress)
    if log.physical_activity_min is not None:
        update_fields.append("physical_activity_min = %s")
        params.append(log.physical_activity_min)
    if log.activity_time is not None:
        update_fields.append("activity_time = %s")
        params.append(log.activity_time)
    if log.screen_time_hours is not None:
        update_fields.append("screen_time_hours = %s")
        params.append(log.screen_time_hours)
    if log.sleep_quality is not None:
        update_fields.append("sleep_quality = %s")
        params.append(log.sleep_quality)
    if log.social_interaction_hours is not None:
        update_fields.append("social_interaction_hours = %s")
        params.append(log.social_interaction_hours)
    if log.notes is not None:
        update_fields.append("notes = %s")
        params.append(log.notes)
    if log.tags is not None:
        update_fields.append("tags = %s")
        params.append(log.tags)
    
    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    params.append(log_id)
    
    update_query = f"""
        UPDATE daily_logs
        SET {', '.join(update_fields)}
        WHERE id = %s
        RETURNING id, user_id, log_date::text, mood, productivity,
                  sleep_hours, sleep_bed_time, sleep_wake_time, stress, physical_activity_min, activity_time, screen_time_hours, tags, created_at::text
    """
    
    updated_log = execute_query(update_query, params=tuple(params), fetch_one=True)
    
    return updated_log

@app.delete("/logs/{log_id}")
def delete_log(log_id: int, current_user: dict = Depends(get_current_user)):
    """
    Delete a specific log by ID
    """
    user_id = current_user['id']
    
    # Check if log exists
    log_check = execute_query(
        "SELECT id, user_id FROM daily_logs WHERE id = %s",
        params=(log_id,),
        fetch_one=True
    )
    
    if not log_check:
        raise HTTPException(status_code=404, detail="Log not found")
    
    # Verify ownership
    if log_check['user_id'] != user_id:
        raise HTTPException(
            status_code=403,
            detail="You don't have permission to delete this log"
        )
    
    # Delete log
    delete_query = "DELETE FROM daily_logs WHERE id = %s"
    execute_query(delete_query, params=(log_id,))
    
    return {
        "message": "Log deleted successfully",
        "deleted_log_id": log_id
    }
    
@app.get("/my-logs", response_model=List[DailyLogResponse])
def get_my_logs(
    current_user: dict = Depends(get_current_user),
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
):
    """
    Get current user's logs
    Requires authentication
    Optional filters: start_date, end_date
    """
    user_id = current_user['id']
    
    # Build query
    query = """
        SELECT id, user_id, log_date::text, mood, productivity,
               sleep_hours, sleep_bed_time, sleep_wake_time, stress, screen_time_hours,
               physical_activity_min, activity_time, sleep_quality,
               social_interaction_hours, notes, tags,
               created_at::text
        FROM daily_logs
        WHERE user_id = %s
    """
    
    params = [user_id]
    
    if start_date:
        query += " AND log_date >= %s"
        params.append(start_date)
    
    if end_date:
        query += " AND log_date <= %s"
        params.append(end_date)
    
    query += " ORDER BY log_date DESC"
    
    logs = execute_query(query, params=tuple(params), fetch_all=True)
    
    return logs if logs else []

@app.get("/analysis")
def get_analysis(current_user: dict = Depends(get_current_user)):
    """
    Get personalized analysis for current user
    Requires at least 7 days of logged data
    
    Returns:
    - Summary statistics
    - Correlation analysis
    - Boosters & drainers
    - Top recommendation
    - Action plan
    - Time series data for charts
    """
    user_id = current_user['id']
    
    # Fetch all logs for user
    query = """
        SELECT * FROM daily_logs
        WHERE user_id = %s
        ORDER BY log_date
    """

    logs = execute_query(query, params=(user_id,), fetch_all=True)
    
    if not logs:
        raise HTTPException(
            status_code=400,
            detail="No logs found. Please log at least 7 days of data."
        )

    # Run analysis
    analysis_result = analyze_user_data(logs, user_id)
    
    # Check if error (not enough data)
    if "error" in analysis_result:
        raise HTTPException(
            status_code=400,
            detail=analysis_result["error"]
        )
        
    return analysis_result

@app.get("/story")
def get_story(current_user: dict = Depends(get_current_user)):
    user_id = current_user['id']
    
    # Fetch all logs for user
    query = """
        SELECT * FROM daily_logs
        WHERE user_id = %s
        ORDER BY log_date
    """

    logs = execute_query(query, params=(user_id,), fetch_all=True)
    
    if not logs:
        raise HTTPException(
            status_code=400,
            detail="No logs found. Please log at least 7 days of data."
        )
    
    story = generate_data_story(logs)
    
    return story
    

# ============================================
# RUN SERVER (for development)
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
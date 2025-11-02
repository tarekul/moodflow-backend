from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# ============================================
# CONFIGURATION
# ============================================

# Secret key for signing JWT tokens
# In production, use environment variable!
SECRET_KEY = "your-secret-key-keep-this-safe-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# ============================================
# PASSWORD FUNCTIONS
# ============================================

def hash_password(password: str) -> str:
    """
    Hash a plain text password
    
    Example:
        hash_password("mypassword123")
        Returns: "$2b$12$xyz..."
    """
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash
    
    Example:
        verify_password("mypassword123", "$2b$12$xyz...")
        Returns: True if match, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)

# ============================================
# JWT TOKEN FUNCTIONS
# ============================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token
    
    Args:
        data: Dictionary to encode in token (usually {"sub": email})
        expires_delta: How long token is valid (default: 30 days)
    
    Returns:
        JWT token string
    
    Example:
        token = create_access_token({"sub": "alice@example.com"})
        Returns: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt

def decode_access_token(token: str) -> dict:
    """
    Decode and verify a JWT token
    
    Args:
        token: JWT token string
    
    Returns:
        Decoded token data (dictionary)
    
    Raises:
        JWTError if token is invalid or expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# ============================================
# AUTHENTICATION DEPENDENCY
# ============================================

def get_current_user_email(token: str = Depends(oauth2_scheme)) -> str:
    """
    Dependency to get current user's email from JWT token
    
    Usage in endpoint:
        @app.get("/protected")
        def protected_route(email: str = Depends(get_current_user_email)):
            return {"message": f"Hello {email}"}
    
    This automatically:
    1. Extracts token from Authorization header
    2. Verifies token is valid
    3. Returns user's email
    4. Returns 401 error if token invalid
    """
    payload = decode_access_token(token)
    email: str = payload.get("sub")
    
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return email
import os 
from mailjet_rest import Client
import secrets
from config import settings

# Mailjet credentials from environment
# Initialize Mailjet client
mailjet = Client(auth=(settings.MAILJET_API_KEY, settings.MAILJET_SECRET_KEY), version='v3.1')


def generate_reset_token():
    """Generate a secure random token"""
    return secrets.token_urlsafe(32)

def send_password_reset_email(email: str, reset_token: str):
    """
    Send password reset email with reset link
    
    Args:
        email: User's email address
        reset_token: The reset token to include in URL
    
    Returns:
        True if sent successfully, False otherwise
    """
    
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5174")
    reset_link = f"{frontend_url}/reset-password?token={reset_token}"
    
    try:
        data = {
            'Messages': [
                {
                    "From": {
                        "Email": settings.MAILJET_SENDER_EMAIL,
                        "Name": "MoodFlow"
                    },
                    "To": [
                        {
                            "Email": email,
                            "Name": ""
                        }
                    ],
                    "Subject": "Reset Your MoodFlow Password",
                    "HTMLPart": f"""
                        <h2>Reset Your Password</h2>
                        <p>You requested to reset your password for MoodFlow.</p>
                        <p>Click the link below to reset your password. This link will expire in 1 hour.</p>
                        <p><a href="{reset_link}" style="background-color: #6366f1; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block;">Reset Password</a></p>
                        <p>Or copy this link: {reset_link}</p>
                        <p>If you didn't request this, you can safely ignore this email.</p>
                        <hr>
                        <p style="color: #666; font-size: 12px;">MoodFlow - Discover your unique productivity drivers</p>
                    """
                }
            ]
        }
        
        result = mailjet.send.create(data=data)
        if result.status_code == 200:
            print(f"✅ Password reset email sent to {email}")
            return True
        else:
            print(f"❌ Failed to send email. Status: {result.status_code}")
            return False
    
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False
        
        
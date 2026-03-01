from jose import jwt
from datetime import datetime, timedelta
from src.config import settings

def generate_test_token():
    """Generate a valid JWT token for testing purposes."""
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {
        "sub": "test-user-123",
        "exp": expire,
        "name": "Test User",
        "role": "admin"
    }
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

if __name__ == "__main__":
    print(generate_test_token())

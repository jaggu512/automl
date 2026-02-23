import base64
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone

from cryptography.fernet import Fernet
from jose import JWTError, jwt
from passlib.context import CryptContext


ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-in-production")

PWD_CONTEXT = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def hash_password(password):
    return PWD_CONTEXT.hash(password)


def verify_password(password, hashed_password):
    return PWD_CONTEXT.verify(password, hashed_password)


def _fernet_key():
    env_key = os.getenv("ENCRYPTION_KEY")
    if env_key:
        return env_key.encode("utf-8")

    digest = hashlib.sha256(JWT_SECRET_KEY.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def encrypt_payload(payload):
    serialized = json.dumps(payload, default=str).encode("utf-8")
    token = Fernet(_fernet_key()).encrypt(serialized)
    return token.decode("utf-8")


def create_access_token(username):
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"sub": username, "exp": expire}
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token):
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            return None
        return username
    except JWTError:
        return None

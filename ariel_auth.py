import secrets
import hashlib
import time
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass
class UserCredentials:
    username: str
    password_hash: str
    salt: str
    totp_secret: Optional[str]
    role: str
    last_login: float
    failed_attempts: int
    locked: bool = False

class ARIELAuth:
    def __init__(self, logger):
        self.logger = logger
        self.users = {}
        self.credentials_file = "ariel_credentials.json"
        self._load_credentials()
    
    def _load_credentials(self):
        if not os.path.exists(self.credentials_file):
            return
        
        try:
            with open(self.credentials_file, 'r') as f:
                data = json.load(f)
            
            for username, user_data in data.items():
                self.users[username] = UserCredentials(
                    username=username,
                    password_hash=user_data["password_hash"],
                    salt=user_data["salt"],
                    totp_secret=user_data.get("totp_secret"),
                    role=user_data.get("role", "user"),
                    last_login=user_data.get("last_login", time.time()),
                    failed_attempts=user_data.get("failed_attempts", 0),
                    locked=user_data.get("locked", False)
                )
        except Exception as e:
            print(f"Error loading credentials: {e}")
    
    def _save_credentials(self):
        data = {}
        for username, user in self.users.items():
            data[username] = {
                "password_hash": user.password_hash,
                "salt": user.salt,
                "totp_secret": user.totp_secret,
                "role": user.role,
                "last_login": user.last_login,
                "failed_attempts": user.failed_attempts,
                "locked": user.locked
            }
        
        try:
            with open(self.credentials_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving credentials: {e}")
    
    def create_user(self, username, password, role="user"):
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        
        try:
            import pyotp
            totp_secret = pyotp.random_base32()
        except ImportError:
            totp_secret = None
        
        self.users[username] = UserCredentials(
            username=username,
            password_hash=password_hash,
            salt=salt,
            totp_secret=totp_secret,
            role=role,
            last_login=time.time(),
            failed_attempts=0
        )
        
        self._save_credentials()
        self.logger.audit("user_created", "system", f"Created user {username} with role {role}")
        
        return totp_secret

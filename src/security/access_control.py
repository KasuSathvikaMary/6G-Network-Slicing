import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
import jwt
import hashlib

class Role(Enum):
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    SLICE_OWNER = "slice_owner"

@dataclass
class User:
    username: str
    password_hash: str
    role: Role
    owned_slices: List[str]

class AccessController:
    def __init__(self, secret_key: str = "6g_secret_key"):
        self.secret_key = secret_key
        self.users: Dict[str, User] = {}
        
    def add_user(self, username: str, password: str, role: Role, slices: Optional[List[str]] = None):
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        self.users[username] = User(username, password_hash, role, slices or [])
        
    def authenticate(self, username: str, password: str) -> Optional[str]:
        if username not in self.users:
            return None
        
        user = self.users[username]
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        if user.password_hash == password_hash:
            return self.create_token(user)
        return None
        
    def create_token(self, user: User) -> str:
        payload = {
            "sub": user.username,
            "role": user.role.value,
            "slices": user.owned_slices,
            "exp": time.time() + 3600
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
        
    def validate_token(self, token: str) -> Optional[Dict]:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            if payload["exp"] < time.time():
                return None
            return payload
        except:
            return None
            
    def authorize(self, token: str, required_role: Role, slice_id: Optional[str] = None) -> bool:
        payload = self.validate_token(token)
        if not payload:
            return False
            
        user_role = Role(payload["role"])
        
        if user_role == Role.ADMIN:
            return True
            
        if user_role == Role.OPERATOR and required_role in [Role.OPERATOR, Role.VIEWER]:
            return True
            
        if user_role == Role.VIEWER and required_role == Role.VIEWER:
            return True
            
        if user_role == Role.SLICE_OWNER and required_role in [Role.SLICE_OWNER, Role.VIEWER]:
            if slice_id and slice_id in payload.get("slices", []):
                return True
                
        return False

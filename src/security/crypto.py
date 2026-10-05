import os
import hashlib
import hmac
import base64
from typing import Tuple

class CryptoManager:
    def __init__(self):
        self.keys = {}
        self.active_key_id = None
        
    def generate_key(self) -> Tuple[str, bytes]:
        key_id = hashlib.md5(os.urandom(16)).hexdigest()
        key = os.urandom(32)  # 256-bit key
        self.keys[key_id] = key
        self.active_key_id = key_id
        return key_id, key
        
    def rotate_keys(self) -> Tuple[str, bytes]:
        return self.generate_key()
        
    def encrypt(self, data: str, key_id: str = None) -> str:
        # Simplistic XOR encryption simulating AES for demonstration
        if key_id is None:
            key_id = self.active_key_id
            
        if not key_id or key_id not in self.keys:
            self.generate_key()
            key_id = self.active_key_id
            
        key = self.keys[key_id]
        data_bytes = data.encode()
        encrypted = bytearray(len(data_bytes))
        for i in range(len(data_bytes)):
            encrypted[i] = data_bytes[i] ^ key[i % len(key)]
            
        return f"{key_id}:{base64.b64encode(encrypted).decode()}"
        
    def decrypt(self, encrypted_data: str) -> str:
        parts = encrypted_data.split(":", 1)
        if len(parts) != 2:
            raise ValueError("Invalid encrypted data format")
            
        key_id, payload = parts
        if key_id not in self.keys:
            raise ValueError("Key not found")
            
        key = self.keys[key_id]
        encrypted_bytes = base64.b64decode(payload)
        
        decrypted = bytearray(len(encrypted_bytes))
        for i in range(len(encrypted_bytes)):
            decrypted[i] = encrypted_bytes[i] ^ key[i % len(key)]
            
        return decrypted.decode()
        
    def compute_hmac(self, data: str, key_id: str = None) -> str:
        if key_id is None:
            key_id = self.active_key_id
            
        if not key_id or key_id not in self.keys:
            raise ValueError("No active key for HMAC")
            
        key = self.keys[key_id]
        h = hmac.new(key, data.encode(), hashlib.sha256)
        return h.hexdigest()

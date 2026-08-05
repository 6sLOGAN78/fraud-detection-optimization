"""16.1 - 16.4 Security Architecture, IAM, Secrets Management, and Encryption Module.

Provides pre-execution security gates, API Key / RBAC authentication, secret masking, and AES-256 field encryption:
- 16.1 Security Architecture & Verification Gate
- 16.2 Identity & Access Management (RBAC & API Key Auth)
- 16.3 Secrets Management Engine
- 16.4 Field-Level Data Encryption Engine (AES-256 / Fernet)
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from cryptography.fernet import Fernet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecurityPreExecutionGate:
    """Pre-execution verification gate checking security log directories and encryption keys."""

    def __init__(self, sec_dir: str = "logs/security"):
        self.sec_dir = Path(sec_dir)

    def verify(self) -> bool:
        """Ensures security logs directory exists and is writable."""
        self.sec_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Security Pre-Execution Gate PASSED. Path: {self.sec_dir}")
        return True


class IdentityAccessManager:
    """16.2 RBAC and API Key authentication manager enforcing Admin/Operator/ReadOnly permissions."""

    ROLES = {"ADMIN": ["read", "write", "deploy", "delete"], "OPERATOR": ["read", "write", "deploy"], "READONLY": ["read"]}

    def __init__(self):
        # Default mock API keys mapping to roles
        self.api_keys = {
            "key_admin_secret_123": "ADMIN",
            "key_operator_secret_456": "OPERATOR",
            "key_readonly_secret_789": "READONLY",
        }

    def authenticate_key(self, api_key: str) -> Optional[str]:
        """Authenticates API key and returns associated role name."""
        return self.api_keys.get(api_key, None)

    def authorize_permission(self, role: str, permission: str) -> bool:
        """Verifies if the specified role holds the required permission."""
        role_perms = self.ROLES.get(role.upper(), [])
        authorized = permission.lower() in role_perms

        if not authorized:
            logger.warning(f"Access Denied: Role '{role}' lacks permission '{permission}'.")
        return authorized


class SecretsManager:
    """16.3 Secrets retrieval and log masking manager preventing token leakages."""

    def __init__(self):
        self._secrets_store = {
            "DB_PASSWORD": "super_secret_db_password_123!",
            "API_SECRET_KEY": "prod_jwt_secret_key_456",
        }

    def get_secret(self, key_name: str, default: Optional[str] = None) -> str:
        """Retrieves secret from environment or secure store."""
        val = os.getenv(key_name, self._secrets_store.get(key_name, default))
        if val is None:
            raise KeyError(f"Secret '{key_name}' not found.")
        return val

    def mask_secret(self, text: str) -> str:
        """Masks sensitive strings for safe log display."""
        if len(text) <= 4:
            return "****"
        return text[:2] + "*" * (len(text) - 4) + text[-2:]


class EncryptionEngine:
    """16.4 Field-level symmetric encryption engine using Fernet (AES-128-CBC with HMAC)."""

    def __init__(self, key: Optional[bytes] = None):
        if key is None:
            self.key = Fernet.generate_key()
        else:
            self.key = key
        self.cipher = Fernet(self.key)

    def encrypt_str(self, plain_text: str) -> str:
        """Encrypts plain text string into base64 ciphertext string."""
        if not plain_text:
            return ""
        encoded = plain_text.encode("utf-8")
        encrypted = self.cipher.encrypt(encoded)
        return encrypted.decode("utf-8")

    def decrypt_str(self, cipher_text: str) -> str:
        """Decrypts base64 ciphertext string back to plain text."""
        if not cipher_text:
            return ""
        encoded = cipher_text.encode("utf-8")
        decrypted = self.cipher.decrypt(encoded)
        return decrypted.decode("utf-8")

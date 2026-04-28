"""Single-user auth: scrypt-hashed password stored in piper/.auth.json.

First-run defaults are seeded by serve.sh (user=137, pass=137). The user can
change both from the UI via /api/change-password.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # piper/
CRED_FILE = HERE / ".auth.json"
SESSION_SECRET_FILE = HERE / ".session_secret"


def _scrypt(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32
    )


def _write_creds(username: str, password: str) -> None:
    salt = secrets.token_bytes(16)
    CRED_FILE.write_text(json.dumps({
        "username": username,
        "salt": salt.hex(),
        "hash": _scrypt(password, salt).hex(),
    }))
    os.chmod(CRED_FILE, 0o600)


def seed_default(username: str = "137", password: str = "137") -> None:
    if not CRED_FILE.exists():
        _write_creds(username, password)


def load_creds() -> dict | None:
    try:
        return json.loads(CRED_FILE.read_text())
    except FileNotFoundError:
        return None


def verify(username: str, password: str) -> bool:
    creds = load_creds()
    if not creds:
        return False
    if not hmac.compare_digest(username, creds["username"]):
        return False
    salt = bytes.fromhex(creds["salt"])
    want = bytes.fromhex(creds["hash"])
    return hmac.compare_digest(_scrypt(password, salt), want)


def change_credentials(
    current_password: str, new_username: str, new_password: str
) -> bool:
    creds = load_creds()
    if not creds:
        return False
    if not verify(creds["username"], current_password):
        return False
    _write_creds(new_username or creds["username"], new_password or current_password)
    return True


def session_secret() -> str:
    if not SESSION_SECRET_FILE.exists():
        SESSION_SECRET_FILE.write_text(secrets.token_urlsafe(48))
        os.chmod(SESSION_SECRET_FILE, 0o600)
    return SESSION_SECRET_FILE.read_text().strip()

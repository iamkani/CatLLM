from dataclasses import dataclass
import os
import pathlib

@dataclass(frozen=True)
class AuthConfig:
    # Path to YAML user database. Created on first run if missing.
    USERS_DB_PATH: str = os.getenv("USERS_DB_PATH", "data/auth/users.yaml")
    # Session TTL hint (minutes) for UI; Streamlit itself manages sessions.
    SESSION_TTL_MINUTES: int = int(os.getenv("SESSION_TTL_MINUTES", "720"))
    # Minimum password length policy.
    PASSWORD_MIN_LENGTH: int = int(os.getenv("PASSWORD_MIN_LENGTH", "8"))


def ensure_auth_dirs() -> None:
    """Ensure the parent folders for the users DB exist.
    Safe to call multiple times.
    """
    p = pathlib.Path(AuthConfig.USERS_DB_PATH).parent
    p.mkdir(parents=True, exist_ok=True)

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
import os
import yaml
from passlib.hash import pbkdf2_sha256
from .auth_config import AuthConfig, ensure_auth_dirs


class Role(str, Enum):
    ROOT = "Root"
    ADMIN = "Admin"
    USER = "User"


@dataclass
class User:
    username: str
    role: Role
    password_hash: str


def _db_path() -> str:
    return AuthConfig.USERS_DB_PATH


def _atomic_write(path: str, data: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        yaml.safe_dump(data, f, sort_keys=True)
    os.replace(tmp, path)


# --- Password helpers (passlib, pure-Python) ---
def _hash_password(pw: str) -> str:
    # strong defaults: 29000 rounds by default; can set explicitly if desired
    return pbkdf2_sha256.hash(pw)


def _verify_password(pw: str, pw_hash: str) -> bool:
    try:
        return pbkdf2_sha256.verify(pw, pw_hash)
    except Exception:
        return False


# --- DB ---
def _load_db() -> Dict[str, Any]:
    ensure_auth_dirs()
    path = _db_path()
    if not os.path.exists(path):
        # Bootstrap Root from Streamlit Secrets or env
        try:
            import streamlit as st
            _S = getattr(st, "secrets", {})
        except Exception:
            _S = {}
        boot_pw = _S.get("CATLLM_BOOTSTRAP_ROOT_PASSWORD") or os.getenv("CATLLM_BOOTSTRAP_ROOT_PASSWORD")
        if not boot_pw:
            raise RuntimeError(
                "User DB not found and CATLLM_BOOTSTRAP_ROOT_PASSWORD is not set. Set it once to initialize the Root account."
            )
        root_hash = _hash_password(boot_pw)
        db = {
            "version": 1,
            "users": {
                "root": {"role": Role.ROOT.value, "password_hash": root_hash}
            },
        }
        _atomic_write(path, db)
        return db

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("users", {})
    return data


def _save_db(db: Dict[str, Any]) -> None:
    _atomic_write(_db_path(), db)


# --- Public API ---
def get_user(username: str) -> Optional[User]:
    db = _load_db()
    rec = db["users"].get(username)
    if not rec:
        return None
    role = Role(rec.get("role", Role.USER.value))
    return User(username=username, role=role, password_hash=rec.get("password_hash", ""))


def verify_user(username: str, password: str) -> Optional[User]:
    u = get_user(username)
    if not u or not u.password_hash:
        return None
    ok = _verify_password(password, u.password_hash)
    return u if ok else None


def _is_root(actor: User) -> bool:
    return actor.role == Role.ROOT


def _is_admin(actor: User) -> bool:
    return actor.role in (Role.ROOT, Role.ADMIN)


def add_user(actor: User, username: str, password: str, role: Role = Role.USER) -> User:
    if not _is_admin(actor):
        raise PermissionError("Only Admin/Root can add users.")
    if role == Role.ADMIN and not _is_root(actor):
        raise PermissionError("Only Root can create Admin users.")
    if len(password) < int(AuthConfig.PASSWORD_MIN_LENGTH):
        raise ValueError(f"Password must be at least {AuthConfig.PASSWORD_MIN_LENGTH} characters.")

    db = _load_db()
    if username in db["users"]:
        raise ValueError("User already exists.")
    ph = _hash_password(password)
    db["users"][username] = {"role": role.value, "password_hash": ph}
    _save_db(db)
    return User(username=username, role=role, password_hash=ph)


def set_password(actor: User, target_username: str, new_password: str) -> None:
    if len(new_password) < int(AuthConfig.PASSWORD_MIN_LENGTH):
        raise ValueError(f"Password must be at least {AuthConfig.PASSWORD_MIN_LENGTH} characters.")
    db = _load_db()
    if target_username not in db["users"]:
        raise ValueError("User not found.")

    target_role = Role(db["users"][target_username]["role"])
    if actor.username != target_username:
        if _is_root(actor):
            pass
        elif _is_admin(actor):
            if target_role in (Role.ADMIN, Role.ROOT):
                raise PermissionError("Admin cannot reset Admin/Root passwords.")
        else:
            raise PermissionError("User cannot reset other users' passwords.")

    ph = _hash_password(new_password)
    db["users"][target_username]["password_hash"] = ph
    _save_db(db)


def change_role(actor: User, target_username: str, new_role: Role) -> None:
    if not _is_root(actor):
        raise PermissionError("Only Root can change roles.")
    if target_username == "root":
        raise PermissionError("Root role cannot be changed.")
    db = _load_db()
    if target_username not in db["users"]:
        raise ValueError("User not found.")
    db["users"][target_username]["role"] = new_role.value
    _save_db(db)


def list_users(actor: User) -> Dict[str, Dict[str, str]]:
    if not _is_admin(actor):
        raise PermissionError("Only Admin/Root can list users.")
    db = _load_db()
    return {u: {"role": rec["role"]} for u, rec in db["users"].items()}

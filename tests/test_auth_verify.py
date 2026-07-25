"""Tests for app.backend.api.auth_verify — GoTrue HS256 JWT verification."""

import jwt as jose_jwt
import pytest

from app.backend.api import auth_verify


def make_token(sub="user-123", secret="s3cr3t"):
    return jose_jwt.encode({"sub": sub, "email": "a@b.com"}, secret, algorithm="HS256")


def test_verify_valid():
    tok = make_token(secret="s3cr3t")
    data = auth_verify.verify_access_token(tok, secret="s3cr3t")
    assert data["sub"] == "user-123"


def test_verify_invalid_secret_raises():
    tok = make_token(secret="s3cr3t")
    with pytest.raises(ValueError):
        auth_verify.verify_access_token(tok, secret="wrong")


def test_verify_malformed_raises():
    with pytest.raises(ValueError):
        auth_verify.verify_access_token("not.a.jwt", secret="s3cr3t")


def test_missing_secret_raises(monkeypatch):
    from config import Config

    monkeypatch.setattr(Config, "GOTRUE_JWT_SECRET", "")
    with pytest.raises(ValueError):
        auth_verify.verify_access_token(make_token())

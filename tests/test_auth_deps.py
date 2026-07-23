"""Tests for app.backend.api.auth_deps — get_current_user dependency."""

import jwt as jose_jwt
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.backend.api import auth_deps
from config import Config


def _app():
    a = FastAPI()

    @a.get("/me")
    def me(u: auth_deps.User = Depends(auth_deps.get_current_user)):
        return {"id": u.id, "email": u.email}

    return a


def _token(secret="s3cr3t", **claims):
    payload = {"sub": "u9", "email": "x@y.com", **claims}
    return jose_jwt.encode(payload, secret, algorithm="HS256")


def test_missing_token():
    c = TestClient(_app())
    assert c.get("/me").status_code == 401


def test_bearer_token(monkeypatch):
    monkeypatch.setattr(Config, "GOTRUE_JWT_SECRET", "s3cr3t")
    c = TestClient(_app())
    r = c.get("/me", headers={"Authorization": f"Bearer {_token()}"})
    assert r.status_code == 200 and r.json()["id"] == "u9"


def test_cookie_token(monkeypatch):
    monkeypatch.setattr(Config, "GOTRUE_JWT_SECRET", "s3cr3t")
    c = TestClient(_app())
    r = c.get("/me", cookies={"access_token": _token()})
    assert r.status_code == 200 and r.json()["id"] == "u9"


def test_invalid_token_rejected(monkeypatch):
    monkeypatch.setattr(Config, "GOTRUE_JWT_SECRET", "s3cr3t")
    c = TestClient(_app())
    r = c.get("/me", headers={"Authorization": f"Bearer {_token(secret='wrong')}"})
    assert r.status_code == 401


def test_token_missing_sub_rejected(monkeypatch):
    monkeypatch.setattr(Config, "GOTRUE_JWT_SECRET", "s3cr3t")
    token = jose_jwt.encode({"email": "x@y.com"}, "s3cr3t", algorithm="HS256")
    c = TestClient(_app())
    r = c.get("/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401

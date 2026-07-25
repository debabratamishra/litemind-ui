"""Tests for app.backend.api.auth — /api/auth router (fake GoTrue service)."""

import jwt as jose_jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.backend.api import auth
from app.backend.api.auth_service import GoTrueError
from config import Config

SECRET = "s3cr3t"
TOKEN = jose_jwt.encode({"sub": "u1", "email": "a@b.com"}, SECRET, algorithm="HS256")


class FakeAuth:
    def login(self, e, p):
        return {"access_token": TOKEN, "token_type": "bearer", "user": {"id": "u1", "email": e}}

    def register(self, e, p):
        return {"access_token": TOKEN, "token_type": "bearer", "user": {"id": "u1", "email": e}}

    def logout(self, t):
        return True

    def get_user(self, t):
        return {"id": "u1", "email": "a@b.com"}


@pytest.fixture(autouse=True)
def _secret(monkeypatch):
    monkeypatch.setattr(Config, "GOTRUE_JWT_SECRET", SECRET)


def _client(service=None):
    a = FastAPI()
    a.include_router(auth.router)
    a.dependency_overrides[auth.get_auth_service] = lambda: service or FakeAuth()
    return TestClient(a)


def test_register():
    r = _client().post("/api/auth/register", json={"email": "a@b.com", "password": "pw"})
    assert r.status_code == 200
    body = r.json()
    assert body["access_token"] == TOKEN
    assert body["token_type"] == "bearer"
    assert body["user"]["id"] == "u1"
    assert r.cookies.get("access_token") == TOKEN


def test_login_sets_cookie_and_body():
    r = _client().post("/api/auth/login", json={"email": "a@b.com", "password": "pw"})
    assert r.status_code == 200
    assert r.cookies.get("access_token") == TOKEN
    assert r.json()["access_token"] == TOKEN
    assert r.json()["user"]["email"] == "a@b.com"


def test_me_requires_auth():
    r = _client().get("/api/auth/me")
    assert r.status_code == 401


def test_me_with_cookie():
    c = _client()
    c.post("/api/auth/login", json={"email": "a@b.com", "password": "pw"})
    r = c.get("/api/auth/me")
    assert r.status_code == 200 and r.json()["email"] == "a@b.com"


def test_logout_clears_cookie():
    c = _client()
    c.post("/api/auth/login", json={"email": "a@b.com", "password": "pw"})
    r = c.post("/api/auth/logout")
    assert r.status_code == 200
    assert r.json()["status"] == "success"
    assert c.cookies.get("access_token") in (None, "", '""')


class InvalidCredsAuth(FakeAuth):
    def login(self, e, p):
        raise GoTrueError("Invalid login credentials", status_code=400)


class EmailTakenAuth(FakeAuth):
    def register(self, e, p):
        raise GoTrueError("User already registered", status_code=422)


class DownAuth(FakeAuth):
    def login(self, e, p):
        raise GoTrueError("Auth service unavailable: connection refused")


def test_login_invalid_credentials_401():
    r = _client(InvalidCredsAuth()).post("/api/auth/login", json={"email": "a@b.com", "password": "bad"})
    assert r.status_code == 401


def test_register_email_in_use_409():
    r = _client(EmailTakenAuth()).post("/api/auth/register", json={"email": "a@b.com", "password": "pw"})
    assert r.status_code == 409


def test_login_service_unavailable_503():
    r = _client(DownAuth()).post("/api/auth/login", json={"email": "a@b.com", "password": "pw"})
    assert r.status_code == 503

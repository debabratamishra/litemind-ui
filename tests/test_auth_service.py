"""Tests for app.backend.api.auth_service — GoTrue REST client (mocked transport)."""

import httpx
import pytest

from app.backend.api import auth_service


def _mock_handler(request):
    if request.url.path == "/token":
        return httpx.Response(
            200,
            json={
                "access_token": "tok",
                "token_type": "bearer",
                "user": {"id": "u1", "email": "a@b.com"},
            },
        )
    if request.url.path == "/signup":
        return httpx.Response(200, json={"access_token": "tok", "user": {"id": "u1", "email": "a@b.com"}})
    if request.url.path == "/user":
        return httpx.Response(200, json={"id": "u1", "email": "a@b.com"})
    if request.url.path == "/logout":
        return httpx.Response(200, json={})
    return httpx.Response(404)


def _svc(handler=_mock_handler):
    return auth_service.GoTrueAuthService(
        client=httpx.Client(base_url="http://gotrue.test", transport=httpx.MockTransport(handler))
    )


def test_login():
    res = _svc().login("a@b.com", "pw")
    assert res["access_token"] == "tok" and res["user"]["id"] == "u1"


def test_register():
    res = _svc().register("a@b.com", "pw")
    assert res["user"]["id"] == "u1"


def test_get_user():
    assert _svc().get_user("tok")["id"] == "u1"


def test_logout():
    assert _svc().logout("tok") is True


def test_login_error_carries_gotrue_message():
    def handler(request):
        return httpx.Response(400, json={"error_description": "Invalid login credentials"})

    with pytest.raises(auth_service.GoTrueError) as exc:
        _svc(handler).login("a@b.com", "bad")
    assert "Invalid login credentials" in str(exc.value)
    assert exc.value.status_code == 400


def test_register_error_uses_msg_field():
    def handler(request):
        return httpx.Response(422, json={"msg": "User already registered"})

    with pytest.raises(auth_service.GoTrueError) as exc:
        _svc(handler).register("a@b.com", "pw")
    assert "User already registered" in str(exc.value)

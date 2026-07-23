"""Auth router: register / login / logout / me backed by GoTrue.

Sessions are hybrid: an HTTP-only ``access_token`` cookie for browsers plus
the JWT in the response body for CLI/desktop clients.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response, status
from pydantic import BaseModel

from app.backend.api.auth_deps import User, get_current_user
from app.backend.api.auth_service import GoTrueAuthService, GoTrueError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Cookie lifetimes (seconds)
SESSION_COOKIE_MAX_AGE = 60 * 60 * 24  # 1 day
REMEMBER_COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days


def get_auth_service() -> GoTrueAuthService:
    """Factory for the GoTrue service; override in tests via dependency_overrides."""
    return GoTrueAuthService()


class CredentialsRequest(BaseModel):
    email: str
    password: str
    remember: bool = False


def _set_session_cookie(response: Response, token: str, remember: bool = False) -> None:
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=REMEMBER_COOKIE_MAX_AGE if remember else SESSION_COOKIE_MAX_AGE,
    )


def _map_gotrue_error(e: GoTrueError, *, registering: bool = False) -> HTTPException:
    message = str(e)
    if e.status_code is None:
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Auth service unavailable")
    if registering and ("already" in message.lower() or e.status_code in (409, 422)):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already in use")
    if e.status_code in (400, 401, 403):
        return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    if e.status_code >= 500:
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Auth service unavailable")
    return HTTPException(status_code=e.status_code, detail=message)


@router.post("/register")
def register(
    payload: CredentialsRequest,
    response: Response,
    service: GoTrueAuthService = Depends(get_auth_service),
) -> dict[str, Any]:
    try:
        result = service.register(payload.email, payload.password)
    except GoTrueError as e:
        raise _map_gotrue_error(e, registering=True) from e
    token = result.get("access_token")
    if token:
        _set_session_cookie(response, token, payload.remember)
    return {
        "access_token": token,
        "token_type": result.get("token_type", "bearer"),
        "user": result.get("user"),
    }


@router.post("/login")
def login(
    payload: CredentialsRequest,
    response: Response,
    service: GoTrueAuthService = Depends(get_auth_service),
) -> dict[str, Any]:
    try:
        result = service.login(payload.email, payload.password)
    except GoTrueError as e:
        raise _map_gotrue_error(e) from e
    token = result.get("access_token")
    if not token:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Auth service unavailable")
    _set_session_cookie(response, token, payload.remember)
    return {
        "access_token": token,
        "token_type": result.get("token_type", "bearer"),
        "user": result.get("user"),
    }


@router.post("/logout")
def logout(
    response: Response,
    access_token: Optional[str] = Cookie(None),
    service: GoTrueAuthService = Depends(get_auth_service),
) -> dict[str, str]:
    if access_token:
        try:
            service.logout(access_token)
        except GoTrueError as e:
            # Session revocation is best-effort; the cookie is cleared regardless.
            logger.warning("GoTrue logout failed: %s", e)
    response.set_cookie(key="access_token", value="", httponly=True, samesite="lax", max_age=0)
    return {"status": "success"}


@router.get("/me")
def me(user: User = Depends(get_current_user)) -> dict[str, Any]:
    return {"id": user.id, "email": user.email}

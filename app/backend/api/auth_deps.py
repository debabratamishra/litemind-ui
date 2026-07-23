"""FastAPI auth dependencies: current-user resolution from JWT.

Accepts a Bearer token (``Authorization`` header) or the ``access_token``
HTTP-only cookie set by the auth router. CLI/desktop clients use the header;
browsers rely on the cookie.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Cookie, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.backend.api.auth_verify import verify_access_token

security = HTTPBearer(auto_error=False)


class User(BaseModel):
    id: str
    email: Optional[str] = None


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    access_token: Optional[str] = Cookie(None),
) -> User:
    """Resolve the authenticated user from a Bearer token or session cookie."""
    token = credentials.credentials if credentials else access_token
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        claims = verify_access_token(token)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token") from e
    sub = claims.get("sub")
    if not sub:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing subject")
    return User(id=sub, email=claims.get("email"))

"""JWT verification for GoTrue-issued access tokens (HS256)."""

from __future__ import annotations

from jose import JWTError, jwt

from config import Config


def verify_access_token(token: str, secret: str | None = None) -> dict:
    """Verify a GoTrue HS256 JWT and return its claims.

    Args:
        token: The raw JWT string.
        secret: Signing secret; defaults to ``Config.GOTRUE_JWT_SECRET``.

    Raises:
        ValueError: If the secret is missing or the token is invalid/expired.
    """
    secret = secret or Config.GOTRUE_JWT_SECRET
    if not secret:
        raise ValueError("GOTRUE_JWT_SECRET is not configured")
    try:
        return jwt.decode(token, secret, algorithms=["HS256"], options={"verify_aud": False})
    except JWTError as e:
        raise ValueError(f"Invalid token: {e}") from e

"""GoTrue REST API client for email/password authentication.

Talks to a self-hosted Supabase GoTrue instance over HTTP. The ``httpx.Client``
is injectable so tests can use ``httpx.MockTransport`` instead of a live server.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx

from config import Config

DEFAULT_TIMEOUT_SECONDS = 10.0


class GoTrueError(Exception):
    """Raised when GoTrue returns a non-2xx response or is unreachable."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class GoTrueAuthService:
    """Thin synchronous client for GoTrue's REST endpoints."""

    def __init__(self, client: Optional[httpx.Client] = None):
        self._client = client or httpx.Client(
            base_url=Config.GOTRUE_API_URL,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )

    # ── public API ────────────────────────────────────────────────

    def login(self, email: str, password: str) -> dict[str, Any]:
        """Authenticate with email/password. Returns GoTrue token payload."""
        response = self._request(
            "POST",
            "/token",
            params={"grant_type": "password"},
            json={"email": email, "password": password},
        )
        return self._token_payload(response)

    def register(self, email: str, password: str) -> dict[str, Any]:
        """Create a new user. Returns GoTrue signup payload."""
        response = self._request("POST", "/signup", json={"email": email, "password": password})
        return self._token_payload(response)

    def logout(self, token: str) -> bool:
        """Revoke the session associated with ``token``."""
        self._request("POST", "/logout", headers=self._bearer(token))
        return True

    def get_user(self, token: str) -> dict[str, Any]:
        """Fetch the user record for ``token``."""
        response = self._request("GET", "/user", headers=self._bearer(token))
        return response.json()

    # ── internals ─────────────────────────────────────────────────

    @staticmethod
    def _bearer(token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    @staticmethod
    def _token_payload(response: httpx.Response) -> dict[str, Any]:
        data = response.json()
        return {
            "access_token": data.get("access_token"),
            "token_type": data.get("token_type", "bearer"),
            "user": data.get("user") or {},
        }

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as e:
            raise GoTrueError(f"Auth service unavailable: {e}") from e
        if response.is_success:
            return response
        raise GoTrueError(self._error_message(response), status_code=response.status_code)

    @staticmethod
    def _error_message(response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            body = {}
        return (
            body.get("msg")
            or body.get("error_description")
            or body.get("error")
            or f"GoTrue request failed with status {response.status_code}"
        )

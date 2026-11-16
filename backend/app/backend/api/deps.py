"""Shared route dependencies: the seeded system user.

No authentication layer; every route resolves this constant system user.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

SYSTEM_USER_ID = "00000000-0000-0000-0000-000000000001"


class User(BaseModel):
    id: str
    email: Optional[str] = None
    name: Optional[str] = None


def get_system_user() -> User:
    """Resolve the seeded system user for every request."""
    return User(id=SYSTEM_USER_ID)

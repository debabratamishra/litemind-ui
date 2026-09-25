from collections.abc import Iterable

from fastapi.routing import APIRoute

from backend.app.backend.api import health as health_api
from backend.app.backend.api import models as models_api
from backend.app.backend.api import rag as rag_api
from backend.app.backend.api import tts as tts_api
from backend.main import app


def _api_routes(routes: Iterable[object]):
    for route in routes:
        if isinstance(route, APIRoute):
            yield route
        original_router = getattr(route, "original_router", None)
        if original_router is not None:
            yield from _api_routes(original_router.routes)


def _route(path: str) -> APIRoute:
    matches = [route for route in _api_routes(app.routes) if route.path == path]
    assert len(matches) == 1, f"expected one route for {path}, found {len(matches)}"
    return matches[0]


def test_public_api_routes_are_owned_by_feature_routers():
    assert _route("/health").endpoint is health_api.health_check
    assert _route("/api/rag/query").endpoint is rag_api.rag_query
    assert _route("/api/rag/check-duplicates").endpoint is rag_api.duplicate_check_rag
    assert _route("/models").endpoint is models_api.get_available_models
    assert _route("/models/enhanced").endpoint is models_api.get_enhanced_models
    assert _route("/api/tts/synthesize").endpoint is tts_api.synthesize_speech

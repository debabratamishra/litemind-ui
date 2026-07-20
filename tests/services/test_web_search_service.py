"""Unit tests for ``app.services.web_search_service`` (offline).

The only external boundary is SerpAPI over ``httpx.AsyncClient``. The shared
``httpx_mock`` fixture routes every httpx call to a test-supplied handler, so no
real network request is ever made. We cover:

* API-key token validation (missing / placeholder / too short / valid),
* request parameter construction sent to SerpAPI,
* raw-response parsing + result normalization (``format_results``),
* rate-limit (HTTP 429) and timeout error handling.
"""
import httpx
import pytest

from app.services import web_search_service as wss

VALID_KEY = "abcdefghij1234567890"


def test_validate_token_missing(monkeypatch):
    monkeypatch.delenv("SERP_API_KEY", raising=False)
    svc = wss.WebSearchService(api_key=None)
    res = svc.validate_token()
    assert res["valid"] is False
    assert "not set" in res["message"]


def test_validate_token_placeholder():
    svc = wss.WebSearchService(api_key="your-serpapi-key-here")
    res = svc.validate_token()
    assert res["valid"] is False
    assert "placeholder" in res["message"]


def test_validate_token_too_short():
    svc = wss.WebSearchService(api_key="short")
    res = svc.validate_token()
    assert res["valid"] is False
    assert "too short" in res["message"]


def test_validate_token_valid():
    svc = wss.WebSearchService(api_key=VALID_KEY)
    res = svc.validate_token()
    assert res["valid"] is True
    assert "configured" in res["message"]


def test_validate_token_explicit_key_overrides_config():
    svc = wss.WebSearchService(api_key=None)
    res = svc.validate_token(api_key="longenoughkey123")
    assert res["valid"] is True


async def test_search_without_key_raises(httpx_mock, monkeypatch):
    monkeypatch.delenv("SERP_API_KEY", raising=False)
    svc = wss.WebSearchService(api_key=None)
    with pytest.raises(ValueError):
        await svc.search("query")


async def test_search_success_builds_params(httpx_mock):
    captured: dict = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            json={"organic_results": [{"title": "T", "link": "http://t", "snippet": "s"}]},
        )

    httpx_mock(handler)

    svc = wss.WebSearchService(api_key=VALID_KEY)
    result = await svc.search("best python books", num_results=5)

    assert result["organic_results"][0]["title"] == "T"
    # Request parameters constructed for SerpAPI.
    assert captured["params"]["q"] == "best python books"
    assert captured["params"]["api_key"] == VALID_KEY
    assert captured["params"]["engine"] == "google"
    assert captured["params"]["num"] == "5"
    assert captured["params"]["google_domain"] == "google.com"
    assert captured["params"]["device"] == "desktop"
    assert captured["url"].startswith(wss.SERP_API_BASE_URL)


async def test_search_rate_limit_raises(httpx_mock):
    def handler(request):
        return httpx.Response(429, json={"error": "rate limited"})

    httpx_mock(handler)

    svc = wss.WebSearchService(api_key=VALID_KEY)
    with pytest.raises(httpx.HTTPStatusError):
        await svc.search("query")


async def test_search_timeout_raises(httpx_mock):
    def handler(request):
        raise httpx.TimeoutException("timeout")

    httpx_mock(handler)

    svc = wss.WebSearchService(api_key=VALID_KEY)
    with pytest.raises(httpx.TimeoutException):
        await svc.search("query")


def test_format_results():
    svc = wss.WebSearchService(api_key=VALID_KEY)
    raw = {
        "organic_results": [
            {"title": "A", "link": "http://a", "snippet": "sa"},
            {"title": "B", "link": "http://b", "snippet": "sb"},
        ]
    }
    formatted = svc.format_results(raw)
    assert isinstance(formatted, list)
    assert len(formatted) == 2
    assert formatted[0]["position"] == 1
    assert formatted[0]["title"] == "A"
    assert formatted[0]["link"] == "http://a"
    assert formatted[0]["snippet"] == "sa"
    assert formatted[1]["position"] == 2


def test_format_results_empty():
    svc = wss.WebSearchService(api_key=VALID_KEY)
    assert svc.format_results({}) == []
    assert svc.format_results({"organic_results": []}) == []


def test_format_results_defaults_when_missing_fields():
    svc = wss.WebSearchService(api_key=VALID_KEY)
    formatted = svc.format_results({"organic_results": [{"link": "http://a"}]})
    assert formatted[0]["title"] == "No title"
    assert formatted[0]["snippet"] == "No description available"
    assert formatted[0]["position"] == 1

"""Tests for ``app.backend.core.ollama_models`` (offline).

``build_enhanced_model_payload`` performs a single ``GET {ollama_url}/api/tags`` via
``httpx.AsyncClient``. We patch ``httpx.AsyncClient`` on the module so no real network
request is made, and assert the function returns the expected ``local_models`` /
``cloud_models`` payload shape.
"""
from unittest.mock import AsyncMock, MagicMock, patch

from app.backend.core import ollama_models


async def test_build_enhanced_model_payload():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "models": [
            {
                "name": "llama3:latest",
                "details": {"parameter_size": "8B", "family": "llama"},
            }
        ]
    }

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = False

    with patch(
        "app.backend.core.ollama_models.httpx.AsyncClient", return_value=mock_client
    ):
        payload = await ollama_models.build_enhanced_model_payload(
            "http://localhost:11434"
        )

    assert isinstance(payload, dict)
    assert "local_models" in payload
    assert "cloud_models" in payload

    # Local model surfaced with metadata and is_local flag.
    assert len(payload["local_models"]) == 1
    local = payload["local_models"][0]
    assert local["name"] == "llama3:latest"
    assert local["parameter_size"] == "8B"
    assert local["family"] == "llama"
    assert local["is_local"] is True

    # Curated cloud catalog is returned (none overlap the local model here).
    assert isinstance(payload["cloud_models"], list)
    assert len(payload["cloud_models"]) == len(ollama_models.OLLAMA_CLOUD_CATALOG)
    assert all(m["is_local"] is False for m in payload["cloud_models"])


async def test_build_enhanced_model_payload_filters_local_cloud_dups():
    # When a cloud-catalog model is also installed locally, it must be omitted
    # from cloud_models.
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "models": [
            {
                "name": ollama_models.OLLAMA_CLOUD_CATALOG[0]["name"],
                "details": {},
            }
        ]
    }

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = False

    with patch(
        "app.backend.core.ollama_models.httpx.AsyncClient", return_value=mock_client
    ):
        payload = await ollama_models.build_enhanced_model_payload(
            "http://localhost:11434"
        )

    local_names = {m["name"] for m in payload["local_models"]}
    cloud_names = {m["name"] for m in payload["cloud_models"]}
    assert ollama_models.OLLAMA_CLOUD_CATALOG[0]["name"] in local_names
    assert ollama_models.OLLAMA_CLOUD_CATALOG[0]["name"] not in cloud_names

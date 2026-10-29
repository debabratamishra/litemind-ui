"""Web search service using SerpAPI for retrieving search results.

This service provides a wrapper around SerpAPI to execute web searches
and format results for the web-search orchestration pipeline.
"""
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

logger = logging.getLogger(__name__)

# SerpAPI configuration
SERP_API_BASE_URL = "https://serpapi.com/search"
SERP_API_TIMEOUT = httpx.Timeout(30.0)  # 30 seconds for all operations
DEFAULT_NUM_RESULTS = 10


class WebSearchService:
    """Service for interacting with SerpAPI to perform web searches."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the web search service.

        Args:
            api_key: Optional SerpAPI key override. When provided (e.g. a
                user-supplied client key) it takes precedence over the
                ``SERP_API_KEY`` environment variable.
        """
        override = (api_key or "").strip()
        self.api_key = override or os.getenv("SERP_API_KEY")
        self.base_url = SERP_API_BASE_URL
        self.timeout = SERP_API_TIMEOUT

        if not self.api_key or self.api_key == "your-serpapi-key-here":
            logger.warning("SERP_API_KEY not configured or using placeholder value")

    def validate_token(self, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Validate the presence and basic format of the SerpAPI token.

        Args:
            api_key: Optional key to validate instead of the configured one
                (e.g. a user-supplied client key).

        Returns:
            dict: Status dictionary with 'valid' (bool) and 'message' (str) keys
        """
        key = (api_key or "").strip() or self.api_key
        if not key:
            return {
                "valid": False,
                "message": "SERP_API_KEY environment variable is not set"
            }

        if key == "your-serpapi-key-here":
            return {
                "valid": False,
                "message": "SERP_API_KEY is using placeholder value. Please configure a valid API key."
            }

        # Basic format check - SerpAPI keys are typically alphanumeric
        if len(key) < 10:
            return {
                "valid": False,
                "message": "SERP_API_KEY appears to be invalid (too short)"
            }

        return {
            "valid": True,
            "message": "SERP_API_KEY is configured"
        }

    async def search(self, query: str, num_results: int = DEFAULT_NUM_RESULTS) -> Dict:
        """Execute a web search query using SerpAPI.

        Args:
            query: The search query string
            num_results: Maximum number of results to retrieve (default: 10)

        Returns:
            dict: Raw search results from SerpAPI

        Raises:
            ValueError: If API key is not configured
            httpx.HTTPError: If the API request fails
            httpx.TimeoutException: If the request times out
        """
        # Validate token before making request
        validation = self.validate_token()
        if not validation["valid"]:
            raise ValueError(f"Invalid API key: {validation['message']}")

        # Prepare request parameters
        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "google",
            "num": num_results,
            "google_domain": "google.com",
            "device": "desktop"
        }

        try:
            logger.info(f"Executing SerpAPI search for query: '{query}' (max {num_results} results)")

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()

                results = response.json()
                logger.info("SerpAPI search completed successfully")

                return results

        except httpx.TimeoutException as e:
            logger.error(f"SerpAPI request timed out: {e}")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"SerpAPI HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"SerpAPI request error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during SerpAPI search: {e}")
            raise

    def format_results(self, raw_results: Dict) -> List[Dict[str, Any]]:
        """Format raw SerpAPI results into a structured list for agent consumption.

        Args:
            raw_results: Raw JSON response from SerpAPI

        Returns:
            list: List of formatted search result dictionaries with keys:
                - title: Result title
                - link: Result URL
                - snippet: Result description/snippet
                - position: Result position in search results
        """
        formatted_results = []

        # Extract organic search results
        organic_results = raw_results.get("organic_results", [])

        if not organic_results:
            logger.warning("No organic results found in SerpAPI response")
            return formatted_results

        for idx, result in enumerate(organic_results, start=1):
            formatted_result = {
                "position": idx,
                "title": result.get("title", "No title"),
                "link": result.get("link", ""),
                "snippet": result.get("snippet", "No description available")
            }
            formatted_results.append(formatted_result)

        logger.info(f"Formatted {len(formatted_results)} search results")
        return formatted_results

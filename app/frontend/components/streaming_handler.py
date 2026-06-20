"""
Streaming handlers for chat and RAG responses with conversation memory support.
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional

import requests

from ..services.chat_service import chat_service
from ..services.rag_service import rag_service
from .text_renderer import StreamingRenderer, web_search_renderer

logger = logging.getLogger(__name__)


class StreamingHandler:
    """Handles streaming responses from various backends with conversation memory."""

    @staticmethod
    def _decode_stream_line(line: str) -> str:
        """Decode a streamed line, supporting both raw text and SSE JSON payloads."""
        if line.startswith("data:"):
            payload = line[5:].lstrip()
            if payload.strip() in ("[DONE]", ""):
                return ""

            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                # Backward compatibility for plain-text SSE payloads.
                return payload

            if isinstance(parsed, dict):
                if parsed.get("error"):
                    return f"Error: {parsed['error']}"
                return str(parsed.get("chunk", ""))

            if isinstance(parsed, str):
                return parsed

            return str(parsed)

        return line

    def stream_chat_response(
        self,
        message: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        backend: str = "ollama",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        placeholder: Optional[Any] = None,
        use_fastapi: bool = True,
        tts_callback: Optional[Callable[[str], None]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        conversation_summary: Optional[str] = None,
        session_id: Optional[str] = None,
        is_voice_mode: bool = False,
        enable_generative_ui: bool = False,
    ) -> Optional[str]:
        """Stream a chat response with reasoning segregation and conversation memory.

        Args:
            message: User message
            model: Model name
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            frequency_penalty: Penalize frequent tokens (-2.0 to 2.0)
            repetition_penalty: Penalize repeated tokens (0.0 to 2.0)
            backend: Backend to use (ollama)
            placeholder: Streamlit placeholder for UI updates
            use_fastapi: Whether to use FastAPI backend
            tts_callback: Optional callback to receive text chunks for TTS synthesis.
                         Called with each text chunk as it arrives for real-time TTS.
            conversation_history: Previous messages in the conversation
            conversation_summary: Summary of earlier messages
            session_id: Session identifier for memory tracking
            is_voice_mode: Whether this is voice mode (uses conversational agent)
            enable_generative_ui: Whether to instruct the model to emit ui:* blocks
        """

        try:
            if not use_fastapi:
                logger.error("Chat requires the FastAPI backend, but it is unavailable.")
                if placeholder:
                    placeholder.error("❌ FastAPI backend is required for chat. Please start the backend server.")
                return None

            # FastAPI streaming with memory
            response = chat_service.stream_fastapi_chat(
                message=message,
                model=model,
                backend=backend,
                api_base=api_base,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                repetition_penalty=repetition_penalty,
                conversation_history=conversation_history,
                conversation_summary=conversation_summary,
                session_id=session_id,
                is_voice_mode=is_voice_mode,
                enable_generative_ui=enable_generative_ui,
            )

            return self._process_streaming_response(response, placeholder, tts_callback=tts_callback)

        except requests.Timeout:
            logger.error("Chat API timed out while streaming.")
            if placeholder:
                placeholder.error("❌ Chat API timed out while streaming.")
        except requests.RequestException as exc:
            logger.error(f"Chat API Error: {exc}")
            if placeholder:
                placeholder.error(f"❌ Chat API Error: {exc}")

        return None

    def stream_rag_response(
        self,
        query: str,
        messages: list,
        model: str,
        system_prompt: str,
        n_results: int = 3,
        use_multi_agent: bool = False,
        use_hybrid_search: bool = False,
        backend: str = "ollama",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        placeholder: Optional[Any] = None,
        tts_callback: Optional[Callable[[str], None]] = None,
        conversation_summary: Optional[str] = None,
        session_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        is_voice_mode: bool = False,
    ) -> Optional[str]:
        """Stream a RAG response with reasoning segregation and conversation memory.

        Args:
            query: User query
            messages: Chat history
            model: Model name
            system_prompt: System prompt for RAG
            n_results: Number of results to retrieve
            use_multi_agent: Whether to use multi-agent processing
            use_hybrid_search: Whether to use hybrid search
            backend: Backend to use (ollama)
            placeholder: Streamlit placeholder for UI updates
            tts_callback: Optional callback to receive text chunks for TTS synthesis
            conversation_summary: Summary of earlier messages
            session_id: Session identifier for memory tracking
            temperature: Temperature for LLM response generation
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            frequency_penalty: Penalize frequent tokens (-2.0 to 2.0)
            repetition_penalty: Penalize repeated tokens (0.0 to 2.0)
            is_voice_mode: Whether this is voice mode (uses conversational agent)
        """

        try:
            response = rag_service.stream_rag_query(
                query=query,
                messages=messages,
                model=model,
                system_prompt=system_prompt,
                n_results=n_results,
                use_multi_agent=use_multi_agent,
                use_hybrid_search=use_hybrid_search,
                backend=backend,
                api_base=api_base,
                api_key=api_key,
                conversation_summary=conversation_summary,
                session_id=session_id,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                repetition_penalty=repetition_penalty,
                is_voice_mode=is_voice_mode,
            )

            # Process the streaming response and capture citations
            return self._process_streaming_response_with_citations(response, placeholder, tts_callback=tts_callback)

        except requests.Timeout:
            logger.error("RAG API timed out while streaming.")
            if placeholder:
                placeholder.error("❌ RAG API timed out while streaming. Try narrowing the query.")
        except requests.RequestException as exc:
            logger.error(f"RAG API Error: {exc}")
            if placeholder:
                placeholder.error(f"❌ RAG API Error: {exc}")

        return None

    def stream_web_search_response(
        self,
        message: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        backend: str = "ollama",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        placeholder: Optional[Any] = None,
        use_fastapi: bool = True,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        conversation_summary: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        """Stream a web search response with raw LLM output (no formatting) and memory."""

        try:
            if not use_fastapi:
                logger.error("Web search requires the FastAPI backend, but it is unavailable.")
                if placeholder:
                    placeholder.error("❌ FastAPI backend is required for web search. Please start the backend server.")
                return None

            # FastAPI web search streaming with memory
            response = chat_service.stream_web_search_chat(
                message=message,
                model=model,
                backend=backend,
                api_base=api_base,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                repetition_penalty=repetition_penalty,
                conversation_history=conversation_history,
                conversation_summary=conversation_summary,
                session_id=session_id,
            )

            # Use raw text processing for web search (no formatting)
            return self._process_streaming_response_raw(response, placeholder)

        except requests.Timeout:
            logger.error("Web search API timed out while streaming.")
            if placeholder:
                placeholder.error("❌ Web search API timed out. Falling back to standard chat.")
        except requests.RequestException as exc:
            logger.error(f"Web search API Error: {exc}")
            if placeholder:
                placeholder.error(f"❌ Web search API Error: {exc}")

        return None

    def _process_streaming_response(
        self,
        response: requests.Response,
        placeholder: Optional[Any],
        tts_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Process streaming response and handle reasoning segregation.

        Args:
            response: Streaming HTTP response
            placeholder: Streamlit placeholder for UI updates
            tts_callback: Optional callback for TTS chunks (called for each text chunk)
        """
        buf = ""
        segregator = StreamingRenderer(placeholder) if placeholder is not None else None

        for line in response.iter_lines(decode_unicode=True, chunk_size=1):
            if line is None:
                continue

            chunk = self._decode_stream_line(line)
            if chunk == "":
                continue

            buf += chunk
            if segregator is not None:
                segregator.feed(chunk)

            # Call TTS callback with the text chunk for streaming synthesis
            if tts_callback is not None:
                try:
                    tts_callback(chunk)
                except Exception as e:
                    logger.debug(f"TTS callback error: {e}")

        return buf

    def _process_streaming_response_with_citations(
        self,
        response: requests.Response,
        placeholder: Optional[Any],
        tts_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Process streaming response and extract citations metadata.

        The first line of the stream may be a JSON object with citations metadata.
        All subsequent lines are text chunks to be concatenated.

        Returns:
            str: The complete text response (citations are stored in session state).
        """
        buf = ""
        citations_found = False
        segregator = StreamingRenderer(placeholder) if placeholder is not None else None

        for line in response.iter_lines(decode_unicode=True, chunk_size=1):
            if line is None:
                continue

            # Ensure line is a string (iter_lines can return bytes in some cases)
            line_str = line.decode("utf-8") if isinstance(line, bytes) else line

            # Check if this is the citations metadata line
            if not citations_found and line_str.startswith("data:"):
                payload = line_str[5:].lstrip()
                try:
                    parsed = json.loads(payload)
                    if isinstance(parsed, dict) and "citations" in parsed:
                        # Store citations in session state for retrieval
                        import streamlit as st
                        st.session_state["rag_citations"] = parsed["citations"]
                        citations_found = True
                        continue
                except (json.JSONDecodeError, ImportError):
                    # Not a valid JSON line, treat as regular text
                    pass

            chunk = self._decode_stream_line(line_str)
            if chunk == "":
                continue

            buf += chunk
            if segregator is not None:
                segregator.feed(chunk)

            # Call TTS callback with the text chunk for streaming synthesis
            if tts_callback is not None:
                try:
                    tts_callback(chunk)
                except Exception as e:
                    logger.debug(f"TTS callback error: {e}")

        return buf

    def _process_streaming_response_raw(self, response: requests.Response, placeholder: Optional[Any]) -> str:
        """Process streaming response for web search with proper formatting."""
        # Reset the web search renderer for new response
        web_search_renderer.reset()

        for line in response.iter_lines(decode_unicode=True, chunk_size=1):
            if line is None:
                continue

            chunk = self._decode_stream_line(line)
            if chunk == "":
                continue

            # Use the web search renderer for streaming
            if placeholder is not None:
                web_search_renderer.render_streaming(chunk, placeholder=placeholder)
            else:
                web_search_renderer.render_streaming(chunk, placeholder=None)

        # Return the final formatted text
        return web_search_renderer.get_final_text()


# Global streaming handler
streaming_handler = StreamingHandler()

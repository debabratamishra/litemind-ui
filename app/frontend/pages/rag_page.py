"""
RAG page implementation with conversation memory support.
"""

import logging
from typing import Any, Dict

import streamlit as st

from ..components.conversation_sidebar import get_rag_sidebar
from ..components.shared_ui import (
    create_simple_summary,
    get_backend_request_config,
    get_default_model_for_backend,
    get_generation_config_from_session,
    get_hosted_backend_config,
    is_hosted_backend,
    render_backend_selector,
    render_generation_settings,
    render_memory_config,
    render_memory_indicator,
    render_reasoning_config,
    validate_backend_setup,
)
from ..components.streaming_handler import streaming_handler
from ..components.text_renderer import render_llm_text
from ..components.voice_input import get_voice_input
from ..config import DEFAULT_RAG_SYSTEM_PROMPT, SUPPORTED_EXTENSIONS
from ..services.backend_service import backend_service
from ..services.rag_service import rag_service
from ..utils.memory_manager import RAGMemoryManager

logger = logging.getLogger(__name__)

EMBEDDING_PROVIDER_OPTIONS = {
    "Ollama": "ollama",
    "HuggingFace": "huggingface",
    "OpenRouter": "openrouter",
    "Nvidia NIM": "nvidia_nim",
}


def _is_embedding_model(name: str) -> bool:
    """Return True for models that are embedding/reranking only (not usable for chat)."""
    name_lower = name.lower()
    return any(kw in name_lower for kw in ("embed", "rerank", "bge-", "e5-", "minilm", "gte-"))


def _normalize_embedding_provider(provider: str | None, embedding_backend: str | None = None) -> str:
    """Normalize legacy and current embedding provider identifiers."""
    normalized = (provider or "ollama").strip().lower().replace(" ", "_")

    if normalized == "litellm":
        return _normalize_embedding_provider(embedding_backend or "ollama")

    aliases = {
        "ollama": "ollama",
        "huggingface": "huggingface",
        "openrouter": "openrouter",
        "nvidia_nim": "nvidia_nim",
    }
    return aliases.get(normalized, "ollama")


class RAGPage:
    """Handles the RAG interface and document processing with conversation memory."""

    def __init__(self):
        self.backend_available = st.session_state.get("backend_available", False)
        self.memory_manager = RAGMemoryManager()
        self.conversation_sidebar = get_rag_sidebar()
        self._initialize_session_state()

    def render(self):
        """Render the complete RAG page."""
        if not self.backend_available:
            st.info("Enhanced RAG functionality requires the FastAPI backend. Please start the backend server.")
            return

        # Check if realtime voice mode is active
        realtime_active = st.session_state.get("realtime_voice_mode_rag", False)

        if not realtime_active:
            st.title("RAG Interface")

            self._initialize_session_state()

            # Display memory indicator if enabled
            if st.session_state.get("rag_memory_enabled", True):
                stats = self.memory_manager.get_stats()
                if stats.total_messages > 0:
                    self._render_memory_indicator(stats)

            self._render_system_prompt_config()

            # Only render main content sections, configuration is in sidebar
            self._render_upload_section()

        self._render_query_section()

    def _initialize_session_state(self):
        """Initialize RAG-specific session state."""
        st.session_state.setdefault("config_saved", False)
        st.session_state.setdefault("config_message", "")
        st.session_state.setdefault("rag_messages", [])
        st.session_state.setdefault("show_reset_confirm", False)

        # Initialize RAG system prompt
        st.session_state.setdefault("rag_system_prompt", DEFAULT_RAG_SYSTEM_PROMPT)

        # Initialize default RAG configuration
        st.session_state.setdefault(
            "rag_config",
            {
                "provider": "ollama",
                "embedding_model": "snowflake-arctic-embed2:latest",
                "chunk_size": 500,
                "n_results": 3,
                "use_multi_agent": False,
                "use_hybrid_search": False,
            },
        )

        # Initialize temperature for RAG
        st.session_state.setdefault("rag_temperature", 0.7)

        # Initialize max_tokens for RAG
        st.session_state.setdefault("rag_max_tokens", 2048)

        # Initialize memory-related session state
        st.session_state.setdefault("rag_memory_enabled", True)

        # Initialize conversation history enabled state
        st.session_state.setdefault("rag_history_enabled", True)

        st.session_state.setdefault(
            "selected_openrouter_rag_model",
            get_default_model_for_backend("openrouter"),
        )
        st.session_state.setdefault(
            "selected_nvidia_nim_rag_model",
            get_default_model_for_backend("nvidia_nim"),
        )

    def _render_memory_indicator(self, stats):
        """Render a visual memory usage indicator."""
        render_memory_indicator(stats)

    def _render_system_prompt_config(self):
        """Render system prompt configuration in main page."""
        with st.expander("📝 System Prompt Configuration", expanded=False):
            st.markdown("**Customize the system prompt for RAG responses:**")

            col1, col2 = st.columns([4, 1])

            with col1:
                rag_system_prompt = st.text_area(
                    "System Prompt:",
                    value=st.session_state.rag_system_prompt,
                    height=120,
                    help="Customize the system prompt for RAG responses",
                    key="rag_system_prompt_main",
                )

            with col2:
                st.markdown("<br>", unsafe_allow_html=True)  # Spacing
                if st.button("🔄 Reset to Default", help="Reset to default system prompt"):
                    st.session_state.rag_system_prompt = DEFAULT_RAG_SYSTEM_PROMPT
                    st.rerun()

            # Update session state
            st.session_state.rag_system_prompt = rag_system_prompt

    def _render_upload_section(self):
        """Render document upload section."""
        st.subheader("📁 Upload Documents")

        # Show current status
        self._display_rag_status()

        if not st.session_state.config_saved:
            st.warning("⚠️ Please save your configuration before uploading documents.")

        # File uploader (hide accepted types in UI by omitting `type`)
        uploaded_files = st.file_uploader(
            label="Upload Documents",
            label_visibility="collapsed",
            accept_multiple_files=True,
            help="Upload CSVs, images, or other documents for enhanced processing",
            disabled=not st.session_state.config_saved,
            key="rag_uploader",
        )

        # Upload and process files
        if uploaded_files and st.button("Upload & Process", type="primary", disabled=not st.session_state.config_saved):
            self._process_uploaded_files(uploaded_files)

    def _display_rag_status(self):
        """Display current RAG system status."""
        rag_status = rag_service.get_status()
        if rag_status:
            if rag_status["status"] == "ready":
                if rag_status.get("uploaded_files", 0) > 0:
                    st.info(
                        f"Current Knowledge Base: {rag_status['uploaded_files']} files, "
                        f"{rag_status['indexed_chunks']} chunks indexed"
                    )
                else:
                    st.info("Knowledge base is empty. Upload documents to get started.")
            else:
                st.warning(f"⚠️ System Status: {rag_status['status']}")
        else:
            st.error("❌ Cannot connect to RAG backend")

    def _process_uploaded_files(self, uploaded_files):
        """Process uploaded files with duplicate checking."""
        # Filter unsupported file types since types are hidden in the uploader UI
        allowed_suffixes = tuple(f".{ext.lower()}" for ext in SUPPORTED_EXTENSIONS)
        candidate_files = uploaded_files or []
        filtered_files = [uf for uf in candidate_files if uf.name.lower().endswith(allowed_suffixes)]
        if not filtered_files:
            st.warning("No compatible files to upload. Please choose different files.")
            return

        # Check for duplicates
        with st.spinner("Checking for duplicate files..."):
            duplicate_results = rag_service.check_file_duplicates(filtered_files)

        if not duplicate_results or "results" not in duplicate_results:
            st.error("❌ Failed to check for duplicates. Please try again.")
            return

        # Separate duplicates and new files
        dup_flags = {r.get("filename"): r.get("is_duplicate") for r in duplicate_results.get("results", [])}
        duplicates = [uf for uf in filtered_files if dup_flags.get(uf.name)]
        new_files = [uf for uf in filtered_files if not dup_flags.get(uf.name)]

        if not new_files:
            st.warning("⚠️ All selected files are duplicates. Nothing to upload.")
            if duplicates:
                st.info("Skipped duplicates:\\n- " + "\\n- ".join(d.name for d in duplicates))
            return

        # Upload new files
        config = st.session_state.rag_config
        with st.spinner(f"Processing {len(new_files)} new file(s)..."):
            success, results = rag_service.upload_files(new_files, config["chunk_size"])

        if success and results:
            st.success("✅ Enhanced processing completed!")
            self._display_upload_results(results)
            st.session_state.rag_messages.clear()
            st.info("Chat history cleared to reflect new knowledge base")
            if duplicates:
                st.info("Skipped duplicates:\\n- " + "\\n- ".join(d.name for d in duplicates))
            st.rerun()
        else:
            st.error("❌ Enhanced processing failed. See details above.")

    def _display_upload_results(self, results: Dict[str, Any]):
        """Display detailed upload results."""
        if not results:
            return

        summary = results.get("summary", {})
        file_results = results.get("results", [])

        # Summary metrics
        if summary:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Files", summary.get("total_files", 0))
            with col2:
                st.metric("Successful", summary.get("successful", 0))
            with col3:
                st.metric("Duplicates", summary.get("duplicates", 0))
            with col4:
                st.metric("Total Chunks", summary.get("total_chunks_created", 0))

        # Detailed results
        if file_results:
            st.markdown("**📋 Detailed Results:**")

            successful = [r for r in file_results if r.get("status") == "success"]
            duplicates = [r for r in file_results if r.get("status") == "duplicate"]
            failed = [r for r in file_results if r.get("status") == "error"]

            if successful:
                with st.expander(f"✅ Successfully Processed ({len(successful)} files)", expanded=True):
                    for r in successful:
                        self._display_file_result(r)

            if duplicates:
                with st.expander(f"⚠️ Duplicate Files Skipped ({len(duplicates)} files)", expanded=True):
                    for r in duplicates:
                        st.warning(f"**{r['filename']}**: {r.get('message', 'Duplicate detected')}")

            if failed:
                with st.expander(f"❌ Processing Errors ({len(failed)} files)", expanded=True):
                    for r in failed:
                        st.error(f"**{r['filename']}**: {r.get('message', 'Unknown error')}")

    def _display_file_result(self, result: Dict[str, Any]):
        """Display individual file processing result."""
        processing_type = result.get("processing_type", "standard")
        chunks = result.get("chunks_created", 0)
        filename = result.get("filename", "Unknown")

        if processing_type == "enhanced_csv":
            st.success(f"📊 **{filename}** — Enhanced CSV Analysis ({chunks} intelligent chunks)")
            details = result.get("details", {}) or {}
            st.markdown(f"   • Column analysis: {'✅' if details.get('column_analysis') else '❌'}")
            st.markdown(f"   • Statistical summaries: {'✅' if details.get('statistical_summaries') else '❌'}")
            st.markdown(f"   • Intelligent chunking: {'✅' if details.get('intelligent_chunking') else '❌'}")
        elif processing_type == "enhanced_image":
            st.success(f"🖼️ **{filename}** — Enhanced Image Processing ({chunks} content chunks)")
            details = result.get("details", {}) or {}
            st.markdown(f"   • OCR text extraction: {'✅' if details.get('ocr_extraction') else '❌'}")
            st.markdown(f"   • Content analysis: {'✅' if details.get('content_analysis') else '❌'}")
            st.markdown(f"   • Structure detection: {'✅' if details.get('structured_detection') else '❌'}")
        else:
            st.success(f"📄 **{filename}** — Standard Processing ({chunks} chunks)")

        if result.get("message"):
            st.markdown(f"   *{result['message']}*")

    def _render_query_section(self):
        """Render the query interface."""
        import io

        # Check if realtime voice mode is active
        realtime_active = st.session_state.get("realtime_voice_mode_rag", False)

        if not realtime_active:
            st.subheader("Query Your Knowledge Base")

            # Display chat history
            for idx, message in enumerate(st.session_state.rag_messages):
                with st.chat_message(message["role"]):
                    render_llm_text(message["content"])

                    # Add TTS play button for assistant messages
                    if message["role"] == "assistant":
                        audio_key = f"rag_tts_audio_{idx}"
                        show_key = f"rag_tts_show_{idx}"
                        error_key = f"rag_tts_error_{idx}"

                        # Show audio player if audio is available
                        if st.session_state.get(show_key) and st.session_state.get(audio_key):
                            col1, col2 = st.columns([15, 1])
                            with col1:
                                audio_bytes = st.session_state[audio_key]
                                st.audio(io.BytesIO(audio_bytes), format="audio/mpeg")
                            with col2:
                                if st.button("✕", key=f"rag_close_{idx}", help="Close"):
                                    st.session_state[show_key] = False
                                    st.rerun()
                        else:
                            # Show error if any
                            if st.session_state.get(error_key):
                                st.error(st.session_state[error_key])
                                del st.session_state[error_key]

                            # Show play button
                            if st.button("🗣️ Read Aloud", key=f"rag_tts_{idx}", help="Read this response aloud"):
                                self._generate_tts(message["content"], idx)

        # Get user input (this handles realtime voice mode internally)
        rag_input = get_voice_input("Ask about your documents...", "rag")

        if rag_input:
            self._process_rag_query(rag_input)

    def _generate_tts(self, text: str, idx: int):
        """Generate TTS audio for the given text."""
        import requests

        from ..config import FASTAPI_URL

        with st.spinner("Generating audio..."):
            try:
                response = requests.post(
                    f"{FASTAPI_URL}/api/tts/synthesize", json={"text": text, "use_cache": True}, timeout=60
                )
                if response.status_code == 200 and "audio" in response.headers.get("content-type", ""):
                    st.session_state[f"rag_tts_audio_{idx}"] = response.content
                    st.session_state[f"rag_tts_show_{idx}"] = True
                    st.rerun()
                else:
                    st.error("Failed to generate audio. Please try again.")
            except Exception as e:
                logger.error(f"TTS error: {e}")
                st.error(f"TTS error: {e}")

    def _process_rag_query(self, query: str):
        """Process RAG query and generate response with conversation memory."""
        backend_provider = st.session_state.get("current_backend", "ollama")

        # Validate setup using shared function
        if not validate_backend_setup(backend_provider):
            return

        # Get conversation context BEFORE adding the new message
        conversation_summary = None
        if st.session_state.get("rag_memory_enabled", True):
            conversation_summary = self.memory_manager.summary

        # Add user message
        st.session_state.rag_messages.append({"role": "user", "content": query})

        # Save user message to conversation history if enabled
        if st.session_state.get("rag_history_enabled", True):
            self.conversation_sidebar.save_message("user", query)

        with st.chat_message("user"):
            render_llm_text(query)

        # Get configuration
        config = st.session_state.rag_config
        history = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.rag_messages[:-1]]

        generation_config = get_generation_config_from_session("rag")
        model = self._get_model_for_backend(backend_provider)
        backend_request = get_backend_request_config(backend_provider)

        # Generate response
        with st.chat_message("assistant"):
            status = self._build_status_message(config, backend_provider)

            # Check if realtime voice mode is active
            is_voice_mode = st.session_state.get("realtime_voice_mode_rag", False)

            out = st.empty()
            with st.spinner(status):
                response_text = streaming_handler.stream_rag_response(
                    query=query,
                    messages=history,
                    model=model,
                    system_prompt=st.session_state.rag_system_prompt,
                    n_results=config["n_results"],
                    use_multi_agent=config["use_multi_agent"],
                    use_hybrid_search=config["use_hybrid_search"],
                    backend=backend_provider,
                    api_base=backend_request.get("api_base"),
                    api_key=backend_request.get("api_key"),
                    placeholder=out,
                    conversation_summary=conversation_summary,
                    session_id=self.memory_manager.session_id,
                    temperature=generation_config["temperature"],
                    max_tokens=generation_config["max_tokens"],
                    top_p=generation_config["top_p"],
                    frequency_penalty=generation_config["frequency_penalty"],
                    repetition_penalty=generation_config["repetition_penalty"],
                    is_voice_mode=is_voice_mode,
                )

        if response_text:
            st.session_state.rag_messages.append({"role": "assistant", "content": response_text})

            # Save assistant message to conversation history if enabled
            if st.session_state.get("rag_history_enabled", True):
                self.conversation_sidebar.save_message("assistant", response_text)

            # Check if we need to trigger summarization
            self._check_and_trigger_summarization()

            # Rerun to show TTS button in the message history
            st.rerun()
        else:
            err = "❌ No response received. Please check your query and try again."
            st.error(err)
            st.session_state.rag_messages.append({"role": "assistant", "content": err})

    def _build_status_message(self, config: Dict, backend_provider: str) -> str:
        """Build status message for spinner."""
        status = "Searching enhanced knowledge base"
        if config["use_hybrid_search"]:
            status += " (hybrid search enabled)"
        if config["use_multi_agent"]:
            status += " with multi-agent orchestration"
        status += "..."
        return status

    def _get_model_for_backend(self, backend_provider: str) -> str:
        """Get appropriate model based on backend."""
        if is_hosted_backend(backend_provider):
            return st.session_state.get(
                f"selected_{backend_provider}_rag_model",
                get_default_model_for_backend(backend_provider),
            )
        return st.session_state.get("selected_ollama_model", "default")

    def render_sidebar_config(self):
        """Render RAG-specific sidebar configuration."""
        # Render conversation history sidebar first
        if st.session_state.get("rag_history_enabled", True):
            self.conversation_sidebar.render()

        st.sidebar.markdown("---")
        st.sidebar.subheader("RAG Configuration")

        render_backend_selector()

        # Base model selection for RAG (only when Ollama backend is selected)
        self._render_sidebar_base_model_selection()

        # Generation settings using shared component
        render_generation_settings(
            "rag",
            expanded=True,
        )

        # RAG Configuration Section
        self._render_sidebar_rag_config()

        # Display status
        self._render_sidebar_status()

        # Reasoning settings using shared component
        render_reasoning_config(key_prefix="rag")

        # Memory configuration using shared component
        render_memory_config(self.memory_manager, "rag", "history_enabled", "memory_enabled")

        # System management
        self._render_sidebar_system_management()

    def _render_sidebar_rag_config(self):
        """Render RAG configuration options in sidebar."""
        provider_labels = list(EMBEDDING_PROVIDER_OPTIONS.keys())
        current_provider = _normalize_embedding_provider(
            st.session_state.rag_config.get("provider", "ollama"),
            st.session_state.rag_config.get("embedding_backend"),
        )
        current_label = next(
            (label for label, provider_id in EMBEDDING_PROVIDER_OPTIONS.items() if provider_id == current_provider),
            "Ollama",
        )
        provider = st.sidebar.selectbox(
            "Embedding Provider:",
            provider_labels,
            index=provider_labels.index(current_label),
            help="Provider used for document embeddings. Hosted providers use the same LiteLLM request flow as chat.",
        )
        provider = EMBEDDING_PROVIDER_OPTIONS[provider]

        if provider == "ollama":
            available_models = backend_service.get_available_models()
            embedding_models = [m for m in available_models if "embed" in m.lower()] or available_models
            current_model = st.session_state.rag_config.get("embedding_model")
            default_index = embedding_models.index(current_model) if current_model in embedding_models else 0
            embedding_model = st.sidebar.selectbox(
                "Ollama Embedding Model:",
                embedding_models,
                index=default_index,
                help="Select an embedding model from your Ollama installation",
            )
            embedding_backend = "ollama"
            embedding_api_base = None
            embedding_api_key = None
        elif provider == "huggingface":
            embedding_model = st.sidebar.text_input(
                "HuggingFace Model Repo:",
                value=st.session_state.rag_config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
                help="Enter a HuggingFace model repository path",
            )
            embedding_backend = None
            embedding_api_base = None
            embedding_api_key = None
        else:
            provider_config = get_hosted_backend_config(provider)
            embedding_api_key_key = f"embedding_{provider_config['api_key_key']}"
            embedding_api_base_key = f"embedding_{provider_config['api_base_key']}"
            st.session_state.setdefault(
                embedding_api_key_key,
                st.session_state.get(provider_config["api_key_key"], ""),
            )
            st.session_state.setdefault(
                embedding_api_base_key,
                st.session_state.get(
                    provider_config["api_base_key"],
                    provider_config["default_api_base"],
                ),
            )
            remote_embedding_model = (
                st.session_state.rag_config.get("embedding_model", "")
                if current_provider == provider
                else provider_config["embedding_model_example"]
            )
            embedding_backend = None
            st.sidebar.text_input(
                f"{provider_config['display_name']} Embedding API Base",
                key=embedding_api_base_key,
                help=f"Defaults to {provider_config['default_api_base']}",
            )
            st.sidebar.text_input(
                f"{provider_config['display_name']} Embedding API Key",
                key=embedding_api_key_key,
                type="password",
                help=(f"Optional if the FastAPI backend already has {provider_config['api_key_env_var']} configured."),
            )
            embedding_model = st.sidebar.text_input(
                f"{provider_config['display_name']} Embedding Model:",
                value=remote_embedding_model,
                help=f"Example: {provider_config['embedding_model_example']}",
            )
            embedding_api_key = (
                (st.session_state.get(embedding_api_key_key) or "").strip()
                or (st.session_state.get(provider_config["api_key_key"]) or "").strip()
                or None
            )
            embedding_api_base = (
                (st.session_state.get(embedding_api_base_key) or "").strip()
                or (st.session_state.get(provider_config["api_base_key"]) or "").strip()
                or provider_config["default_api_base"]
            )

        chunk_size = st.sidebar.number_input(
            "Chunk Size:",
            value=500,
            min_value=100,
            max_value=2000,
            step=100,
            help="Text chunk size for standard processing",
        )

        n_results = st.sidebar.number_input(
            "Number of Results:",
            value=3,
            min_value=1,
            max_value=10,
            help="Number of relevant chunks to retrieve",
        )

        use_multi_agent = st.sidebar.checkbox(
            "Enable Multi-Agent Processing",
            value=st.session_state.rag_config.get("use_multi_agent", False),  # Use session state value
            help="Use multiple specialized agents for enhanced processing",
        )

        use_hybrid_search = st.sidebar.checkbox(
            "Enable Hybrid Search",
            value=st.session_state.rag_config.get("use_hybrid_search", False),  # Use session state value
            help="Combine BM25 with semantic vector search (sparse + dense retrieval)",
        )

        # Save configuration
        if st.sidebar.button("💾 Save Configuration", type="primary"):
            # Save core RAG configuration via API
            success, message = rag_service.save_configuration(
                provider=provider,
                embedding_model=embedding_model,
                embedding_backend=embedding_backend,
                embedding_api_base=embedding_api_base,
                embedding_api_key=embedding_api_key,
                chunk_size=chunk_size,
            )
            st.session_state.config_saved = success
            st.session_state.config_message = message

        # Store values in session state for later use
        st.session_state.rag_config = {
            "provider": provider,
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
            "n_results": n_results,
            "use_multi_agent": use_multi_agent,
            "use_hybrid_search": use_hybrid_search,
        }

        # Display configuration status
        if st.session_state.config_message:
            if st.session_state.config_saved:
                st.sidebar.success(st.session_state.config_message)
            else:
                st.sidebar.error(st.session_state.config_message)

    def _render_sidebar_base_model_selection(self):
        """Render base model selection for RAG in sidebar."""
        backend_provider = st.session_state.get("current_backend", "ollama")
        if backend_provider == "ollama" and st.session_state.get("backend_available", False):
            enhanced = backend_service.get_enhanced_models()
            # Exclude embedding / reranking models from the base model list
            local_models = [m["name"] for m in enhanced.get("local_models", []) if not _is_embedding_model(m["name"])]
            cloud_models = enhanced.get("cloud_models", [])

            # Build unified option list with indicators
            options: list[str] = []
            local_set: set[str] = set()
            for name in local_models:
                label = f"🟢 {name}"
                options.append(label)
                local_set.add(label)
            for cm in cloud_models:
                label = f"☁️ {cm['name']}"
                desc = cm.get("description", "")
                if desc:
                    label += f"  ({desc})"
                options.append(label)

            if options:
                selected_label = st.sidebar.selectbox(
                    "Select Base Model:",
                    options,
                    help="🟢 = installed locally  ·  ☁️ = listed in cloud catalog",
                    key="selected_ollama_model_label",
                )

                # Extract raw model name
                raw_name = selected_label
                for prefix in ("🟢 ", "☁️ "):
                    if raw_name.startswith(prefix):
                        raw_name = raw_name[len(prefix) :]
                        break
                if "  (" in raw_name:
                    raw_name = raw_name[: raw_name.index("  (")]

                # Cloud models are catalog entries only (no pull action in UI)
                if selected_label not in local_set:
                    st.sidebar.info(f"☁️ **{raw_name}** is from the cloud catalog.")

                st.session_state["selected_ollama_model"] = raw_name
            else:
                st.sidebar.warning("⚠️ No Ollama models available")
        elif backend_provider == "openrouter":
            st.sidebar.text_input(
                "OpenRouter Base Model:",
                key="selected_openrouter_rag_model",
                help="Example: meta-llama/llama-3.3-70b-instruct or google/gemini-2.5-flash",
            )
            st.sidebar.info("OpenRouter requests are routed directly through LiteLLM.")
        elif backend_provider == "nvidia_nim":
            st.sidebar.text_input(
                "Nvidia NIM Base Model:",
                key="selected_nvidia_nim_rag_model",
                help="Example: meta/llama3-70b-instruct or mistralai/mixtral-8x7b-instruct",
            )
            st.sidebar.info("Nvidia NIM requests are routed directly through LiteLLM.")
        else:
            st.sidebar.warning("⚠️ Unsupported backend selected.")

    def _render_sidebar_status(self):
        """Render RAG status in sidebar."""
        st.sidebar.markdown("---")
        st.sidebar.subheader("RAG System Status")

        rag_status = rag_service.get_status()
        if rag_status:
            if rag_status["status"] == "ready":
                st.sidebar.write(f"📁 Files: {rag_status.get('uploaded_files', 0)}")
                st.sidebar.write(f"📊 Chunks: {rag_status.get('indexed_chunks', 0)}")
            else:
                st.sidebar.warning(f"⚠️ Status: {rag_status['status']}")
        else:
            st.sidebar.error("❌ Cannot connect to backend")

    def _render_sidebar_model_selection(self):
        """Render model selection in sidebar."""
        available_models = backend_service.get_available_models()
        st.sidebar.selectbox(
            "Select Base Model:",
            available_models,
            help="Language model used for generating responses",
            key="selected_rag_model",
        )

    def _render_sidebar_system_prompt(self):
        """Render system prompt configuration in sidebar."""
        st.sidebar.markdown("---")
        st.sidebar.subheader("📝 System Prompt")

        rag_system_prompt = st.sidebar.text_area(
            "RAG System Prompt:",
            value=st.session_state.rag_system_prompt,
            height=120,
            help="Customize the system prompt for RAG responses",
            key="rag_system_prompt_input",
        )

        if st.sidebar.button("🔄 Reset to Default", help="Reset to default system prompt"):
            st.session_state.rag_system_prompt = DEFAULT_RAG_SYSTEM_PROMPT
            st.rerun()

        # Update session state
        st.session_state.rag_system_prompt = rag_system_prompt

    def _render_sidebar_system_management(self):
        """Render system management options in sidebar."""
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔄 System Management")

        # Reset system
        if st.sidebar.button(
            "🗑️ Reset System",
            type="secondary",
            help="Clear all uploaded documents, embeddings, and chat history",
            disabled=not st.session_state.config_saved,
        ):
            st.session_state.show_reset_confirm = True

        # Reset confirmation
        if st.session_state.show_reset_confirm:
            st.sidebar.markdown("---")
            st.sidebar.warning("⚠️ **Confirm Reset**")
            st.sidebar.write("This will permanently delete:")
            st.sidebar.write("• All uploaded documents")
            st.sidebar.write("• All embeddings and indexes")
            st.sidebar.write("• Current chat history")

            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("✅ Confirm", type="primary", key="confirm_reset"):
                    self._confirm_reset()

            with col2:
                if st.button("❌ Cancel", key="cancel_reset"):
                    st.session_state.show_reset_confirm = False
                    st.rerun()

    def _confirm_reset(self):
        """Confirm and execute system reset."""
        with st.spinner("Resetting RAG system..."):
            success, message = rag_service.reset_system()
            if success:
                st.success(f"✅ {message}")
                st.session_state.rag_messages.clear()
                st.session_state.show_reset_confirm = False
                # Also clear conversation memory
                self.memory_manager.clear()
                st.rerun()
            else:
                st.error(f"❌ {message}")
                st.session_state.show_reset_confirm = False

    def _check_and_trigger_summarization(self):
        """Check if summarization is needed and trigger it."""
        if not st.session_state.get("rag_memory_enabled", True):
            return

        stats = self.memory_manager.get_stats()

        if stats.needs_summarization:
            logger.info("Context limit approaching, triggering summarization for RAG...")

            # Get messages to summarize
            messages_to_summarize = self.memory_manager.prune_for_summarization()

            if messages_to_summarize:
                # Create a simple summary using shared function
                simple_summary = create_simple_summary(messages_to_summarize, self.memory_manager.summary)

                self.memory_manager.set_summary(simple_summary)
                logger.info(f"Created RAG summary with {len(simple_summary)} characters")


def render_rag_page():
    """Render the RAG page."""
    rag_page = RAGPage()
    rag_page.render_sidebar_config()
    rag_page.render()

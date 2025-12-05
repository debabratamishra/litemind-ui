"""
RAG page implementation.
"""
import logging
import streamlit as st
from typing import Dict, Optional, Any

from ..components.voice_input import get_voice_input
from ..components.text_renderer import render_llm_text
from ..components.streaming_handler import streaming_handler
from ..components.tts_playback import check_tts_available
from ..services.backend_service import backend_service
from ..services.rag_service import rag_service
from ..config import DEFAULT_RAG_SYSTEM_PROMPT, SUPPORTED_EXTENSIONS, FASTAPI_URL

logger = logging.getLogger(__name__)


class RAGPage:
    """Handles the RAG interface and document processing."""
    
    def __init__(self):
        self.backend_available = st.session_state.get("backend_available", False)
        self._initialize_session_state()
        
    def render(self):
        """Render the complete RAG page."""
        if not self.backend_available:
            st.info("📡 Enhanced RAG functionality requires the FastAPI backend. Please start the backend server.")
            return
        
        st.title("📚 RAG Interface")
        
        self._initialize_session_state()
        
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
        st.session_state.setdefault("rag_config", {
            "provider": "Ollama",
            "embedding_model": "snowflake-arctic-embed2:latest",
            "chunk_size": 500,
            "n_results": 3,
            "use_multi_agent": False,
            "use_hybrid_search": False,
        })
    
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
                    key="rag_system_prompt_main"
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
        if uploaded_files and st.button("📤 Upload & Process", type="primary", 
                                       disabled=not st.session_state.config_saved):
            self._process_uploaded_files(uploaded_files)
    
    def _display_rag_status(self):
        """Display current RAG system status."""
        rag_status = rag_service.get_status()
        if rag_status:
            if rag_status["status"] == "ready":
                if rag_status.get("uploaded_files", 0) > 0:
                    st.info(f"📊 Current Knowledge Base: {rag_status['uploaded_files']} files, "
                           f"{rag_status['indexed_chunks']} chunks indexed")
                else:
                    st.info("📭 Knowledge base is empty. Upload documents to get started.")
            else:
                st.warning(f"⚠️ System Status: {rag_status['status']}")
        else:
            st.error("❌ Cannot connect to RAG backend")
    
    def _process_uploaded_files(self, uploaded_files):
        """Process uploaded files with duplicate checking."""
        # Filter unsupported file types since types are hidden in the uploader UI
        allowed_suffixes = tuple(f".{ext.lower()}" for ext in SUPPORTED_EXTENSIONS)
        candidate_files = uploaded_files or []
        filtered_files = [
            uf for uf in candidate_files if uf.name.lower().endswith(allowed_suffixes)
        ]
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
        dup_flags = {r.get("filename"): r.get("is_duplicate") 
                    for r in duplicate_results.get("results", [])}
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
            st.info("💡 Chat history cleared to reflect new knowledge base")
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
        st.subheader("Query Your Knowledge Base")
        
        tts_available = check_tts_available() if st.session_state.get("tts_enabled", True) else False
        
        # Display chat history
        for idx, message in enumerate(st.session_state.rag_messages):
            with st.chat_message(message["role"]):
                render_llm_text(message["content"])
                
                # Add TTS play button for assistant messages
                if message["role"] == "assistant" and tts_available:
                    self._render_tts_button(message["content"], idx)
        
        # Get user input
        rag_input = get_voice_input(
            "Ask about your documents...", 
            "rag"
        )
        
        if rag_input:
            self._process_rag_query(rag_input)
    
    def _render_tts_button(self, text: str, message_index: int):
        """Render TTS play button for a message."""
        import io
        import requests
        
        audio_key = f"rag_tts_audio_{message_index}"
        show_key = f"rag_tts_show_{message_index}"
        
        # Initialize state
        if audio_key not in st.session_state:
            st.session_state[audio_key] = None
        if show_key not in st.session_state:
            st.session_state[show_key] = False
        
        # If audio is loaded, show player
        if st.session_state[show_key] and st.session_state[audio_key]:
            try:
                audio_bytes = st.session_state[audio_key]
                logger.info(f"Rendering audio player: {len(audio_bytes) if audio_bytes else 0} bytes")
                
                col1, col2 = st.columns([15, 1])
                with col1:
                    # Use BytesIO wrapper for audio bytes
                    if isinstance(audio_bytes, bytes):
                        audio_io = io.BytesIO(audio_bytes)
                        st.audio(audio_io, format="audio/mpeg")
                    else:
                        st.audio(audio_bytes, format="audio/mpeg")
                with col2:
                    if st.button("✕", key=f"rag_close_{message_index}", help="Close"):
                        st.session_state[show_key] = False
                        st.rerun()
            except Exception as e:
                logger.error(f"Audio player error: {type(e).__name__}: {e}")
                st.error(f"Audio playback error: {e}")
        else:
            # Show play button
            if st.button("🗣️", key=f"rag_play_{message_index}", help="Read aloud"):
                with st.spinner("Generating audio..."):
                    try:
                        response = requests.post(
                            f"{FASTAPI_URL}/api/tts/synthesize",
                            json={"text": text, "use_cache": True},
                            timeout=60
                        )
                        logger.info(f"TTS response: status={response.status_code}, size={len(response.content)}, content_type={response.headers.get('content-type')}")
                        
                        if response.status_code == 200 and len(response.content) > 0:
                            # Check if it's actually audio and not an error JSON
                            content_type = response.headers.get('content-type', '')
                            if 'audio' in content_type:
                                st.session_state[audio_key] = response.content
                                st.session_state[show_key] = True
                                logger.info(f"Audio saved to session state, triggering rerun")
                                st.rerun()
                            else:
                                # Might be an error response
                                try:
                                    error_data = response.json()
                                    logger.error(f"TTS returned non-audio response: {error_data}")
                                    st.error(f"TTS Error: {error_data.get('detail', 'Unknown error')}")
                                except:
                                    # Not JSON, treat as audio anyway
                                    st.session_state[audio_key] = response.content
                                    st.session_state[show_key] = True
                                    st.rerun()
                        else:
                            error_msg = f"Status: {response.status_code}"
                            try:
                                error_data = response.json()
                                error_msg = error_data.get('detail', error_msg)
                            except:
                                pass
                            logger.error(f"TTS failed: {error_msg}")
                            st.error(f"Failed to generate audio: {error_msg}")
                    except requests.exceptions.Timeout:
                        logger.error("TTS request timed out")
                        st.error("Audio generation timed out. Try a shorter message.")
                    except requests.exceptions.ConnectionError as e:
                        logger.error(f"TTS connection error: {e}")
                        st.error("Cannot connect to backend. Is the server running?")
                    except Exception as e:
                        logger.error(f"TTS failed: {type(e).__name__}: {e}")
                        st.error(f"Audio generation failed: {str(e)}")
    
    def _process_rag_query(self, query: str):
        """Process RAG query and generate response."""
        backend_provider = st.session_state.get("current_backend", "ollama")
        is_docker = st.session_state.get("is_docker_deployment", False)
        
        # Validate setup - prevent vLLM usage in Docker
        if backend_provider == "vllm" and is_docker:
            st.error("❌ vLLM is not supported with Docker installation yet. Please use Ollama backend.")
            return
        
        if backend_provider == "vllm" and not st.session_state.get("vllm_model"):
            st.error("❌ Please configure and load a vLLM model first")
            return
        
        # Add user message
        st.session_state.rag_messages.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            render_llm_text(query)
        
        # Get configuration
        config = st.session_state.rag_config
        history = [{"role": msg["role"], "content": msg["content"]} 
                  for msg in st.session_state.rag_messages[:-1]]
        
        # Generate response
        with st.chat_message("assistant"):
            status = self._build_status_message(config, backend_provider)
            
            out = st.empty()
            with st.spinner(status):
                model = self._get_model_for_backend(backend_provider)
                response_text = streaming_handler.stream_rag_response(
                    query=query,
                    messages=history,
                    model=model,
                    system_prompt=st.session_state.rag_system_prompt,
                    n_results=config["n_results"],
                    use_multi_agent=config["use_multi_agent"],
                    use_hybrid_search=config["use_hybrid_search"],
                    backend=backend_provider,
                    hf_token=st.session_state.get("hf_token") if backend_provider == "vllm" else None,
                    placeholder=out
                )

        if response_text:
            st.session_state.rag_messages.append({"role": "assistant", "content": response_text})
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
        if backend_provider == "vllm":
            model = st.session_state.get("vllm_model", "unknown")
            status += f" (vLLM - {model})"
        status += "..."
        return status
    
    def _get_model_for_backend(self, backend_provider: str) -> str:
        """Get appropriate model based on backend."""
        if backend_provider == "vllm":
            return st.session_state.get("vllm_model", "no-model")
        else:
            return st.session_state.get("selected_ollama_model", "default")
    
    def render_sidebar_config(self):
        """Render RAG-specific sidebar configuration."""
        st.sidebar.markdown("---")
        st.sidebar.subheader("📚 RAG Configuration")
        
        # Base model selection for RAG (only when Ollama backend is selected)
        self._render_sidebar_base_model_selection()

        # RAG Configuration Section
        self._render_sidebar_rag_config()
        
        # Display status
        self._render_sidebar_status()
        
        # Reasoning settings
        self._render_sidebar_reasoning_config()
        
        # System management
        self._render_sidebar_system_management()
    
    def _render_sidebar_rag_config(self):
        """Render RAG configuration options in sidebar."""
        provider = st.sidebar.selectbox(
            "Embedding Provider:",
            ["Ollama", "HuggingFace"],
            index=0,
            help="Provider used for document embeddings",
        )

        if provider == "Ollama":
            available_models = backend_service.get_available_models()
            embedding_models = [m for m in available_models if "embed" in m.lower()] or available_models
            embedding_model = st.sidebar.selectbox(
                "Ollama Embedding Model:",
                embedding_models,
                help="Select an embedding model from your Ollama installation",
            )
        else:
            embedding_model = st.sidebar.text_input(
                "HuggingFace Model Repo:",
                value="sentence-transformers/all-MiniLM-L6-v2",
                help="Enter a HuggingFace model repository path",
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
                chunk_size=chunk_size
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
        is_docker = st.session_state.get("is_docker_deployment", False)
        
        # Handle vLLM in Docker deployment
        if backend_provider == "vllm" and is_docker:
            st.sidebar.warning("⚠️ vLLM not supported in Docker deployment")
            st.sidebar.info("Please use Ollama backend for RAG operations")
        # Only show Ollama base model selection when Ollama backend is selected
        elif backend_provider == "ollama" and st.session_state.get("backend_available", False):
            
            available_models = backend_service.get_available_models()
            if available_models:
                st.sidebar.selectbox(
                    "Select Base Model:",
                    available_models,
                    help="Language model used for generating RAG responses",
                    key="selected_ollama_model"
                )
            else:
                st.sidebar.warning("⚠️ No Ollama models available")
        elif backend_provider == "vllm":
            vllm_model = st.session_state.get("vllm_model")
            if vllm_model:
                st.sidebar.success(f"🎯 Active vLLM Model: {vllm_model}")
            else:
                st.sidebar.warning("⚠️ No vLLM model loaded")
                st.sidebar.info("Configure vLLM in the backend section above")
    
    def _render_sidebar_status(self):
        """Render RAG status in sidebar."""
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
        backend_provider = st.session_state.get("current_backend", "ollama")
        
        if backend_provider == "vllm":
            vllm_model = st.session_state.get("vllm_model")
            if vllm_model:
                st.sidebar.success(f"🎯 Active Model: {vllm_model}")
            else:
                st.sidebar.warning("⚠️ No vLLM model loaded")
                st.sidebar.info("👆 Configure vLLM above to load a model")
        else:
            available_models = backend_service.get_available_models()
            st.sidebar.selectbox(
                "Select Base Model:",
                available_models,
                help="Language model used for generating responses",
                key="selected_rag_model"
            )
    
    def _render_sidebar_reasoning_config(self):
        """Render reasoning configuration in sidebar."""
        st.sidebar.markdown("---")
        st.sidebar.subheader("🧠 Reasoning Display")
        
        st.session_state.show_reasoning_expanded = st.sidebar.checkbox(
            "Expand reasoning by default",
            value=st.session_state.get("show_reasoning_expanded", False),
            help="Show model reasoning sections expanded by default",
            key="rag_reasoning_expanded"
        )
        
        hide_reasoning = st.sidebar.checkbox(
            "Hide reasoning completely",
            value=st.session_state.get("hide_reasoning", False),
            help="Completely hide reasoning sections from responses",
            key="rag_hide_reasoning"
        )
        st.session_state.hide_reasoning = hide_reasoning
    
    def _render_sidebar_system_prompt(self):
        """Render system prompt configuration in sidebar."""
        st.sidebar.markdown("---")
        st.sidebar.subheader("📝 System Prompt")
        
        rag_system_prompt = st.sidebar.text_area(
            "RAG System Prompt:",
            value=st.session_state.rag_system_prompt,
            height=120,
            help="Customize the system prompt for RAG responses",
            key="rag_system_prompt_input"
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
        if st.sidebar.button("🗑️ Reset System", 
                           type="secondary", 
                           help="Clear all uploaded documents, embeddings, and chat history",
                           disabled=not st.session_state.config_saved):
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
                st.rerun()
            else:
                st.error(f"❌ {message}")
                st.session_state.show_reset_confirm = False


def render_rag_page():
    """Render the RAG page."""
    rag_page = RAGPage()
    rag_page.render_sidebar_config()
    rag_page.render()

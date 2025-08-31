"""
Text rendering components for LLM responses with reasoning support.
"""
import re
import streamlit as st
from typing import Optional, Any

from ..utils.text_processing import (
    unescape_text, clean_text_formatting, clean_markdown_text, 
    sanitize_links, extract_thinking_content
)


class ThinkingRenderer:
    """Renders model reasoning/thinking sections"""
    
    def __init__(self):
        self.reasoning_hidden = st.session_state.get("hide_reasoning", False)
        self.reasoning_expanded_default = st.session_state.get("show_reasoning_expanded", False)

    def render_thinking_section(self, thinking_content: str) -> None:
        """Render the thinking/reasoning section in a collapsible expander."""
        if not thinking_content.strip() or self.reasoning_hidden:
            return
        
        with st.expander("🧠 Model Reasoning Process", expanded=self.reasoning_expanded_default):
            st.markdown("*This section shows the model's internal reasoning and thought process:*")
            st.markdown("---")
            
            cleaned_thinking = clean_text_formatting(thinking_content)
            cleaned_thinking = clean_markdown_text(cleaned_thinking)
            
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, rgba(63, 81, 181, 0.1), rgba(33, 150, 243, 0.05));
                    padding: 1.2rem;
                    border-radius: 0.75rem;
                    border-left: 4px solid #3f51b5;
                    margin: 0.5rem 0;
                    font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, sans-serif;
                ">
                    {sanitize_links(cleaned_thinking)}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.caption("💡 This reasoning helps understand how the model arrived at its answer.")


class TextRenderer:
    """Renders mixed Markdown and fenced code blocks with reasoning support."""
    
    def __init__(self):
        self.thinking_renderer = ThinkingRenderer()
        
    def render_llm_text(self, text: str) -> None:
        """Render mixed Markdown and fenced code blocks in Streamlit with reasoning support."""
        text = unescape_text(text).strip()
        
        # Extract thinking/reasoning content first
        thinking_content, clean_text = extract_thinking_content(text)
        
        # Clean up common formatting issues
        clean_text = clean_text_formatting(clean_text)
        
        # Render the thinking section if present and not hidden
        if thinking_content:
            self.thinking_renderer.render_thinking_section(thinking_content)
            st.markdown("---")  # Separator between thinking and answer
        
        # Render the main answer content
        self._render_main_content(clean_text)

    def _render_main_content(self, text: str) -> None:
        """Render the main content with code blocks."""
        code_fence = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)

        pos = 0
        last_end = 0
        rendered_any = False

        for match in code_fence.finditer(text):
            before = text[pos:match.start()]
            if before.strip():
                cleaned_before = clean_markdown_text(before)
                st.markdown(sanitize_links(cleaned_before))
                rendered_any = True

            lang = (match.group(1) or "").strip()
            code = match.group(2).strip("\n")
            st.code(code, language=lang if lang else None)
            rendered_any = True

            pos = match.end()
            last_end = match.end()

        tail = text[last_end:] if last_end else text
        if tail.strip() or not rendered_any:
            cleaned_tail = clean_markdown_text(tail)
            st.markdown(sanitize_links(cleaned_tail))


class StreamingRenderer:
    """
    Handles live streaming text with reasoning segregation.
    Separates <think>...</think> content during streaming.
    """
    
    _OPEN_TAG_RE = re.compile(r"<\s*(think|thinking|reasoning|thought)\s*>", re.IGNORECASE)
    _CLOSE_TAG_RE = re.compile(r"<\s*/\s*(think|thinking|reasoning|thought)\s*>", re.IGNORECASE)
    _ANY_TAG_RE = re.compile(r"(?is)<\s*/?\s*(think|thinking|reasoning|thought)\s*>")

    def __init__(self, placeholder: Optional[Any]):
        self.placeholder = placeholder
        self.in_think = False
        self.pending = ""
        self.thinking_text = ""
        self.answer_text = ""
        self.reasoning_hidden = st.session_state.get("hide_reasoning", False)
        self.reasoning_expanded_default = st.session_state.get("show_reasoning_expanded", False)

        if self.placeholder is not None:
            self.root = self.placeholder.container()
            self._thinking_outer = self.root.empty()
            self._separator = self.root.empty()
            self._answer_box = self.root.empty()
        else:
            self.root = None
            self._thinking_outer = None
            self._separator = None
            self._answer_box = None

        self._expander_created = False
        self._reasoning_box = None

    def _ensure_reasoning_ui(self) -> None:
        """Create the reasoning UI components."""
        if self.reasoning_hidden or self._expander_created or self._thinking_outer is None:
            return
            
        with self._thinking_outer:
            with st.expander("🧠 Model Reasoning Process", expanded=self.reasoning_expanded_default):
                st.markdown("*This section shows the model's internal reasoning and thought process:*")
                st.markdown("---")
                self._reasoning_box = st.empty()
                
        if self._separator is not None:
            self._separator.markdown("---")
        self._expander_created = True

    def _render_reasoning(self) -> None:
        """Render the reasoning content."""
        if self.reasoning_hidden or self._reasoning_box is None:
            return
            
        cleaned = clean_text_formatting(self.thinking_text)
        cleaned = clean_markdown_text(cleaned)
        cleaned = sanitize_links(unescape_text(cleaned))
        
        self._reasoning_box.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, rgba(63, 81, 181, 0.1), rgba(33, 150, 243, 0.05));
                padding: 1.2rem;
                border-radius: 0.75rem;
                border-left: 4px solid #3f51b5;
                margin: 0.5rem 0;
                font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, sans-serif;
            ">
                {cleaned}
            </div>
            """,
            unsafe_allow_html=True
        )

    def _render_answer(self) -> None:
        """Render the answer content."""
        if self._answer_box is None:
            return
            
        cleaned = clean_text_formatting(self.answer_text)
        cleaned = clean_markdown_text(cleaned)
        cleaned = sanitize_links(unescape_text(cleaned))
        self._answer_box.markdown(cleaned)

    def _append_text(self, text: str) -> None:
        """Append text to the appropriate section."""
        if not text:
            return
            
        if self.in_think:
            self._ensure_reasoning_ui()
            self.thinking_text += text
            self._render_reasoning()
        else:
            self.answer_text += text
            self._render_answer()

    def feed(self, chunk: str) -> None:
        """
        Feed a new chunk of text from the stream.
        Incrementally parses thinking tags and updates the UI.
        """
        self.pending += chunk

        # Process complete tags found in the buffer
        while True:
            m = self._ANY_TAG_RE.search(self.pending)
            if not m:
                break

            before = self.pending[:m.start()]
            self._append_text(before)

            tag_text = m.group(0)
            is_open = bool(self._OPEN_TAG_RE.fullmatch(tag_text))
            is_close = bool(self._CLOSE_TAG_RE.fullmatch(tag_text))

            if is_open:
                self.in_think = True
                self._ensure_reasoning_ui()
            elif is_close:
                self.in_think = False

            self.pending = self.pending[m.end():]

        # Handle partial tag at the end
        last_lt = self.pending.rfind("<")
        if last_lt != -1 and ">" not in self.pending[last_lt:]:
            self._append_text(self.pending[:last_lt])
            self.pending = self.pending[last_lt:]
        else:
            self._append_text(self.pending)
            self.pending = ""


# Global renderer instances
text_renderer = TextRenderer()

def render_llm_text(text: str) -> None:
    """Render LLM text with reasoning support."""
    text_renderer.render_llm_text(text)

"""Web search orchestrator using CrewAI agents.

This module provides a two-agent system for web search and synthesis:
- Serper Agent: Retrieves search results via WebSearchService
- Synthesizer Agent: Processes results and generates responses using Base LLM
"""
import os
import logging
from typing import List, Dict, Optional, AsyncGenerator
from crewai import Agent, LLM
import asyncio
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

from app.services.web_search_service import WebSearchService

logger = logging.getLogger(__name__)


class WebSearchOrchestrator:
    """Orchestrates web search using CrewAI agents for retrieval and synthesis."""
    
    def __init__(self):
        """Initialize the orchestrator with WebSearchService and Ollama LLM."""
        self.web_search_service = WebSearchService()
        
        # Get Ollama URL using the same pattern as RAGService
        ollama_url = self._get_ollama_url()
        
        # Get the model name from environment or use default
        model_name = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
        
        # Initialize the LLM for CrewAI agents
        self.llm = LLM(
            model=f"ollama/{model_name}",
            base_url=ollama_url
        )
        
        logger.info(f"WebSearchOrchestrator initialized with model: {model_name} at {ollama_url}")
        
        # Initialize agents
        self._initialize_agents()
    
    def _get_ollama_url(self) -> str:
        """Get the appropriate Ollama URL based on execution environment.
        
        This follows the same pattern as RAGService to ensure consistency.
        
        Returns:
            str: Ollama API URL
        """
        try:
            from app.services.host_service_manager import host_service_manager
            url = host_service_manager.environment_config.ollama_url
            logger.debug(f"Using Ollama URL from host service manager: {url}")
            return url
        except ImportError:
            logger.warning("Host service manager not available, using fallback config")
            url = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
            logger.debug(f"Using fallback Ollama URL: {url}")
            return url
    
    def _initialize_agents(self):
        """Initialize the Query Optimizer, Serper, and Synthesizer agents."""
        # Query Optimizer Agent: Optimizes user queries for better search results
        self.query_optimizer_agent = Agent(
            role="Query Optimization Expert",
            goal="Transform user queries into optimal search queries that will retrieve the most relevant results",
            backstory=(
                "You are an expert in search query optimization with years of experience in information retrieval. "
                "You understand how to refine ambiguous or complex questions into clear, focused search queries. "
                "You identify key terms, remove unnecessary words, and structure queries for maximum relevance. "
                "You consider search engine behavior and user intent to create queries that yield the best results."
            ),
            llm=self.llm,
            verbose=False,
            allow_delegation=False
        )
        
        # Serper Agent: Responsible for retrieving web search results
        self.serper_agent = Agent(
            role="Web Search Specialist",
            goal="Retrieve relevant and accurate web search results using optimized queries",
            backstory=(
                "You are an expert web search specialist with deep knowledge of information retrieval. "
                "Your mission is to find the most relevant and up-to-date information from the web "
                "by executing precise search queries. You work with optimized queries to ensure "
                "the best possible results that directly address user needs."
            ),
            llm=self.llm,
            verbose=False,
            allow_delegation=False
        )
        
        # Synthesizer Agent: Responsible for processing results and generating natural responses
        self.synthesizer_agent = Agent(
            role="Information Synthesizer",
            goal="Answer questions based on web search results",
            backstory=(
                "You synthesize information from web search results to answer user questions. "
                "You cite sources using [1], [2], etc."
            ),
            llm=self.llm,
            verbose=False,
            allow_delegation=False
        )
        
        logger.info("CrewAI agents initialized successfully (3-agent pipeline)")

    async def process_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """Process a user query through the web search workflow.
        
        This is the main entry point for web search queries. It orchestrates:
        1. Search execution via Serper Agent
        2. Result synthesis via Synthesizer Agent
        3. Streaming response generation
        
        Args:
            query: The user's search query
            conversation_history: Optional list of previous messages for context
            stream: Whether to stream the response (default: True)
            
        Yields:
            str: Chunks of the synthesized response
            
        Raises:
            Exception: Falls back to Base LLM on any error
        """
        try:
            logger.info(f"Processing web search query: '{query}'")
            
            # Step 1: Optimize query via Query Optimizer Agent
            optimized_query = await self._query_optimizer_process(query)
            logger.info(f"Optimized query: '{optimized_query}'")
            
            # Step 2: Execute search via Serper Agent using optimized query
            search_results = await self._serper_agent_search(optimized_query)
            
            if not search_results:
                logger.warning("No search results retrieved, falling back to Base LLM")
                async for chunk in self._fallback_to_base_llm(query, conversation_history):
                    yield chunk
                return
            
            # Step 3: Synthesize results via Synthesizer Agent
            async for chunk in self._synthesizer_agent_process(
                query=query,  # Use original query for context
                optimized_query=optimized_query,
                search_results=search_results,
                conversation_history=conversation_history,
                stream=stream
            ):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in web search workflow: {e}", exc_info=True)
            logger.info("Falling back to Base LLM due to error")
            
            # Fallback to Base LLM on any error
            async for chunk in self._fallback_to_base_llm(query, conversation_history):
                yield chunk
    
    async def _query_optimizer_process(self, query: str) -> str:
        """Optimize the user query for better search results.
        
        Args:
            query: The original user query
            
        Returns:
            str: Optimized search query
        """
        try:
            logger.info(f"Query Optimizer processing: '{query}'")
            
            optimization_prompt = (
                f"Optimize the following user question into a clear, focused search query. "
                f"Extract key terms, remove unnecessary words, and create a query that will "
                f"retrieve the most relevant search results. Return ONLY the optimized query, "
                f"nothing else.\n\n"
                f"Original question: {query}\n\n"
                f"Optimized search query:"
            )
            
            # Get optimized query from LLM
            optimized_query = ""
            async for chunk in self._stream_from_ollama(optimization_prompt):
                optimized_query += chunk
            
            # Clean up the response - remove any extra text
            optimized_query = optimized_query.strip()
            
            # If optimization failed or is too short, use original
            if not optimized_query or len(optimized_query) < 3:
                logger.warning("Query optimization failed, using original query")
                return query
            
            return optimized_query
            
        except Exception as e:
            logger.error(f"Query optimization failed: {e}", exc_info=True)
            return query  # Fallback to original query
    
    async def _serper_agent_search(self, query: str) -> List[Dict[str, any]]:
        """Execute search via WebSearchService and return formatted results.
        
        Args:
            query: The search query string
            
        Returns:
            list: Formatted search results (up to 10 results)
            
        Raises:
            Exception: If search fails
        """
        try:
            logger.info(f"Serper Agent executing search for: '{query}'")
            
            # Execute search via WebSearchService (retrieves exactly 10 results)
            raw_results = await self.web_search_service.search(query, num_results=10)
            
            # Format results for agent consumption
            formatted_results = self.web_search_service.format_results(raw_results)
            
            logger.info(f"Serper Agent retrieved {len(formatted_results)} search results")
            return formatted_results
            
        except ValueError as e:
            # API key validation error
            logger.error(f"SerpAPI token validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Search execution failed: {e}", exc_info=True)
            raise
    
    async def _synthesizer_agent_process(
        self,
        query: str,
        optimized_query: str,
        search_results: List[Dict[str, any]],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """Synthesize search results into a comprehensive response using Base LLM.
        
        Args:
            query: The original user query
            optimized_query: The optimized search query used
            search_results: Formatted search results from Serper Agent
            conversation_history: Optional conversation context
            stream: Whether to stream the response
            
        Yields:
            str: Chunks of the synthesized response
        """
        try:
            logger.info("Synthesizer Agent processing search results")
            
            # Build context from search results
            search_context = self._build_search_context(search_results)
            
            # Build conversation context if available
            conversation_context = ""
            if conversation_history:
                conversation_context = self._build_conversation_context(conversation_history)
            
            # Create synthesis prompt
            synthesis_prompt = self._create_synthesis_prompt(
                query=query,
                optimized_query=optimized_query,
                search_context=search_context,
                conversation_context=conversation_context
            )
            
            # Stream response from Base LLM
            logger.info("Streaming synthesis response from Base LLM")
            async for chunk in self._stream_from_ollama(synthesis_prompt):
                yield chunk
                
        except Exception as e:
            logger.error(f"Synthesis failed: {e}", exc_info=True)
            raise
    
    def _build_search_context(self, search_results: List[Dict[str, any]]) -> str:
        """Build a formatted context string from search results.
        
        Args:
            search_results: List of formatted search results
            
        Returns:
            str: Formatted search context for the synthesis prompt
        """
        if not search_results:
            return "No search results available."
        
        context_parts = ["Here are the top search results:\n"]
        
        for result in search_results:
            position = result.get("position", "?")
            title = result.get("title", "No title")
            link = result.get("link", "")
            snippet = result.get("snippet", "No description")
            
            context_parts.append(
                f"\n[{position}] {title}\n"
                f"URL: {link}\n"
                f"Description: {snippet}\n"
            )
        
        return "".join(context_parts)
    
    def _build_conversation_context(self, conversation_history: List[Dict[str, str]]) -> str:
        """Build conversation context from message history.
        
        Args:
            conversation_history: List of previous messages
            
        Returns:
            str: Formatted conversation context
        """
        if not conversation_history:
            return ""
        
        context_parts = ["Previous conversation:\n"]
        
        # Include last few messages for context (limit to avoid token overflow)
        recent_messages = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
        
        for msg in recent_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            context_parts.append(f"{role.capitalize()}: {content}\n")
        
        return "".join(context_parts)
    
    def _create_synthesis_prompt(
        self,
        query: str,
        optimized_query: str,
        search_context: str,
        conversation_context: str
    ) -> str:
        """Create the synthesis prompt for the Base LLM.
        
        Args:
            query: The user's original query
            optimized_query: The optimized search query used
            search_context: Formatted search results
            conversation_context: Formatted conversation history
            
        Returns:
            str: Complete synthesis prompt
        """
        prompt_parts = []
        
        # Add conversation context if available
        if conversation_context:
            prompt_parts.append(conversation_context)
            prompt_parts.append("\n---\n")
        
        # Add search context
        prompt_parts.append(search_context)
        prompt_parts.append("\n---\n")
        
        # Simple, minimal prompt - request plain text output
        prompt_parts.append(
            f"Using the search results above, answer this question: {query}\n\n"
            f"Important: Write in plain text only. Do not use any markdown formatting (no *, **, #, bullets, etc.). "
            f"Write naturally as if speaking. Cite sources using [1], [2], etc.\n\n"
            f"Answer:"
        )
        
        return "".join(prompt_parts)
    
    async def _stream_from_ollama(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream response from Ollama Base LLM.
        
        Args:
            prompt: The complete prompt to send to Ollama
            
        Yields:
            str: Response chunks from the LLM
        """
        import httpx
        
        ollama_url = self._get_ollama_url()
        model_name = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
        
        # Prepare request payload
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": True
        }
        
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                async with client.stream(
                    "POST",
                    f"{ollama_url}/api/generate",
                    json=payload
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                import json
                                chunk_data = json.loads(line)
                                
                                if "response" in chunk_data:
                                    yield chunk_data["response"]
                                
                                # Check if generation is complete
                                if chunk_data.get("done", False):
                                    break
                                    
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse Ollama response chunk: {line}")
                                continue
                                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error streaming from Ollama: {e}")
            raise
        except Exception as e:
            logger.error(f"Error streaming from Ollama: {e}", exc_info=True)
            raise
    
    async def _fallback_to_base_llm(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        """Fallback to Base LLM when web search fails.
        
        Args:
            query: The user's query
            conversation_history: Optional conversation context
            
        Yields:
            str: Response chunks from Base LLM
        """
        logger.info("Using Base LLM fallback (no web search)")
        
        # Build simple prompt without search results
        prompt_parts = []
        
        if conversation_history:
            conversation_context = self._build_conversation_context(conversation_history)
            prompt_parts.append(conversation_context)
            prompt_parts.append("\n---\n")
        
        prompt_parts.append(f"User: {query}\n\nAssistant: ")
        
        fallback_prompt = "".join(prompt_parts)
        
        async for chunk in self._stream_from_ollama(fallback_prompt):
            yield chunk

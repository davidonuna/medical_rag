# app/llm/ollama_client.py

"""
Async Ollama client for text generation and embeddings.
Provides retry logic, backoff, and connection pooling.
"""

import asyncio
from typing import List
import aiohttp


class OllamaClient:
    """
    Async Ollama client for text generation and embeddings.
    
    Features:
        - Automatic session reuse for connection pooling
        - Retry logic with exponential backoff
        - Configurable timeout
        - Support for both generation and embedding endpoints
    
    Example:
        client = OllamaClient("http://localhost:11434")
        response = await client.generate("llama3.2:1b", "Your prompt")
        embeddings = await client.embed(["text1", "text2"], "nomic-embed-text")
    """

    def __init__(
        self,
        host: str,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        timeout: int = 300,
    ):
        """
        Initialize the Ollama client.
        
        Args:
            host: Base URL for Ollama API (e.g., "http://localhost:11434")
            max_retries: Maximum number of retry attempts (default: 3)
            retry_backoff: Exponential backoff multiplier (default: 2.0)
            timeout: Request timeout in seconds (default: 300)
        """
        self.base_url = host.rstrip("/")
        self._session: aiohttp.ClientSession | None = None
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    # -----------------------------
    # SESSION MANAGEMENT
    # -----------------------------
    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create an aiohttp session.
        Reuses existing session for connection pooling.
        
        Returns:
            aiohttp.ClientSession: Active HTTP session
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                raise_for_status=False,
            )
        return self._session

    async def close(self):
        """
        Close the HTTP session gracefully.
        Should be called on application shutdown.
        """
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    # -----------------------------
    # TEXT GENERATION
    # -----------------------------
    async def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate text completion using Ollama API.
        
        Args:
            model: Model name (e.g., "llama3.2:1b")
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate (default: 512)
            temperature: Sampling temperature (default: 0.0)
            
        Returns:
            Generated text string
            
        Raises:
            RuntimeError: If all retries fail
        """
        url = f"{self.base_url}/v1/completions"
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_exception = None

        # Retry loop with exponential backoff
        for attempt in range(1, self.max_retries + 2):
            try:
                session = await self._get_session()
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(
                            f"Ollama generate failed (status={resp.status}): {text}"
                        )

                    data = await resp.json()
                    return data["choices"][0]["text"].strip()

            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                last_exception = e
                if attempt <= self.max_retries:
                    await asyncio.sleep(self.retry_backoff ** attempt)

        raise RuntimeError(
            f"Ollama generate failed after retries: {last_exception}"
        )

    # -----------------------------
    # EMBEDDINGS
    # -----------------------------
    async def embed(self, texts: List[str], model: str) -> List[List[float]]:
        """
        Generate embeddings for text(s) using Ollama API.
        
        Args:
            texts: List of text strings to embed
            model: Embedding model name (e.g., "nomic-embed-text")
            
        Returns:
            List of embedding vectors (one per input text)
            
        Raises:
            RuntimeError: If all retries fail
        """
        url = f"{self.base_url}/v1/embeddings"
        payload = {"model": model, "input": texts}

        last_exception = None

        # Retry loop with exponential backoff
        for attempt in range(1, self.max_retries + 2):
            try:
                session = await self._get_session()
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(
                            f"Ollama embed failed (status={resp.status}): {text}"
                        )

                    data = await resp.json()
                    return [d["embedding"] for d in data["data"]]

            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                last_exception = e
                if attempt <= self.max_retries:
                    await asyncio.sleep(self.retry_backoff ** attempt)

        raise RuntimeError(
            f"Ollama embed failed after retries: {last_exception}"
        )

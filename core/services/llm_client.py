# core/services/llm_client.py

import logging
import json # For hashing complex objects like message lists
import hashlib # For creating cache keys
from typing import List, Dict, Optional, Any
from openai import AsyncOpenAI, OpenAIError # Use official library for OpenRouter compatibility
from lru import LRU # Simple LRU cache implementation

# Import configuration centrally
try:
    from config import (
        OPENROUTER_API_KEY,
        OPENROUTER_MODEL_NAME,
        OPENROUTER_SITE_URL,
        OPENROUTER_APP_NAME,
        LLM_CACHE_SIZE
    )
except ImportError:
     logger.error("Failed to import LLM config. Using defaults / disabling.")
     OPENROUTER_API_KEY = None
     OPENROUTER_MODEL_NAME = "openai/gpt-4o"
     OPENROUTER_SITE_URL = None
     OPENROUTER_APP_NAME = "BoutiqueAI"
     LLM_CACHE_SIZE = 100 # Default cache size

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client for interacting with Large Language Models via OpenRouter's
    OpenAI-compatible API endpoint. Includes simple LRU caching for responses.
    Handles prompt construction and response parsing.
    """

    def __init__(self):
        """
        Initializes the asynchronous OpenAI client configured for OpenRouter
        and sets up the LRU cache.
        """
        if not OPENROUTER_API_KEY:
            logger.error("OpenRouter API key is not configured. LLMClient cannot function.")
            raise ValueError("OPENROUTER_API_KEY is required.")

        # Prepare optional headers for OpenRouter identification
        self.default_headers: Dict[str, str] = {}
        if OPENROUTER_SITE_URL:
            self.default_headers["HTTP-Referer"] = OPENROUTER_SITE_URL
        if OPENROUTER_APP_NAME:
            self.default_headers["X-Title"] = OPENROUTER_APP_NAME

        try:
            # Initialize the AsyncOpenAI client pointing to OpenRouter
            self.client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
                default_headers=self.default_headers if self.default_headers else None,
                max_retries=3,
                timeout=60.0
            )
            logger.info(f"LLMClient initialized for OpenRouter. Default model: {OPENROUTER_MODEL_NAME}")
            if self.default_headers: logger.debug(f"Using default headers: {self.default_headers}")

            # Initialize LRU cache
            self.cache_size = LLM_CACHE_SIZE
            self.response_cache = LRU(self.cache_size)
            logger.info(f"Initialized LLM response cache with size {self.cache_size}.")

        except Exception as e:
            logger.error(f"Failed to initialize AsyncOpenAI client for OpenRouter: {e}", exc_info=True)
            raise ConnectionError("Could not initialize LLMClient for OpenRouter.") from e

    def _generate_cache_key(self, model: str, messages: List[Dict[str, str]], temperature: float) -> str:
        """ Generates a stable cache key based on model, messages, and temperature. """
        try:
            # Serialize messages deterministically and include model/temp
            # Using json.dumps with sort_keys ensures order doesn't matter in dicts
            # Using separators removes whitespace variations
            payload_str = json.dumps({"model": model, "messages": messages, "temp": temperature}, sort_keys=True, separators=(',', ':'))
            # Hash the string representation
            return hashlib.sha256(payload_str.encode('utf-8')).hexdigest()
        except TypeError as e:
            logger.warning(f"Could not generate cache key due to non-serializable data: {e}. Skipping cache.")
            return None # Cannot cache if key generation fails

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: str = OPENROUTER_MODEL_NAME, # Allow overriding default model
        temperature: float = 0.7,
        max_tokens: int = 300,
        use_cache: bool = True # Allow bypassing cache if needed
        ) -> Optional[str]:
        """
        Generates a response from the configured LLM via OpenRouter,
        utilizing an LRU cache to avoid redundant API calls.

        Args:
            messages: List of message dictionaries (OpenAI format).
            model: Specific OpenRouter model ID.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            use_cache: Whether to check and store results in the cache.

        Returns:
            The generated text content string, or None if an error occurs.
        """
        cache_key = None
        if use_cache:
            cache_key = self._generate_cache_key(model, messages, temperature)
            if cache_key and cache_key in self.response_cache:
                logger.info(f"LLM Cache HIT for model '{model}'. Returning cached response.")
                return self.response_cache[cache_key]
            elif cache_key:
                 logger.info(f"LLM Cache MISS for model '{model}'.")
            # Proceed to API call if key generation failed or cache miss

        logger.info(f"Generating LLM response via OpenRouter API (Model: '{model}')...")
        logger.debug(f"Input messages: {messages}") # Careful with sensitive data

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if response.choices and response.choices[0].message and response.choices[0].message.content:
                ai_response = response.choices[0].message.content.strip()
                logger.info(f"Successfully received LLM response (length: {len(ai_response)} chars).")
                # Store in cache if caching is enabled and key was generated
                if use_cache and cache_key:
                    self.response_cache[cache_key] = ai_response
                    logger.debug(f"Stored response in cache with key {cache_key[:8]}...")
                return ai_response
            else:
                logger.warning("LLM response received, but content was empty or malformed.")
                logger.debug(f"Full LLM API response: {response}")
                return None

        except OpenAIError as e:
            logger.error(f"OpenRouter API error: {getattr(e, 'status_code', 'N/A')} - {getattr(e.response, 'text', str(e))}", exc_info=False)
            # Specific error handling
            status_code = getattr(e, 'status_code', None)
            response_text = getattr(e.response, 'text', '')
            if status_code == 402: return "Error: Service unavailable due to billing issue."
            if status_code == 429: return "Error: Service temporarily overloaded."
            if status_code == 400 and "moderation" in response_text.lower(): return "Error: Content policy violation."
            return None # General API error
        except Exception as e:
            logger.error(f"Unexpected error during LLM response generation: {e}", exc_info=True)
            return None

# (Conceptual Test Runner - Keep as is)
async def main():
    print("Testing LLMClient with Cache...")
    try:
        client = LLMClient()
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        print("\nFirst call (should hit API):")
        response1 = await client.generate_response(messages=test_messages)
        if response1: print(f"Response 1: {response1}")
        else: print("Failed to get response 1.")

        print("\nSecond call (should hit cache):")
        response2 = await client.generate_response(messages=test_messages)
        if response2: print(f"Response 2: {response2}")
        else: print("Failed to get response 2.")

        print("\nThird call (different temp, should hit API):")
        response3 = await client.generate_response(messages=test_messages, temperature=0.9)
        if response3: print(f"Response 3: {response3}")
        else: print("Failed to get response 3.")

    except Exception as e:
        print(f"\nError during test: {e}")

if __name__ == "__main__":
    # import asyncio
    # import config # Ensure config is loaded
    # asyncio.run(main()) # Uncomment to run test (requires async context and valid OpenRouter key)
    print("LLMClient structure defined (with Cache). Run test manually in an async context.")


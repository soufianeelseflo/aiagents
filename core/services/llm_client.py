# core/services/llm_client.py

import logging
from typing import List, Dict, Optional, Any
from openai import AsyncOpenAI, OpenAIError # Use official library for OpenRouter compatibility

# Import configuration centrally
from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL_NAME,
    OPENROUTER_SITE_URL,
    OPENROUTER_APP_NAME
)

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client for interacting with Large Language Models via OpenRouter's
    OpenAI-compatible API endpoint. Handles prompt construction and response parsing.
    """

    def __init__(self):
        """
        Initializes the asynchronous OpenAI client configured for OpenRouter.
        """
        if not OPENROUTER_API_KEY:
            logger.error("OpenRouter API key is not configured in environment variables.")
            raise ValueError("OPENROUTER_API_KEY is required.")

        # Prepare optional headers for OpenRouter identification
        self.default_headers: Dict[str, str] = {}
        if OPENROUTER_SITE_URL:
            self.default_headers["HTTP-Referer"] = OPENROUTER_SITE_URL
            logger.debug(f"Setting OpenRouter HTTP-Referer header to: {OPENROUTER_SITE_URL}")
        if OPENROUTER_APP_NAME:
            self.default_headers["X-Title"] = OPENROUTER_APP_NAME
            logger.debug(f"Setting OpenRouter X-Title header to: {OPENROUTER_APP_NAME}")

        try:
            # Initialize the AsyncOpenAI client pointing to OpenRouter
            self.client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
                # Pass headers only if they are actually set
                default_headers=self.default_headers if self.default_headers else None,
                max_retries=3, # Add some basic retry logic
                timeout=60.0 # Set a reasonable timeout for API calls
            )
            logger.info(f"LLMClient initialized for OpenRouter. Default model: {OPENROUTER_MODEL_NAME}")
            logger.debug(f"Using OpenRouter Base URL: https://openrouter.ai/api/v1")
            if self.default_headers:
                 logger.debug(f"Using default headers: {self.default_headers}")

        except Exception as e:
            logger.error(f"Failed to initialize AsyncOpenAI client for OpenRouter: {e}", exc_info=True)
            raise ConnectionError("Could not initialize LLMClient for OpenRouter.") from e

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: str = OPENROUTER_MODEL_NAME, # Allow overriding default model
        temperature: float = 0.7,
        max_tokens: int = 300 # Increased default slightly for potentially more complex sales talk
        ) -> Optional[str]:
        """
        Generates a response from the configured LLM via OpenRouter.

        Args:
            messages: A list of message dictionaries (e.g., [{"role": "user", "content": "..."}]).
                      Must follow OpenAI chat completion format.
            model: The specific OpenRouter model ID to use (e.g., "openai/gpt-4o").
            temperature: Sampling temperature.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            The generated text content as a string, or None if an error occurs.
        """
        logger.info(f"Generating LLM response using model '{model}' via OpenRouter...")
        logger.debug(f"Input messages: {messages}") # Be careful logging full prompts in production if sensitive

        try:
            # Make the asynchronous API call to OpenRouter
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                # stream=False # Not using streaming here for simplicity, SalesAgent handles TTS stream
            )

            # Extract the response content
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                ai_response = response.choices[0].message.content.strip()
                logger.info(f"Successfully received LLM response (length: {len(ai_response)} chars).")
                logger.debug(f"LLM Raw Choice: {response.choices[0]}") # Log choice details for debugging
                return ai_response
            else:
                logger.warning("LLM response received, but content was empty or malformed.")
                logger.debug(f"Full LLM API response: {response}")
                return None

        except OpenAIError as e: # Catch specific OpenAI library errors (which apply to OpenRouter calls)
            logger.error(f"OpenRouter API error: {e.status_code} - {e.response.text}", exc_info=False) # Log API error details
            # Handle specific common errors gracefully if possible
            if e.status_code == 402: # Payment Required
                 logger.error("OpenRouter returned 402 Payment Required. Check credits/billing.")
                 # Potentially raise a specific exception or return a specific error message
                 return "Error: Service unavailable due to billing issue."
            elif e.status_code == 429: # Rate Limit Exceeded
                 logger.warning("OpenRouter rate limit exceeded. Consider backoff/retry.")
                 return "Error: Service temporarily overloaded."
            elif e.status_code == 400 and "moderation" in e.response.text.lower():
                 logger.warning("OpenRouter request flagged by moderation.")
                 return "Error: Content policy violation."
            # Add more specific error handling as needed
            return None # General API error
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM response generation: {e}", exc_info=True)
            return None

# Example Usage (Conceptual - requires running event loop)
async def main():
    print("Testing LLMClient...")
    try:
        client = LLMClient()
        test_messages = [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Explain the concept of API resourcefulness in one sentence."}
        ]
        response = await client.generate_response(messages=test_messages)
        if response:
            print("\nLLM Response:")
            print(response)
        else:
            print("\nFailed to get LLM response.")
    except Exception as e:
        print(f"\nError during test: {e}")

if __name__ == "__main__":
    # This check prevents running the async main function automatically when imported.
    # To run this test, you would need to uncomment the line below
    # or run it from an async context (e.g., inside an existing event loop).
    # asyncio.run(main())
    print("LLMClient structure defined. Run test manually in an async context.")

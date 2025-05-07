# core/services/llm_client.py

import logging
import json
import hashlib
import base64 # For encoding images
from typing import List, Dict, Optional, Any, Union
from openai import AsyncOpenAI, OpenAIError, APIResponse # Use official library
from openai.types.chat import ChatCompletion # For type hinting response
from lru import LRU

import config # Assumes config.py is at project root

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client for interacting with Large Language Models via OpenRouter's
    OpenAI-compatible API endpoint. Includes LRU caching and support for multi-modal inputs (text & images).
    """

    def __init__(self):
        if not config.OPENROUTER_API_KEY:
            logger.critical("CRITICAL: OPENROUTER_API_KEY is not configured. LLMClient cannot function.")
            raise ValueError("OPENROUTER_API_KEY is required.")

        self.default_headers: Dict[str, str] = {}
        if config.OPENROUTER_SITE_URL:
            self.default_headers["HTTP-Referer"] = config.OPENROUTER_SITE_URL
        if config.OPENROUTER_APP_NAME:
            self.default_headers["X-Title"] = config.OPENROUTER_APP_NAME

        try:
            self.client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1", # Standard OpenRouter base URL
                api_key=config.OPENROUTER_API_KEY,
                default_headers=self.default_headers if self.default_headers else None,
                max_retries=config.get_int_env_var("OPENROUTER_MAX_RETRIES", default=2, required=False), # Configurable retries
                timeout=config.get_int_env_var("OPENROUTER_TIMEOUT_SECONDS", default=120, required=False) # Configurable timeout
            )
            logger.info(f"LLMClient initialized for OpenRouter. Default model: {config.OPENROUTER_MODEL_NAME}")
            if self.default_headers: logger.debug(f"Using default headers: {self.default_headers}")

            self.response_cache = LRU(config.LLM_CACHE_SIZE)
            logger.info(f"Initialized LLM response cache with size {config.LLM_CACHE_SIZE}.")

        except Exception as e:
            logger.critical(f"Failed to initialize AsyncOpenAI client for OpenRouter: {e}", exc_info=True)
            raise ConnectionError("Could not initialize LLMClient for OpenRouter.") from e

    def _generate_cache_key(
        self,
        model: str,
        messages: List[Dict[str, Any]], # Messages can now be more complex for multi-modal
        temperature: float,
        max_tokens: Optional[int]
    ) -> str:
        """Generates a stable cache key based on model, messages, temperature, and max_tokens."""
        # For multi-modal, messages can contain image data. Hashing large image data directly
        # in the key might be inefficient. Consider hashing a representation (e.g., image checksums or URIs if stable).
        # For simplicity here, we'll serialize the whole message structure.
        # If images are passed as raw bytes, this could make keys very long / memory intensive.
        # A better approach for caching with images might involve hashing the image bytes separately
        # and using those hashes in the cache key payload.
        try:
            # Create a serializable representation of messages, especially for images
            serializable_messages = []
            for msg in messages:
                s_msg = msg.copy()
                if isinstance(s_msg.get("content"), list): # Multi-part content
                    s_content_list = []
                    for part in s_msg["content"]:
                        s_part = part.copy()
                        if part.get("type") == "image_url" and isinstance(part.get("image_url"), dict):
                            img_url_data = part["image_url"].get("url", "")
                            # If it's a base64 string, hash it instead of including the whole string
                            if img_url_data.startswith("data:image"):
                                s_part["image_url"]["url_hash"] = hashlib.sha256(img_url_data.encode('utf-8')).hexdigest()
                                del s_part["image_url"]["url"] # Remove potentially large base64 string from key
                        s_content_list.append(s_part)
                    s_msg["content"] = s_content_list
                serializable_messages.append(s_msg)

            payload_str = json.dumps(
                {"model": model, "messages": serializable_messages, "temp": temperature, "max_tokens": max_tokens},
                sort_keys=True, separators=(',', ':')
            )
            return hashlib.sha256(payload_str.encode('utf-8')).hexdigest()
        except TypeError as e:
            logger.warning(f"Could not generate cache key due to non-serializable data: {e}. Skipping cache.")
            return None # Cannot cache if key generation fails

    async def generate_response(
        self,
        messages: List[Dict[str, Any]], # Content can be string or list for multi-modal
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 1024, # Default max_tokens
        use_cache: bool = True,
        # Multi-modal specific parameters (simplified for this example)
        # A more robust way is to include images directly in the 'messages' structure
        # following OpenAI's multi-modal message format.
        # This 'images_base64' is a helper if you want to pass them separately and have this client format them.
        images_base64: Optional[List[str]] = None # List of base64 encoded image strings
    ) -> Optional[str]:
        """
        Generates a response from the LLM, supporting multi-modal inputs (text and images).

        Args:
            messages: List of message dictionaries. For multi-modal, the 'content' of a message
                      can be a list of parts (text and image_url).
                      Example multi-modal message content:
                      [
                          {"type": "text", "text": "What's in this image?"},
                          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
                      ]
            model: Specific OpenRouter model ID. Defaults to config.OPENROUTER_MODEL_NAME.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            use_cache: Whether to use the cache.
            images_base64: Optional list of base64 encoded image strings. If provided,
                           and the last user message in `messages` is text-only, these images
                           will be appended to that last user message's content.

        Returns:
            The generated text content string, or None if an error occurs.
        """
        final_model = model or config.OPENROUTER_MODEL_NAME
        
        # --- Prepare messages for multi-modal if images_base64 is provided ---
        # This is a helper to inject images into the last user message if it's simple text.
        # The preferred way is to construct the multi-modal `messages` list directly.
        processed_messages = [msg.copy() for msg in messages] # Work on a copy
        if images_base64:
            if not processed_messages or processed_messages[-1]["role"] != "user":
                logger.warning("images_base64 provided, but no final user message to append them to. Ignoring images.")
            else:
                last_user_message = processed_messages[-1]
                if isinstance(last_user_message["content"], str): # Simple text content
                    new_content = [{"type": "text", "text": last_user_message["content"]}]
                    for img_b64 in images_base64:
                        if not img_b64.startswith("data:image"):
                            # Assume JPEG if no prefix, adjust if other types are common
                            img_b64 = f"data:image/jpeg;base64,{img_b64}"
                        new_content.append({"type": "image_url", "image_url": {"url": img_b64}})
                    last_user_message["content"] = new_content
                    logger.info(f"Appended {len(images_base64)} images to the last user message for multi-modal input.")
                elif isinstance(last_user_message["content"], list): # Already multi-part
                    for img_b64 in images_base64:
                        if not img_b64.startswith("data:image"):
                            img_b64 = f"data:image/jpeg;base64,{img_b64}"
                        last_user_message["content"].append({"type": "image_url", "image_url": {"url": img_b64}})
                    logger.info(f"Added {len(images_base64)} images to existing multi-part user message.")
                else:
                    logger.warning("Last user message content is not string or list, cannot append images_base64.")
        # --- End multi-modal message preparation ---

        cache_key = None
        if use_cache:
            cache_key = self._generate_cache_key(final_model, processed_messages, temperature, max_tokens)
            if cache_key and cache_key in self.response_cache:
                logger.info(f"LLM Cache HIT for model '{final_model}'. Returning cached response.")
                return self.response_cache[cache_key]
            elif cache_key:
                 logger.info(f"LLM Cache MISS for model '{final_model}'. Key: {cache_key[:10]}...")

        logger.info(f"Generating LLM response via OpenRouter (Model: '{final_model}', Temp: {temperature}, MaxTokens: {max_tokens}).")
        # Avoid logging full messages if they contain large base64 images
        log_messages_summary = []
        for msg in processed_messages:
            summary_msg = {"role": msg["role"]}
            if isinstance(msg.get("content"), str):
                summary_msg["content_summary"] = msg["content"][:70] + "..." if len(msg["content"]) > 70 else msg["content"]
            elif isinstance(msg.get("content"), list):
                summary_msg["content_summary"] = f"{len(msg['content'])} parts (text/image)"
            log_messages_summary.append(summary_msg)
        logger.debug(f"Input messages (summary): {log_messages_summary}")


        try:
            # Type casting for messages to satisfy OpenAI SDK strictness if needed,
            # though List[Dict[str, Any]] should generally work if content is structured correctly.
            chat_completion_params = {
                "model": final_model,
                "messages": processed_messages, # type: ignore
                "temperature": temperature,
            }
            if max_tokens is not None: # max_tokens is optional for some models/APIs
                chat_completion_params["max_tokens"] = max_tokens

            response: ChatCompletion = await self.client.chat.completions.create(**chat_completion_params) # type: ignore

            if response.choices and response.choices[0].message and response.choices[0].message.content:
                ai_response_text = response.choices[0].message.content.strip()
                logger.info(f"LLM response received (length: {len(ai_response_text)} chars). Finish reason: {response.choices[0].finish_reason}")
                logger.debug(f"LLM Raw Response Text: {ai_response_text[:500]}")

                if use_cache and cache_key:
                    self.response_cache[cache_key] = ai_response_text
                    logger.debug(f"Stored response in cache with key {cache_key[:10]}...")
                return ai_response_text
            else:
                logger.warning("LLM response received, but content was empty or malformed.")
                logger.debug(f"Full LLM API response object: {response.model_dump_json(indent=2)}")
                return None

        except OpenAIError as e: # Catch specific OpenAI/OpenRouter errors
            status_code = getattr(e, 'status_code', 'N/A')
            error_body = getattr(e, 'body', {}) or {} # body might be None
            error_message = error_body.get('error', {}).get('message', str(e)) if isinstance(error_body.get('error'), dict) else str(e)
            
            logger.error(f"OpenRouter API error: Status {status_code} - Message: {error_message}", exc_info=False) # Set exc_info=False if error message is clear
            logger.debug(f"Full OpenRouter APIError details: {e}")
            # Provide more user-friendly messages for common errors
            if status_code == 401: return "Error: Authentication failed. Check your OpenRouter API key."
            if status_code == 402: return "Error: OpenRouter quota exceeded or billing issue."
            if status_code == 429: return "Error: Rate limit exceeded or model overloaded. Try again later."
            if status_code == 400 and "moderation" in error_message.lower(): return "Error: Request blocked due to content moderation."
            if "context_length_exceeded" in error_message.lower(): return "Error: Input is too long for the selected model."
            return f"Error: LLM API request failed (Status: {status_code})." # General API error
        except Exception as e:
            logger.error(f"Unexpected error during LLM response generation: {e}", exc_info=True)
            return "Error: An unexpected error occurred while contacting the LLM."

# --- Test function ---
async def _test_llm_client_multimodal():
    print("--- Testing LLMClient (Multi-Modal Focus) ---")
    
    # Ensure OPENROUTER_API_KEY is in .env for this test
    if not config.OPENROUTER_API_KEY:
        print("Skipping LLMClient test: OPENROUTER_API_KEY not found in .env.")
        return

    llm = LLMClient()

    # Test 1: Simple text prompt
    print("\n1. Testing simple text prompt...")
    text_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France in one word?"}
    ]
    response1 = await llm.generate_response(messages=text_messages, model="mistralai/mistral-7b-instruct", max_tokens=10) # Use a fast model
    print(f"  Response 1 (Text): {response1}")

    # Test 2: Multi-modal prompt (requires a model that supports vision, e.g., gpt-4-vision-preview or a Gemini vision model)
    # You will need to select an appropriate multi-modal model on OpenRouter and set it in your .env
    # or pass it directly to generate_response.
    # Create a tiny dummy base64 PNG image (1x1 red pixel)
    # In a real scenario, this would be a screenshot from Playwright or other source.
    # (This is a 1x1 red PNG: iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=)
    dummy_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    
    # Option A: Using the images_base64 helper (simpler to call)
    print("\n2a. Testing multi-modal with images_base64 helper...")
    multimodal_messages_option_a = [
        {"role": "user", "content": "What color is the tiny square in the image?"}
    ]
    # IMPORTANT: Replace with an actual multi-modal model available on OpenRouter that you have access to.
    # E.g., "openai/gpt-4o", "google/gemini-pro-vision" (check exact model ID on OpenRouter)
    # If OPENROUTER_MODEL_NAME in your .env is already a vision model, you don't need to specify 'model' here.
    # For this test, let's assume config.OPENROUTER_MODEL_NAME is set to a vision model.
    vision_model_name = config.OPENROUTER_MODEL_NAME # Use the one from config
    if "vision" not in vision_model_name.lower() and "gpt-4o" not in vision_model_name.lower() and "gemini" not in vision_model_name.lower(): # Basic check
        print(f"  WARNING: Default model '{vision_model_name}' might not be multi-modal. Test may fail or ignore image.")
        print(f"  Please set OPENROUTER_MODEL_NAME to a vision model like 'openai/gpt-4o' or 'google/gemini-pro-vision' for this test.")


    response2a = await llm.generate_response(
        messages=multimodal_messages_option_a,
        images_base64=[dummy_image_base64],
        model=vision_model_name, # Explicitly use a vision model
        max_tokens=50
    )
    print(f"  Response 2a (Multi-modal with helper): {response2a}")

    # Option B: Constructing the multi-modal message structure directly
    print("\n2b. Testing multi-modal with direct message structure...")
    multimodal_messages_option_b = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the object in this image. It's very small."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{dummy_image_base64}"}}
            ]
        }
    ]
    response2b = await llm.generate_response(
        messages=multimodal_messages_option_b,
        model=vision_model_name, # Explicitly use a vision model
        max_tokens=50
    )
    print(f"  Response 2b (Multi-modal direct structure): {response2b}")

    # Test 3: Caching (second call to the same text prompt)
    print("\n3. Testing caching (second call to text prompt)...")
    response3 = await llm.generate_response(messages=text_messages, model="mistralai/mistral-7b-instruct", max_tokens=10)
    print(f"  Response 3 (Text, cached): {response3}")


if __name__ == "__main__":
    # To run this test:
    # 1. Ensure async environment.
    # 2. `config.py` accessible.
    # 3. `.env` with OPENROUTER_API_KEY.
    # 4. For multi-modal tests to work well, ensure OPENROUTER_MODEL_NAME in .env
    #    is set to a model that supports vision (e.g., "openai/gpt-4o", "google/gemini-pro-vision").
    # Example:
    # import asyncio
    # load_dotenv() # Ensure .env is loaded if running this file directly
    # asyncio.run(_test_llm_client_multimodal())
    print("LLMClient (Multi-Modal Support) defined. Run test manually with proper .env setup.")
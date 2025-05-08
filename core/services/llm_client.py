# boutique_ai_project/core/services/llm_client.py

import logging
import json
import hashlib
import base64
from typing import List, Dict, Optional, Any, Union
from openai import AsyncOpenAI, OpenAIError, APIResponse
from openai.types.chat import ChatCompletion
from lru import LRU

import config # Root config

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client for OpenRouter OpenAI-compatible API. Includes caching and multi-modal support. (Level 45)
    """

    def __init__(self):
        if not config.OPENROUTER_API_KEY:
            logger.critical("CRITICAL: OPENROUTER_API_KEY not configured.")
            raise ValueError("OPENROUTER_API_KEY is required.")

        self.default_headers: Dict[str, str] = {}
        if config.OPENROUTER_SITE_URL: self.default_headers["HTTP-Referer"] = config.OPENROUTER_SITE_URL
        if config.OPENROUTER_APP_NAME: self.default_headers["X-Title"] = config.OPENROUTER_APP_NAME

        try:
            self.client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=config.OPENROUTER_API_KEY,
                default_headers=self.default_headers if self.default_headers else None,
                max_retries=config.OPENROUTER_CLIENT_MAX_RETRIES,
                timeout=config.OPENROUTER_CLIENT_TIMEOUT_SECONDS
            )
            logger.info(f"LLMClient initialized for OpenRouter. Default Conv Model: {config.OPENROUTER_DEFAULT_CONVERSATIONAL_MODEL}")
            if self.default_headers: logger.debug(f"Using default headers: {self.default_headers}")

            self.response_cache = LRU(config.LLM_CACHE_SIZE)
            logger.info(f"Initialized LLM response cache with size {config.LLM_CACHE_SIZE}.")
        except Exception as e:
            logger.critical(f"Failed to initialize AsyncOpenAI client for OpenRouter: {e}", exc_info=True)
            raise ConnectionError("Could not initialize LLMClient for OpenRouter.") from e

    def _generate_cache_key(
        self, model: str, messages: List[Dict[str, Any]], temperature: float, max_tokens: Optional[int]
    ) -> Optional[str]:
        """Generates cache key, hashing image data representations. Skips caching for image requests."""
        try:
            serializable_messages = []
            has_images = False
            for msg in messages:
                s_msg = msg.copy()
                if isinstance(s_msg.get("content"), list):
                    s_content_list = []
                    for part in s_msg["content"]:
                        s_part = part.copy()
                        if part.get("type") == "image_url" and isinstance(part.get("image_url"), dict):
                            has_images = True
                            img_url_data = part["image_url"].get("url", "")
                            if img_url_data.startswith("data:image"):
                                base64_content = img_url_data.split(",", 1)[-1]
                                s_part["image_url"]["url_hash"] = hashlib.sha256(base64_content.encode('utf-8')).hexdigest()
                                if "url" in s_part["image_url"]: del s_part["image_url"]["url"]
                        s_content_list.append(s_part)
                    s_msg["content"] = s_content_list
                serializable_messages.append(s_msg)
            
            if has_images: # Do not cache multi-modal requests by default
                logger.debug("Skipping cache key generation for multi-modal request.")
                return None

            payload_str = json.dumps(
                {"model": model, "messages": serializable_messages, "temp": temperature, "max_tokens": max_tokens},
                sort_keys=True, separators=(',', ':')
            )
            return hashlib.sha256(payload_str.encode('utf-8')).hexdigest()
        except TypeError as e:
            logger.warning(f"Could not generate cache key due to non-serializable data: {e}. Skipping cache.")
            return None

    async def generate_response(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: Optional[int] = 1500,
        use_cache: bool = True,
        purpose: str = "general" # 'general', 'strategy', 'analysis', 'vision'
    ) -> Optional[str]:
        """Generates LLM response, supports multi-modal, selects model based on purpose."""
        
        # Determine model
        if model: final_model = model
        elif purpose == "strategy": final_model = config.OPENROUTER_DEFAULT_STRATEGY_MODEL
        elif purpose == "analysis": final_model = config.OPENROUTER_DEFAULT_ANALYSIS_MODEL
        elif purpose == "vision": final_model = config.OPENROUTER_DEFAULT_VISION_MODEL
        else: final_model = config.OPENROUTER_DEFAULT_CONVERSATIONAL_MODEL

        is_multimodal_request = any(isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"]) for m in messages)

        if is_multimodal_request and purpose != "vision":
            logger.warning(f"Request contains images but purpose is '{purpose}'. Using VISION model '{config.OPENROUTER_DEFAULT_VISION_MODEL}' instead.")
            final_model = config.OPENROUTER_DEFAULT_VISION_MODEL
        
        if not final_model: # Check if a model was determined
             logger.error(f"No suitable LLM model found for purpose '{purpose}' (is vision model configured if needed?).")
             return f"Error: No suitable LLM model configured for purpose '{purpose}'."

        # Caching logic (skip cache for vision requests)
        cache_key = None
        effective_use_cache = use_cache and not is_multimodal_request
        if effective_use_cache:
            cache_key = self._generate_cache_key(final_model, messages, temperature, max_tokens)
            if cache_key and cache_key in self.response_cache:
                logger.info(f"LLM Cache HIT for model '{final_model}'.")
                return self.response_cache[cache_key]
            elif cache_key:
                 logger.info(f"LLM Cache MISS for model '{final_model}'.")

        logger.info(f"Generating LLM response via OpenRouter (Model: '{final_model}', Purpose: {purpose}, Temp: {temperature}, MaxTokens: {max_tokens}).")
        log_messages_summary = [{"role": msg["role"], "content_summary": f"{len(str(msg.get('content', '')))} chars/parts"} for msg in messages]
        logger.debug(f"Input messages (summary): {log_messages_summary}")

        try:
            chat_completion_params = { "model": final_model, "messages": messages, "temperature": temperature }
            if max_tokens is not None: chat_completion_params["max_tokens"] = max_tokens

            response: ChatCompletion = await self.client.chat.completions.create(**chat_completion_params) # type: ignore

            if response.choices and response.choices[0].message and response.choices[0].message.content:
                ai_response_text = response.choices[0].message.content.strip()
                finish_reason = response.choices[0].finish_reason
                logger.info(f"LLM response received ({len(ai_response_text)} chars). Finish: {finish_reason}")
                logger.debug(f"LLM Raw Response Text: {ai_response_text[:500]}")
                if finish_reason == "length": logger.warning(f"LLM response truncated due to max_tokens ({max_tokens}).")

                if effective_use_cache and cache_key:
                    self.response_cache[cache_key] = ai_response_text
                    logger.debug(f"Stored text-only response in cache.")
                return ai_response_text
            else:
                logger.warning("LLM response empty/malformed.")
                logger.debug(f"Full LLM API response: {response.model_dump_json(indent=2)}")
                return None
        except OpenAIError as e:
            status_code = getattr(e, 'status_code', 'N/A'); error_body = getattr(e, 'body', {}) or {}
            error_message = error_body.get('error', {}).get('message', str(e)) if isinstance(error_body.get('error'), dict) else str(e)
            logger.error(f"OpenRouter API error: Status {status_code} - Message: {error_message}", exc_info=False)
            if status_code == 401: return "Error: OpenRouter Authentication failed."
            if status_code == 402: return "Error: OpenRouter quota/billing issue."
            if status_code == 429: return "Error: OpenRouter rate limit/overload."
            if status_code == 400 and "moderation" in error_message.lower(): return "Error: Content moderation."
            if "context_length_exceeded" in error_message.lower(): return "Error: Input too long for model."
            return f"Error: LLM API failed (Status: {status_code})."
        except Exception as e:
            logger.error(f"Unexpected error during LLM generation: {e}", exc_info=True)
            return "Error: Unexpected LLM client error."

# --- Test function ---
async def _test_llm_client_final():
    print("--- Testing LLMClient (FINAL - Multi-Modal) ---")
    if not config.OPENROUTER_API_KEY: print("Skipping test: OPENROUTER_API_KEY not set."); return

    llm = LLMClient()
    dummy_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

    print("\n1. Testing Text (Strategy Model)...")
    text_messages = [{"role": "user", "content": "Suggest 3 strategies for cold outreach."}]
    response1 = await llm.generate_response(messages=text_messages, purpose="strategy", max_tokens=150)
    print(f"  Response 1 (Strategy): {response1}")

    print("\n2. Testing Vision Model...")
    vision_model = config.OPENROUTER_DEFAULT_VISION_MODEL
    if not vision_model: print("  Skipping vision test: No vision model configured."); return
    
    multimodal_messages = [{"role": "user", "content": [
                {"type": "text", "text": "What color is the tiny square in the image?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{dummy_image_base64}"}}
            ]}]
    # Explicitly use purpose='vision' to ensure correct model selection if default isn't vision
    response2 = await llm.generate_response(messages=multimodal_messages, purpose="vision", max_tokens=50)
    print(f"  Response 2 (Vision): {response2}")

    print("\n3. Testing Caching (Second call to text prompt)...")
    # Use same parameters as first call to test cache
    response3 = await llm.generate_response(messages=text_messages, purpose="strategy", max_tokens=150, use_cache=True)
    print(f"  Response 3 (Strategy, cached): {response3}")

if __name__ == "__main__":
    # import asyncio
    # from dotenv import load_dotenv
    # load_dotenv()
    # asyncio.run(_test_llm_client_final())
    print("LLMClient (FINAL - Multi-Modal) defined. Test requires OpenRouter key.")
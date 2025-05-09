# core/services/llm_client.py
import logging
import json
import hashlib
import base64
from typing import List, Dict, Optional, Any, Union
from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletion
from lru import LRU

import config # Root config

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client for OpenRouter OpenAI-compatible API. Includes caching and multi-modal support.
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
            logger.info(
                f"LLMClient initialized for OpenRouter. "
                f"Default Conv: {config.OPENROUTER_DEFAULT_CONVERSATIONAL_MODEL}, "
                f"Strategy: {config.OPENROUTER_DEFAULT_STRATEGY_MODEL}, "
                f"Analysis: {config.OPENROUTER_DEFAULT_ANALYSIS_MODEL}, "
                f"Vision: {config.OPENROUTER_DEFAULT_VISION_MODEL}"
            )
            if self.default_headers: logger.debug(f"Using default headers: {self.default_headers}")

            self.response_cache = LRU(config.LLM_CACHE_SIZE)
            logger.info(f"Initialized LLM response cache with size {config.LLM_CACHE_SIZE}.")

        except Exception as e:
            logger.critical(f"Failed to initialize AsyncOpenAI client for OpenRouter: {e}", exc_info=True)
            raise ConnectionError("Could not initialize LLMClient for OpenRouter.") from e

    def _generate_cache_key(
        self, model: str, messages: List[Dict[str, Any]], temperature: float, max_tokens: Optional[int]
    ) -> Optional[str]:
        try:
            serializable_messages = []
            has_images = False
            for msg in messages:
                s_msg = msg.copy()
                if isinstance(s_msg.get("content"), list):
                    s_content_list = []
                    for part in s_msg["content"]:
                        s_part_item = part.copy() # Use a different variable name here
                        if part.get("type") == "image_url" and isinstance(part.get("image_url"), dict):
                            has_images = True
                            # No need to hash image URLs for cache key if we're skipping cache for images
                        s_content_list.append(s_part_item)
                    s_msg["content"] = s_content_list
                serializable_messages.append(s_msg)

            if has_images: # Do not cache multi-modal requests by default
                logger.debug("Skipping cache key generation for request containing images.")
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
        purpose: str = "general"
    ) -> Optional[str]:
        final_model: Optional[str] = model
        is_multimodal_request = any(
            isinstance(m.get("content"), list) and
            any(p.get("type") == "image_url" for p in m["content"])
            for m in messages
        )

        if not final_model:
            if is_multimodal_request:
                final_model = config.OPENROUTER_DEFAULT_VISION_MODEL
                logger.info(f"Multimodal request detected. Using vision model: {final_model}")
                purpose = "vision" # Ensure purpose matches
            elif purpose == "strategy": final_model = config.OPENROUTER_DEFAULT_STRATEGY_MODEL
            elif purpose == "analysis": final_model = config.OPENROUTER_DEFAULT_ANALYSIS_MODEL
            else: final_model = config.OPENROUTER_DEFAULT_CONVERSATIONAL_MODEL

        if not final_model:
             logger.error(f"No suitable LLM model determined for purpose '{purpose}' or explicit model '{model}'. Check OpenRouter model name configurations.")
             return "Error: No suitable LLM model configured."

        cache_key = None
        effective_use_cache = use_cache and not is_multimodal_request

        if effective_use_cache:
            cache_key = self._generate_cache_key(final_model, messages, temperature, max_tokens)
            if cache_key and cache_key in self.response_cache:
                logger.info(f"LLM Cache HIT for model '{final_model}' (Purpose: {purpose}).")
                return self.response_cache[cache_key]
            elif cache_key:
                 logger.info(f"LLM Cache MISS for model '{final_model}' (Purpose: {purpose}).")

        logger.info(f"Generating LLM response (Model: '{final_model}', Purpose: {purpose}, Temp: {temperature}, MaxTokens: {max_tokens}, Cache: {effective_use_cache}).")
        
        # Truncate message content for logging if too long
        log_messages_summary = []
        for msg in messages:
            content_summary = str(msg.get('content', ''))
            if len(content_summary) > 100:
                content_summary = content_summary[:100] + "..."
            log_messages_summary.append({"role": msg["role"], "content_summary": content_summary})
        logger.debug(f"Input messages (summary): {log_messages_summary}")

        try:
            chat_completion_params: Dict[str, Any] = {
                "model": final_model, "messages": messages, "temperature": temperature
            }
            if max_tokens is not None: chat_completion_params["max_tokens"] = max_tokens

            response: ChatCompletion = await self.client.chat.completions.create(**chat_completion_params)

            if response.choices and response.choices[0].message and response.choices[0].message.content:
                ai_response_text = response.choices[0].message.content.strip()
                finish_reason = response.choices[0].finish_reason
                usage = response.usage
                logger.info(
                    f"LLM response received ({len(ai_response_text)} chars). Finish: {finish_reason}. "
                    f"Usage: Prompt={usage.prompt_tokens if usage else 'N/A'}, Completion={usage.completion_tokens if usage else 'N/A'}, Total={usage.total_tokens if usage else 'N/A'}"
                )
                if finish_reason == "length":
                    logger.warning(f"LLM response for model '{final_model}' was truncated due to max_tokens ({max_tokens}).")

                if effective_use_cache and cache_key:
                    self.response_cache[cache_key] = ai_response_text
                return ai_response_text
            else:
                logger.warning(f"LLM response from model '{final_model}' was empty/malformed. Full response: {response.model_dump_json(indent=2)}")
                return None
        except OpenAIError as e:
            status_code = getattr(e, 'status_code', 'N/A')
            error_body = getattr(e, 'body', {}) or {}
            error_message_detail = str(error_body.get('error', {}).get('message', str(e))) if isinstance(error_body.get('error'), dict) else str(e.body or e)
            logger.error(f"OpenRouter API error (Model: '{final_model}'): Status {status_code} - Message: {error_message_detail}")
            # Construct a user-friendly error message
            user_error_message = f"Error: LLM API failed (Model: {final_model}, Status: {status_code})."
            if status_code == 401: user_error_message = "Error: OpenRouter Authentication failed. Check API Key."
            elif status_code == 402: user_error_message = "Error: OpenRouter quota/billing issue."
            elif status_code == 429: user_error_message = "Error: OpenRouter rate limit/overload for model."
            elif "context_length_exceeded" in error_message_detail.lower(): user_error_message = f"Error: Input too long for model '{final_model}'."
            return user_error_message
        except Exception as e:
            logger.error(f"Unexpected error during LLM generation (Model: '{final_model}'): {e}", exc_info=True)
            return "Error: Unexpected LLM client error."

# (Test function remains similar, ensure it's commented out for deployment)
# if __name__ == "__main__":
#     import asyncio
#     from dotenv import load_dotenv, find_dotenv
#     if find_dotenv(): load_dotenv(find_dotenv(raise_error_if_not_found=False, usecwd=True))
#     async def _test_llm_client_final():
#         # ... (test logic as before) ...
#         pass
#     # asyncio.run(_test_llm_client_final())
#     print("LLMClient defined.")
# /core/services/llm_client.py: (Holistically Reviewed - May 9, 2025)
# --------------------------------------------------------------------------------
# boutique_ai_project/core/services/llm_client.py

import logging
import json
import time # For basic delay, though tenacity is preferred for retries
from typing import Dict, Any, Optional, List, Tuple, Callable, Coroutine
from openai import OpenAI, APIError, APITimeoutError, APIConnectionError, RateLimitError, APIStatusError
from lru import LRU # From lru-dict package
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import config # Root config

logger = logging.getLogger(__name__)

class LLMClientSetupError(Exception):
    """Custom exception for critical errors during LLMClient setup."""
    pass

class LLMResponseError(Exception):
    """Custom exception for errors in LLM response processing."""
    pass

class LLMClient:
    """
    Client for interacting with LLMs via OpenRouter, with caching,
    adaptive retries via tenacity, and improved error handling.
    """
    DEFAULT_MODEL = config.OPENROUTER_DEFAULT_CONVERSATIONAL_MODEL
    DEFAULT_TIMEOUT_SECONDS = config.OPENROUTER_CLIENT_TIMEOUT_SECONDS
    DEFAULT_MAX_RETRIES = config.OPENROUTER_CLIENT_MAX_RETRIES

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.OPENROUTER_API_KEY
        if not self.api_key:
            msg = "OpenRouter API key not found. Please set OPENROUTER_API_KEY."
            logger.critical(msg)
            raise LLMClientSetupError(msg)

        try:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
                timeout=self.DEFAULT_TIMEOUT_SECONDS,
                max_retries=0 # Retries will be handled by tenacity
            )
        except Exception as e:
            msg = f"Failed to initialize OpenAI client for OpenRouter: {e}"
            logger.critical(msg, exc_info=True)
            raise LLMClientSetupError(msg) from e

        self.cache_size = config.LLM_CACHE_SIZE
        self.cache: Optional[LRU] = LRU(self.cache_size) if self.cache_size > 0 else None
        
        log_msg = f"LLMClient initialized. Default model: {self.DEFAULT_MODEL}. "
        log_msg += f"Cache size: {self.cache_size if self.cache else 'Disabled'}."
        log_msg += f" Default Timeout: {self.DEFAULT_TIMEOUT_SECONDS}s. Default Max Retries (via Tenacity): {self.DEFAULT_MAX_RETRIES}."
        logger.info(log_msg)

    def _generate_cache_key(self, model: str, messages: List[Dict[str, str]], **kwargs) -> Tuple:
        try:
            # Include only relevant parts of messages for caching (role, content, tool_calls if present)
            frozen_messages_parts = []
            for m in messages:
                part = {"role": m.get("role"), "content": m.get("content")}
                if "tool_calls" in m and m["tool_calls"] is not None: # Handle tool calls for caching
                    part["tool_calls"] = tuple(
                        tuple(sorted(tc.items())) if isinstance(tc, dict) else tc for tc in m["tool_calls"]
                    )
                frozen_messages_parts.append(tuple(sorted(part.items())))
            
            frozen_messages = tuple(frozen_messages_parts)
            frozen_kwargs = tuple(sorted(kwargs.items()))
            return (model, frozen_messages, frozen_kwargs)
        except Exception as e:
            logger.warning(f"Error generating cache key (kwargs/messages may contain unhashable types): {e}. Caching may be skipped for this request.")
            # Fallback to a less precise but still somewhat unique key
            return (model, str(messages), str(kwargs))

    @retry(
        stop=stop_after_attempt(DEFAULT_MAX_RETRIES + 1), # +1 because first attempt is not a "retry"
        wait=wait_exponential(multiplier=1, min=1, max=10), # Exponential backoff: 1s, 2s, 4s, ... up to 10s
        retry=retry_if_exception_type((APITimeoutError, APIConnectionError, RateLimitError, APIStatusError)), # Retry on these specific OpenAI/HTTP errors
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying LLM request due to {type(retry_state.outcome.exception()).__name__}: {retry_state.outcome.exception()}. "
            f"Attempt {retry_state.attempt_number}/{LLMClient.DEFAULT_MAX_RETRIES + 1}. "
            f"Waiting {getattr(retry_state.next_action, 'sleep', 0):.2f}s before next attempt."
        )
    )
    async def _make_openai_api_call(self, method_name: str, **kwargs):
        """Makes an awaitable call to a method of the self.client object."""
        method_to_call = getattr(self.client.chat.completions, method_name)
        if not method_to_call:
            raise AttributeError(f"Method '{method_name}' not found on OpenAI client's chat.completions service.")
        return await method_to_call(**kwargs) # type: ignore

    async def get_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2048,
        json_mode: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None, # For function calling
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None, # For function calling
        **kwargs # For other parameters like top_p, presence_penalty, etc.
    ) -> Optional[Dict[str, Any]]:
        """
        Gets a chat completion. Returns the full API response dictionary or None on failure.
        Handles caching and retries using tenacity.
        """
        current_model = model or self.DEFAULT_MODEL
        request_start_time = time.monotonic()
        
        logger.debug(
            f"Requesting chat completion: model='{current_model}', temp={temperature}, max_tokens={max_tokens}, "
            f"json_mode={json_mode}, tools_present={bool(tools)}, tool_choice='{tool_choice}'"
        )

        # --- Caching ---
        cache_key = None
        if self.cache is not None:
            cache_kwargs = {
                "temperature": temperature, "max_tokens": max_tokens,
                "json_mode": json_mode, **kwargs
            }
            if tools: cache_kwargs["tools"] = json.dumps(tools, sort_keys=True) # Serialize for consistency
            if tool_choice: cache_kwargs["tool_choice"] = json.dumps(tool_choice, sort_keys=True)

            cache_key = self._generate_cache_key(current_model, messages, **cache_kwargs)
            if cache_key in self.cache:
                logger.info(f"Cache hit for model '{current_model}'. Returning cached response.")
                return self.cache[cache_key]

        # --- Prepare Request ---
        request_params: Dict[str, Any] = {
            "model": current_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        if json_mode:
            request_params["response_format"] = {"type": "json_object"}
        if tools:
            request_params["tools"] = tools
        if tool_choice:
            request_params["tool_choice"] = tool_choice
        
        try:
            completion = await self._make_openai_api_call("create", **request_params)
            duration = time.monotonic() - request_start_time

            if not completion: # Should be caught by retry or be an APIError
                logger.error(f"LLM call to '{current_model}' returned None after retries.")
                return None

            # Convert Pydantic model to dict for consistent handling and caching
            api_response_dict = completion.model_dump(mode='json') # mode='json' ensures Enums, etc., are serializable

            # Log usage and basic info
            usage = api_response_dict.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            
            logger.info(
                f"LLM call to '{current_model}' successful. Duration: {duration:.2f}s. "
                f"Tokens: Prompt={prompt_tokens}, Completion={completion_tokens}, Total={total_tokens}."
            )

            # Handle JSON parsing if json_mode was requested
            if json_mode and api_response_dict.get("choices"):
                try:
                    first_choice = api_response_dict["choices"][0]
                    if first_choice.get("message", {}).get("content"):
                        parsed_json = json.loads(first_choice["message"]["content"])
                        api_response_dict["parsed_json_content"] = parsed_json
                        logger.debug("Successfully parsed JSON response content.")
                    else:
                        logger.warning("JSON mode requested, but no content found in the first choice message.")
                        api_response_dict["parsed_json_content"] = None
                        api_response_dict["parsing_error"] = "No content in LLM response for JSON mode."
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response from model '{current_model}' despite json_mode=True. Error: {e}. "
                                 f"Content: {first_choice.get('message', {}).get('content', '')[:200]}...")
                    api_response_dict["parsed_json_content"] = None # Indicate parsing failure
                    api_response_dict["parsing_error"] = f"JSONDecodeError: {e}"
                except Exception as e_parse: # Catch other unexpected parsing errors
                    logger.error(f"Unexpected error parsing JSON response: {e_parse}", exc_info=True)
                    api_response_dict["parsed_json_content"] = None
                    api_response_dict["parsing_error"] = f"Unexpected parsing error: {e_parse}"


            if self.cache is not None and cache_key is not None:
                self.cache[cache_key] = api_response_dict
            
            return api_response_dict

        except LLMClientSetupError: # Propagate setup errors
            raise
        except APIError as e: # Non-retryable API errors from OpenAI/OpenRouter after tenacity retries
            logger.error(f"Non-retryable APIError after all attempts with '{current_model}': {e} (Status: {e.status_code}, Type: {e.type})", exc_info=True)
            return None
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"Unexpected error during chat completion with '{current_model}' after tenacity block: {e}", exc_info=True)
            return None

    @retry(
        stop=stop_after_attempt(DEFAULT_MAX_RETRIES + 1),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((APITimeoutError, APIConnectionError, RateLimitError, APIStatusError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying Embedding request due to {type(retry_state.outcome.exception()).__name__}. "
            f"Attempt {retry_state.attempt_number}/{LLMClient.DEFAULT_MAX_RETRIES + 1}."
        )
    )
    async def _make_embedding_api_call(self, **kwargs):
        return await self.client.embeddings.create(**kwargs)


    async def get_embedding(self, text: str, model: str = "openai/text-embedding-ada-002") -> Optional[List[float]]:
        """
        Generates an embedding for the given text.
        OpenRouter typically uses prefixes like 'openai/' for OpenAI models.
        """
        # Ensure model name compatibility for common cases
        effective_model = model
        if model == "text-embedding-ada-002" and "/" not in model:
            effective_model = f"openai/{model}"
            logger.debug(f"Adjusted embedding model name to OpenRouter format: {effective_model}")
        
        request_start_time = time.monotonic()
        logger.debug(f"Requesting embedding from model: '{effective_model}' for text (first 50 chars): '{text[:50]}...'")

        try:
            response = await self._make_embedding_api_call(
                input=[text.replace("\n", " ")], # API expects a list of strings; replace newlines as per OpenAI best practices
                model=effective_model,
                # timeout=self.DEFAULT_TIMEOUT_SECONDS # Handled by client if not overridden per call
            )
            duration = time.monotonic() - request_start_time

            if response and response.data and len(response.data) > 0 and response.data[0].embedding:
                embedding_data = response.data[0].embedding
                usage = response.usage
                logger.info(
                    f"Embedding from '{effective_model}' successful. Duration: {duration:.2f}s. "
                    f"Tokens: Prompt={usage.prompt_tokens if usage else 'N/A'}, Total={usage.total_tokens if usage else 'N/A'}."
                )
                return embedding_data
            else:
                logger.error(f"LLM embedding call to '{effective_model}' returned no data or empty embedding.")
                return None
        except LLMClientSetupError:
            raise
        except APIError as e:
            logger.error(f"APIError after all embedding attempts with '{effective_model}': {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error during embedding with '{effective_model}': {e}", exc_info=True)
            return None

# Example Usage (for testing)
async def main_test_llm_client():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s')
    logger.info("--- LLMClient Comprehensive Test ---")
    
    if not config.OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY not found in environment. Aborting LLMClient test.")
        return

    try:
        llm_client = LLMClient()

        # Test Chat Completion (Basic)
        logger.info("\n--- Testing Basic Chat Completion ---")
        test_messages_basic = [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        basic_response = await llm_client.get_chat_completion(
            test_messages_basic,
            model=config.OPENROUTER_DEFAULT_CONVERSATIONAL_MODEL, # Use configured default
            temperature=0.5, max_tokens=50
        )
        if basic_response and basic_response.get("choices"):
            logger.info(f"Basic Test Completion: {basic_response['choices'][0]['message']['content']}")
        else:
            logger.error("Basic Test Completion Failed or returned empty.")

        # Test JSON Mode Chat Completion
        logger.info("\n--- Testing JSON Mode Chat Completion ---")
        test_messages_json = [
            {"role": "system", "content": "You are an assistant that strictly outputs JSON. Create a JSON object with a key 'city' and value 'Berlin'."},
            {"role": "user", "content": "Provide the capital of Germany in the specified JSON format."}
        ]
        json_response = await llm_client.get_chat_completion(
            test_messages_json,
            model=config.OPENROUTER_DEFAULT_CONVERSATIONAL_MODEL, # Test with a model good at JSON
            temperature=0.1, json_mode=True, max_tokens=100
        )
        if json_response and json_response.get("parsed_json_content"):
            logger.info(f"JSON Mode Test Parsed Content: {json_response['parsed_json_content']}")
        elif json_response and json_response.get("parsing_error"):
            logger.error(f"JSON Mode Test Failed Parsing. Error: {json_response['parsing_error']}. Raw: {json_response.get('choices', [{}])[0].get('message', {}).get('content')}")
        else:
            logger.error("JSON Mode Test Completion Failed or returned empty.")

        # Test Embedding
        logger.info("\n--- Testing Embedding ---")
        embedding_text = "The quick brown fox jumps over the lazy dog."
        embedding_response = await llm_client.get_embedding(embedding_text, model="openai/text-embedding-ada-002") # More specific model name
        if embedding_response:
            logger.info(f"Embedding Test: Retrieved {len(embedding_response)} dimensions. First 5: {embedding_response[:5]}")
        else:
            logger.error("Embedding Test Failed.")

        # Test Tool Use / Function Calling (Conceptual)
        logger.info("\n--- Testing Tool Use Chat Completion (Conceptual) ---")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        tool_messages = [{"role": "user", "content": "What's the weather like in Boston?"}]
        tool_response = await llm_client.get_chat_completion(
            messages=tool_messages,
            model=config.OPENROUTER_DEFAULT_CONVERSATIONAL_MODEL, # A model that supports tool use
            tools=tools,
            tool_choice="auto"
        )
        if tool_response and tool_response.get("choices") and tool_response["choices"][0].get("message", {}).get("tool_calls"):
            logger.info(f"Tool Use Test - LLM wants to call a tool: {tool_response['choices'][0]['message']['tool_calls']}")
        elif tool_response and tool_response.get("choices"):
             logger.info(f"Tool Use Test - LLM direct response: {tool_response['choices'][0]['message']['content']}")
        else:
            logger.error("Tool Use Test Completion Failed or returned unexpected structure.")

    except LLMClientSetupError as e:
        logger.error(f"LLMClient setup failed during comprehensive test: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during LLMClient comprehensive test: {e}", exc_info=True)

if __name__ == "__main__":
    # This block is crucial for allowing direct testing of this file.
    # It ensures .env is loaded and logging is configured before main_test_llm_client runs.
    print("Running llm_client.py directly for testing...")
    logging.basicConfig(level=logging.DEBUG) # Set to DEBUG for verbose output during test
    
    # Explicitly load .env for direct script execution if not already done by top-level config
    if not os.getenv("OPENROUTER_API_KEY"): # Check if already loaded
        test_dotenv_path = find_dotenv(raise_error_if_not_found=False, usecwd=True)
        if test_dotenv_path and os.path.exists(test_dotenv_path):
            print(f"Direct script run: Loading .env from {test_dotenv_path}")
            load_dotenv(dotenv_path=test_dotenv_path, override=True)
            # Re-initialize config module's variables if they were set before .env load
            import importlib
            importlib.reload(config)
        else:
            print("Direct script run: .env file not found. OPENROUTER_API_KEY must be in system environment.")

    asyncio.run(main_test_llm_client())
# --------------------------------------------------------------------------------
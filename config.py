"""
config.py

Loads and validates environment variables from .env file.
Provides configuration constants for the Boutique AI system.
"""

import os
import logging
from dotenv import load_dotenv
from typing import Any

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Load .env file into environment
load_dotenv()

def get_env_var(key: str, default: Any = None, required: bool = True) -> Any:
    """
    Retrieve environment variable or default.
    Raises EnvironmentError if required and missing.
    """
    val = os.getenv(key, default)
    if required and (val is None or val == ""):
        logger.error(f"Missing required environment variable: {key}")
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return val

# Core service credentials
OPENROUTER_API_KEY: str = get_env_var("OPENROUTER_API_KEY")
OPENROUTER_API_URL: str = get_env_var("OPENROUTER_API_URL", "https://openrouter.ai/v1/chat/completions", required=False)

DEEPGRAM_API_KEY: str = get_env_var("DEEPGRAM_API_KEY")

TWILIO_ACCOUNT_SID: str = get_env_var("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN: str = get_env_var("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER: str = get_env_var("TWILIO_FROM_NUMBER")
TWILIO_PHONE_NUMBER: str = TWILIO_FROM_NUMBER  # Alias for TelephonyWrapper
BASE_WEBHOOK_URL: str = get_env_var("BASE_WEBHOOK_URL")

SALESFORCE_API_KEY: str = get_env_var("SALESFORCE_API_KEY", "", required=False)
HUBSPOT_API_KEY: str = get_env_var("HUBSPOT_API_KEY", "", required=False)

CLAY_API_KEY: str = get_env_var("CLAY_API_KEY")
CLAY_API_BASE_URL: str = get_env_var("CLAY_API_BASE_URL", "https://api.clay.com", required=False)

PROXY_API_KEY: str = get_env_var("PROXY_API_KEY", "", required=False)

# OpenRouter model settings
OPENROUTER_MODEL_NAME: str = get_env_var("OPENROUTER_MODEL_NAME", "openai/gpt-4o", required=False)
OPENROUTER_SITE_URL: str = get_env_var("OPENROUTER_SITE_URL", "", required=False)
OPENROUTER_APP_NAME: str = get_env_var("OPENROUTER_APP_NAME", "BoutiqueAI", required=False)
LLM_CACHE_SIZE: int = int(get_env_var("LLM_CACHE_SIZE", "100", required=False))

# Deepgram model settings
DEEPGRAM_STT_MODEL: str = get_env_var("DEEPGRAM_STT_MODEL", "nova-2-Phonecall", required=False)
DEEPGRAM_TTS_MODEL: str = get_env_var("DEEPGRAM_TTS_MODEL", "aura", required=False)

# Supabase configuration
SUPABASE_URL: str = get_env_var("SUPABASE_URL", "", required=False)
SUPABASE_KEY: str = get_env_var("SUPABASE_KEY", "", required=False)
SUPABASE_CALL_LOG_TABLE: str = get_env_var("SUPABASE_CALL_LOG_TABLE", "call_logs", required=False)
SUPABASE_CONTACTS_TABLE: str = get_env_var("SUPABASE_CONTACTS_TABLE", "contacts", required=False)
SUPABASE_RESOURCES_TABLE: str = get_env_var("SUPABASE_RESOURCES_TABLE", "managed_resources", required=False)
SUPABASE_ENABLED: bool = bool(SUPABASE_URL and SUPABASE_KEY)

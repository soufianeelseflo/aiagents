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
BASE_WEBHOOK_URL: str = get_env_var("BASE_WEBHOOK_URL")

SALESFORCE_API_KEY: str = get_env_var("SALESFORCE_API_KEY", "", required=False)
HUBSPOT_API_KEY: str = get_env_var("HUBSPOT_API_KEY", "", required=False)

CLAY_API_KEY: str = get_env_var("CLAY_API_KEY")

PROXY_API_KEY: str = get_env_var("PROXY_API_KEY", "", required=False)

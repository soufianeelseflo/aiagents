# boutique_ai_project/config.py

"""
Loads and validates environment variables from a single .env file at the project root.
Provides configuration constants for the Boutique AI system.
"""

import os
import logging
from dotenv import load_dotenv, find_dotenv
from typing import Optional, Any

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
try:
    dotenv_path = find_dotenv(raise_error_if_not_found=False, usecwd=True)
    if dotenv_path:
        logger.info(f"Loading environment variables from: {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path, override=True)
    else:
        logger.warning(".env file not found. Relying on system environment variables.")
except Exception as e:
     logger.error(f"Error finding or loading .env file: {e}", exc_info=True)

# --- Helper Functions for Typed Environment Variable Retrieval ---
def get_env_var(var_name: str, required: bool = True, default: Optional[Any] = None) -> Optional[str]:
    value = os.getenv(var_name)
    if value is not None and value.strip() != "":
        is_secret = any(k in var_name.upper() for k in ["KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH"])
        log_value = f"{value[:4]}...{value[-4:]}" if is_secret and len(value) > 8 else value
        logger.debug(f"Found env var '{var_name}': {log_value}")
        return value
    elif default is not None:
        logger.warning(f"Env var '{var_name}' not found. Using default: '{default}'")
        return str(default)
    elif required:
        msg = f"CRITICAL: Required env var '{var_name}' is missing or empty."
        logger.critical(msg)
        raise ValueError(msg)
    else:
        logger.info(f"Optional env var '{var_name}' not found.")
        return None

def get_int_env_var(var_name: str, required: bool = True, default: Optional[int] = None) -> Optional[int]:
    value_str = get_env_var(var_name, required=required, default=str(default) if default is not None else None)
    if value_str is None: return None
    try: return int(value_str)
    except (ValueError, TypeError):
        msg = f"Invalid integer value for env var '{var_name}': '{value_str}'."
        logger.error(msg)
        if required and default is None : raise ValueError(msg)
        return default

def get_bool_env_var(var_name: str, default: bool = False) -> bool:
     value_str = get_env_var(var_name, required=False, default=str(default))
     if value_str is None: return default
     return value_str.lower() in ['true', '1', 'yes', 'y']

# --- Core Configuration Constants ---
# Twilio
TWILIO_ACCOUNT_SID: Optional[str] = get_env_var("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN: Optional[str] = get_env_var("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER: Optional[str] = get_env_var("TWILIO_PHONE_NUMBER")

# Deepgram
DEEPGRAM_API_KEY: Optional[str] = get_env_var("DEEPGRAM_API_KEY")
DEEPGRAM_STT_MODEL: str = get_env_var("DEEPGRAM_STT_MODEL", default="nova-2-phonecall")
DEEPGRAM_TTS_MODEL: str = get_env_var("DEEPGRAM_TTS_MODEL", default="aura-asteria-en")

# OpenRouter
OPENROUTER_API_KEY: Optional[str] = get_env_var("OPENROUTER_API_KEY")
OPENROUTER_MODEL_NAME: str = get_env_var("OPENROUTER_MODEL_NAME", default="google/gemini-1.5-flash")
OPENROUTER_SITE_URL: Optional[str] = get_env_var("OPENROUTER_SITE_URL", required=False, default="https://your-app.com")
OPENROUTER_APP_NAME: Optional[str] = get_env_var("OPENROUTER_APP_NAME", required=False, default="BoutiqueAI")

# Proxies
PROXY_USERNAME: Optional[str] = get_env_var("PROXY_USERNAME", required=False)
PROXY_PASSWORD: Optional[str] = get_env_var("PROXY_PASSWORD", required=False)

# Clay.com
CLAY_API_KEY: Optional[str] = get_env_var("CLAY_API_KEY", required=False) # May be used by ResourceManager for some scenarios
CLAY_API_BASE_URL: str = get_env_var("CLAY_API_BASE_URL", default="https://api.clay.com/v1") # VERIFY if used
CLAY_RESULTS_WEBHOOK_SECRET: Optional[str] = get_env_var("CLAY_RESULTS_WEBHOOK_SECRET", required=False) # For securing your inbound webhook

# Supabase
SUPABASE_URL: Optional[str] = get_env_var("SUPABASE_URL", required=False)
SUPABASE_KEY: Optional[str] = get_env_var("SUPABASE_KEY", required=False)
SUPABASE_ENABLED: bool = bool(SUPABASE_URL and SUPABASE_KEY)
SUPABASE_CALL_LOG_TABLE: str = get_env_var("SUPABASE_CALL_LOG_TABLE", default="call_logs")
SUPABASE_CONTACTS_TABLE: str = get_env_var("SUPABASE_CONTACTS_TABLE", default="contacts")
SUPABASE_RESOURCES_TABLE: str = get_env_var("SUPABASE_RESOURCES_TABLE", default="managed_resources")

# Agent Behavior
AGENT_TARGET_NICHE_DEFAULT: str = get_env_var("AGENT_TARGET_NICHE_DEFAULT", default="B2B SaaS Companies")
AGENT_MAX_CALL_DURATION_DEFAULT: int = get_int_env_var("AGENT_MAX_CALL_DURATION_DEFAULT", default=3600)
ACQUISITION_AGENT_RUN_INTERVAL_SECONDS: int = get_int_env_var("ACQUISITION_AGENT_RUN_INTERVAL_SECONDS", default=14400)
ACQUISITION_AGENT_BATCH_SIZE: int = get_int_env_var("ACQUISITION_AGENT_BATCH_SIZE", default=25)

# Server & Webhooks
BASE_WEBHOOK_URL: Optional[str] = get_env_var("BASE_WEBHOOK_URL") # Must be your public URL (e.g., https://myapp.com or ngrok URL)
LOCAL_SERVER_PORT: int = get_int_env_var("LOCAL_SERVER_PORT", default=8080)

# System Settings
LOG_LEVEL: str = get_env_var("LOG_LEVEL", default="INFO").upper()
LOG_FILE: Optional[str] = get_env_var("LOG_FILE", required=False, default="logs/boutique_ai.log")
LLM_CACHE_SIZE: int = get_int_env_var("LLM_CACHE_SIZE", default=200)
MAX_CONCURRENT_BROWSER_AUTOMATIONS: int = get_int_env_var("MAX_CONCURRENT_BROWSER_AUTOMATIONS", default=2)

# --- Apply Logging Configuration ---
_root_logger = logging.getLogger()
for _handler in _root_logger.handlers[:]: _root_logger.removeHandler(_handler)
_valid_log_levels = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}
_final_log_level_val = _valid_log_levels.get(LOG_LEVEL, logging.INFO)
if LOG_LEVEL not in _valid_log_levels: logger.warning(f"Invalid LOG_LEVEL '{LOG_LEVEL}'. Defaulting to INFO.")
_root_logger.setLevel(_final_log_level_val)
_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(pathname)s:%(lineno)d %(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)
_root_logger.addHandler(_console_handler)
logger.info(f"Console logging re-configured with level {LOG_LEVEL}.")
if LOG_FILE:
    try:
        _log_dir = os.path.dirname(LOG_FILE)
        if _log_dir and not os.path.exists(_log_dir): os.makedirs(_log_dir, exist_ok=True)
        from logging.handlers import RotatingFileHandler
        _file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        _file_handler.setFormatter(_formatter)
        _root_logger.addHandler(_file_handler)
        logger.info(f"Rotating file logging configured to '{LOG_FILE}' with level {LOG_LEVEL}.")
    except Exception as e: logger.error(f"Failed to configure file logging to '{LOG_FILE}': {e}", exc_info=True)

logger.info("Configuration loading process complete.")
if not BASE_WEBHOOK_URL: logger.critical("CRITICAL: BASE_WEBHOOK_URL is not set. Twilio webhooks will fail.")
if not SUPABASE_ENABLED: logger.warning("Supabase persistence is DISABLED.")
if not CLAY_API_KEY and not CLAY_RESULTS_WEBHOOK_SECRET:
    logger.warning("Neither CLAY_API_KEY nor CLAY_RESULTS_WEBHOOK_SECRET are set. Clay.com interactions might be limited or insecure.")
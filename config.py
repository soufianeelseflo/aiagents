# /config.py: CORRECTED AND VERIFIED VERSION
# --------------------------------------------------------------------------------
# boutique_ai_project/config.py

"""
Loads and validates environment variables from a single .env file at the project root.
Provides configuration constants for the Boutique AI system. (Level 45)
"""

import os
import logging
import json # For parsing JSON env vars like CSV mapping
from dotenv import load_dotenv, find_dotenv
# *** THIS LINE INCLUDES Union ***
from typing import Optional, Any, Dict, List, Union

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
try:
    # Look for .env in the current working directory or parent directories
    dotenv_path = find_dotenv(raise_error_if_not_found=False, usecwd=True)
    if dotenv_path:
        logger.info(f"Loading environment variables from: {dotenv_path}")
        # Use override=True to ensure .env values take precedence over system vars if needed
        load_dotenv(dotenv_path=dotenv_path, override=True)
    else:
        # Check common relative path for containerized environments
        alt_dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env') # Check one level up from config.py's dir
        if os.path.exists(alt_dotenv_path):
             logger.info(f"Loading environment variables from alternative path: {alt_dotenv_path}")
             load_dotenv(dotenv_path=alt_dotenv_path, override=True)
        else:
            logger.warning(".env file not found in expected locations. Relying on system environment variables.")
except Exception as e:
     logger.error(f"Error finding or loading .env file: {e}", exc_info=True)


# --- Helper Functions for Typed Environment Variable Retrieval ---
def get_env_var(var_name: str, required: bool = True, default: Optional[Any] = None) -> Optional[str]:
    """Retrieves an environment variable, logs appropriately, handles requirement."""
    value = os.getenv(var_name)
    if value is not None and value.strip() != "":
        is_secret = any(k in var_name.upper() for k in ["KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH", "SID"])
        log_value = f"{value[:4]}...{value[-4:]}" if is_secret and len(value) > 8 else value
        logger.debug(f"Found env var '{var_name}': {log_value}")
        return value
    elif default is not None:
        # Using logger.info for visibility when defaults are used, was warning.
        logger.info(f"Env var '{var_name}' not found or empty. Using default value: '{default}'.")
        return str(default)
    elif required:
        msg = f"CRITICAL: Required env var '{var_name}' is missing or empty."
        logger.critical(msg)
        raise ValueError(msg)
    else:
        logger.debug(f"Optional env var '{var_name}' not found or empty.")
        return None

def get_int_env_var(var_name: str, required: bool = True, default: Optional[int] = None) -> Optional[int]:
    """Retrieves an environment variable and attempts to cast to integer."""
    value_str = get_env_var(var_name, required=required, default=str(default) if default is not None else None)
    if value_str is None: return None
    try:
        return int(value_str)
    except (ValueError, TypeError):
        msg = f"Invalid integer value for env var '{var_name}': '{value_str}'. Required={required}, Default={default}"
        logger.error(msg)
        if required and default is None : raise ValueError(msg)
        logger.warning(f"Falling back to default value {default} for '{var_name}' due to conversion error.")
        return default

def get_bool_env_var(var_name: str, default: bool = False) -> bool:
    """Retrieves an environment variable and casts to boolean (True for 'true', '1', 'yes', 'y')."""
    value_str = get_env_var(var_name, required=False, default=str(default))
    if value_str is None: return default
    return value_str.lower() in ['true', '1', 'yes', 'y']

def get_json_env_var(var_name: str, required: bool = False, default: Optional[Union[Dict, List]] = None) -> Optional[Union[Dict, List]]:
    """Retrieves an environment variable and attempts to parse as JSON."""
    value_str = get_env_var(var_name, required=required, default=None)
    if value_str:
        try:
            return json.loads(value_str)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format for env var '{var_name}'. Value: '{value_str[:100]}...'")
            if required: raise ValueError(f"Invalid JSON for required env var '{var_name}'")
            return default
    else:
        if required:
             msg = f"CRITICAL: Required JSON env var '{var_name}' is missing or empty."
             logger.critical(msg)
             raise ValueError(msg)
        return default


# --- Core Configuration Constants ---
# Twilio
TWILIO_ACCOUNT_SID: Optional[str] = get_env_var("TWILIO_ACCOUNT_SID", required=True)
TWILIO_AUTH_TOKEN: Optional[str] = get_env_var("TWILIO_AUTH_TOKEN", required=True)
TWILIO_PHONE_NUMBER: Optional[str] = get_env_var("TWILIO_PHONE_NUMBER", required=True) # Matched to user's .env list

# Deepgram
DEEPGRAM_API_KEY: Optional[str] = get_env_var("DEEPGRAM_API_KEY", required=True)
DEEPGRAM_STT_MODEL: str = get_env_var("DEEPGRAM_STT_MODEL", required=False, default="nova-2-phonecall")
DEEPGRAM_TTS_MODEL: str = get_env_var("DEEPGRAM_TTS_MODEL", required=False, default="aura-asteria-en")

# OpenRouter
OPENROUTER_API_KEY: Optional[str] = get_env_var("OPENROUTER_API_KEY", required=True)
OPENROUTER_DEFAULT_CONVERSATIONAL_MODEL: str = get_env_var("OPENROUTER_DEFAULT_CONVERSATIONAL_MODEL", required=False, default="google/gemini-flash-1.5") # User's specified default
OPENROUTER_DEFAULT_STRATEGY_MODEL: str = get_env_var("OPENROUTER_DEFAULT_STRATEGY_MODEL", required=False, default="google/gemini-flash-1.5") # User's specified default
OPENROUTER_DEFAULT_ANALYSIS_MODEL: str = get_env_var("OPENROUTER_DEFAULT_ANALYSIS_MODEL", required=False, default="google/gemini-flash-1.5") # User's specified default
OPENROUTER_DEFAULT_VISION_MODEL: str = get_env_var("OPENROUTER_DEFAULT_VISION_MODEL", required=False, default="google/gemini-pro-vision") # Changed from :thinking based on common model IDs
OPENROUTER_SITE_URL: Optional[str] = get_env_var("OPENROUTER_SITE_URL", required=False, default="https://github.com/soufianeelseflo/aiagents")
OPENROUTER_APP_NAME: Optional[str] = get_env_var("OPENROUTER_APP_NAME", required=False, default="BoutiqueAI")
OPENROUTER_CLIENT_MAX_RETRIES: int = get_int_env_var("OPENROUTER_CLIENT_MAX_RETRIES", required=False, default=2)
OPENROUTER_CLIENT_TIMEOUT_SECONDS: int = get_int_env_var("OPENROUTER_CLIENT_TIMEOUT_SECONDS", required=False, default=120)

# Proxies
PROXY_HOST: Optional[str] = get_env_var("PROXY_HOST", required=False)
PROXY_PORT: Optional[int] = get_int_env_var("PROXY_PORT", required=False)
PROXY_USERNAME: Optional[str] = get_env_var("PROXY_USERNAME", required=False)
PROXY_PASSWORD: Optional[str] = get_env_var("PROXY_PASSWORD", required=False)

# Clay.com
CLAY_API_KEY: Optional[str] = get_env_var("CLAY_API_KEY", required=False)
CLAY_API_BASE_URL: str = get_env_var("CLAY_API_BASE_URL", required=False, default="https://api.clay.com/v1")
CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY: Optional[str] = get_env_var("CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY", required=False)
CLAY_RESULTS_CALLBACK_SECRET_TOKEN: Optional[str] = get_env_var("CLAY_RESULTS_CALLBACK_SECRET_TOKEN", required=False)
CLAY_CALLBACK_AUTH_HEADER_NAME: str = get_env_var("CLAY_CALLBACK_AUTH_HEADER_NAME", required=False, default="X-Callback-Auth-Token")

# Supabase
SUPABASE_URL: Optional[str] = get_env_var("SUPABASE_URL", required=True)
SUPABASE_KEY: Optional[str] = get_env_var("SUPABASE_KEY", required=True) # Service Role Key
SUPABASE_ENABLED: bool = bool(SUPABASE_URL and SUPABASE_KEY)
SUPABASE_CALL_LOGS_TABLE: str = get_env_var("SUPABASE_CALL_LOGS_TABLE", required=False, default="call_logs")
SUPABASE_CONTACTS_TABLE: str = get_env_var("SUPABASE_CONTACTS_TABLE", required=False, default="contacts")
SUPABASE_RESOURCES_TABLE: str = get_env_var("SUPABASE_RESOURCES_TABLE", required=False, default="managed_resources")

# Agent Behavior
AGENT_TARGET_NICHE_DEFAULT: str = get_env_var("AGENT_TARGET_NICHE_DEFAULT", required=False, default="B2B Technology Scale-ups")
AGENT_MAX_CALL_DURATION_DEFAULT: int = get_int_env_var("AGENT_MAX_CALL_DURATION_DEFAULT", required=False, default=3600)
ACQUISITION_AGENT_RUN_INTERVAL_SECONDS: int = get_int_env_var("ACQUISITION_AGENT_RUN_INTERVAL_SECONDS", required=False, default=7200)
ACQUISITION_AGENT_BATCH_SIZE: int = get_int_env_var("ACQUISITION_AGENT_BATCH_SIZE", required=False, default=25)
ACQ_LEAD_SOURCE_TYPE: str = get_env_var("ACQ_LEAD_SOURCE_TYPE", required=False, default="supabase_query")
ACQ_LEAD_SOURCE_PATH: str = get_env_var("ACQ_LEAD_SOURCE_PATH", required=False, default="data/initial_leads.csv")
ACQ_LEAD_SOURCE_CSV_MAPPING: Dict = get_json_env_var("ACQ_LEAD_SOURCE_CSV_MAPPING", required=False, default={"company_name": "company_name", "domain": "domain", "email": "primary_contact_email_guess"}) or {}
ACQ_SUPABASE_PENDING_LEAD_STATUS: str = get_env_var("ACQ_SUPABASE_PENDING_LEAD_STATUS", required=False, default="New_Raw_Lead") # Matched to user's provided value
ACQ_QUALIFICATION_THRESHOLD: int = get_int_env_var("ACQ_QUALIFICATION_THRESHOLD", required=False, default=7)

# Server & Webhooks
BASE_WEBHOOK_URL: Optional[str] = get_env_var("BASE_WEBHOOK_URL", required=True)
LOCAL_SERVER_PORT: int = get_int_env_var("LOCAL_SERVER_PORT", required=False, default=8080)
UVICORN_RELOAD: bool = get_bool_env_var("UVICORN_RELOAD", default=False)

# System Settings
LOG_LEVEL: str = get_env_var("LOG_LEVEL", required=False, default="INFO").upper()
LOG_FILE: Optional[str] = get_env_var("LOG_FILE", required=False, default="logs/boutique_ai_app.log") # From user's version
LLM_CACHE_SIZE: int = get_int_env_var("LLM_CACHE_SIZE", required=False, default=200) # From user's version
MAX_CONCURRENT_BROWSER_AUTOMATIONS: int = get_int_env_var("MAX_CONCURRENT_BROWSER_AUTOMATIONS", required=False, default=1)

# Playwright Settings
PLAYWRIGHT_NAV_TIMEOUT_MS: int = get_int_env_var("PLAYWRIGHT_NAV_TIMEOUT_MS", required=False, default=60000)
PLAYWRIGHT_ACTION_TIMEOUT_MS: int = get_int_env_var("PLAYWRIGHT_ACTION_TIMEOUT_MS", required=False, default=20000)
PLAYWRIGHT_SELECTOR_TIMEOUT_MS: int = get_int_env_var("PLAYWRIGHT_SELECTOR_TIMEOUT_MS", required=False, default=15000)
USE_REAL_BROWSER_AUTOMATOR: bool = get_bool_env_var("USE_REAL_BROWSER_AUTOMATOR", default=False)
PLAYWRIGHT_HEADFUL_MODE: bool = get_bool_env_var("PLAYWRIGHT_HEADFUL_MODE", default=False)

# --- Apply Logging Configuration ---
_root_logger = logging.getLogger()
for _handler in _root_logger.handlers[:]: _root_logger.removeHandler(_handler)

_valid_log_levels = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}
_final_log_level_val = _valid_log_levels.get(LOG_LEVEL, logging.INFO)
if LOG_LEVEL not in _valid_log_levels: logger.warning(f"Invalid LOG_LEVEL '{LOG_LEVEL}'. Defaulting to INFO.")
_root_logger.setLevel(_final_log_level_val)

_formatter = logging.Formatter( # Using user's specific format string
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
        if _log_dir and not os.path.exists(_log_dir):
            logger.info(f"Creating log directory: {_log_dir}")
            os.makedirs(_log_dir, exist_ok=True)
        from logging.handlers import RotatingFileHandler
        _file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8') # User's backupCount
        _file_handler.setFormatter(_formatter)
        _root_logger.addHandler(_file_handler)
        logger.info(f"Rotating file logging configured to '{LOG_FILE}' with level {LOG_LEVEL}.")
    except Exception as e:
        logger.error(f"Failed to configure file logging to '{LOG_FILE}': {e}", exc_info=True)

# --- Final Sanity Checks & Info ---
logger.info("--- Configuration Loaded ---")
if not BASE_WEBHOOK_URL: logger.critical("CRITICAL: BASE_WEBHOOK_URL is not set. Twilio webhooks WILL fail.")
if not SUPABASE_ENABLED: logger.warning("Supabase persistence is DISABLED (URL or KEY might be missing).")
if CLAY_API_KEY and not CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY: logger.warning("CLAY_API_KEY is set, but CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY is not. AcquisitionAgent Clay enrichment may fail unless webhook URL is discovered.")
if CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY and not CLAY_RESULTS_CALLBACK_SECRET_TOKEN: logger.warning("CLAY_RESULTS_CALLBACK_SECRET_TOKEN not set for Clay Webhook. Endpoint is INSECURE.")
logger.info(f"Using Playwright: {USE_REAL_BROWSER_AUTOMATOR} (Headless: {not PLAYWRIGHT_HEADFUL_MODE})")
logger.info("Configuration loading complete.")
# --------------------------------------------------------------------------------
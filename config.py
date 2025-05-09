# /config.py:
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
from typing import Optional, Any, Dict, List, Union # Added Union

# --- Configure Logging ---
# Initial basic configuration, will be refined later in this file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
try:
    # Try to find .env up to two directories above the current file, or in cwd
    # This helps if config.py is nested, e.g. in a 'core' directory.
    # For this project, it's at the root, so find_dotenv(usecwd=True) is fine.
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
        is_secret = any(k in var_name.upper() for k in ["KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH", "SID"]) # Added SID
        log_value = f"{value[:4]}...{value[-4:]}" if is_secret and len(value) > 8 else value
        # Use a more specific logger for config loading if desired, e.g., logging.getLogger("config_loader")
        # logger.debug(f"Found env var '{var_name}': {log_value}")
        return value
    elif default is not None:
        logger.warning(f"Env var '{var_name}' not found. Using default: '{str(default)[:50]}{'...' if len(str(default)) > 50 else ''}'")
        return str(default)
    elif required:
        msg = f"CRITICAL: Required env var '{var_name}' is missing or empty."
        logger.critical(msg)
        raise ValueError(msg)
    else:
        # logger.info(f"Optional env var '{var_name}' not found, no default provided.")
        return None

def get_int_env_var(var_name: str, required: bool = True, default: Optional[int] = None) -> Optional[int]:
    value_str = get_env_var(var_name, required=required, default=str(default) if default is not None else None)
    if value_str is None: return None # Handles case where optional var is not found and no default
    try: return int(value_str)
    except (ValueError, TypeError):
        msg = f"Invalid integer value for env var '{var_name}': '{value_str}'. Expected an integer."
        logger.error(msg)
        if required and default is None : raise ValueError(msg) # Raise if required and no valid default path
        logger.warning(f"Falling back to default value for '{var_name}' ({default}) due to conversion error.")
        return default

def get_bool_env_var(var_name: str, default: bool = False) -> bool:
     value_str = get_env_var(var_name, required=False, default=str(default))
     if value_str is None: return default # Should not happen if default is always str(default)
     return value_str.lower() in ['true', '1', 'yes', 'y']

def get_json_env_var(var_name: str, required: bool = False, default: Optional[Union[Dict, List]] = None) -> Optional[Union[Dict, List]]:
    value_str = get_env_var(var_name, required=required, default=None) # Get raw string or None
    if value_str:
        try:
            return json.loads(value_str)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format for env var '{var_name}'. Value: '{value_str[:100]}...'")
            if required: raise ValueError(f"Invalid JSON for required env var '{var_name}'")
            logger.warning(f"Falling back to default value for JSON var '{var_name}' due to decode error.")
            return default
    else: # Value not found or empty string
        if required: # This case implies get_env_var would have raised if truly required and missing
            # This path is more for required=True but an empty string was somehow passed.
            msg = f"CRITICAL: Required JSON env var '{var_name}' is missing or effectively empty."
            logger.critical(msg)
            raise ValueError(msg)
        return default


# --- Core Configuration Constants ---
# Twilio
TWILIO_ACCOUNT_SID: Optional[str] = get_env_var("TWILIO_ACCOUNT_SID", required=True)
TWILIO_AUTH_TOKEN: Optional[str] = get_env_var("TWILIO_AUTH_TOKEN", required=True)
TWILIO_PHONE_NUMBER: Optional[str] = get_env_var("TWILIO_PHONE_NUMBER", required=True)

# Deepgram
DEEPGRAM_API_KEY: Optional[str] = get_env_var("DEEPGRAM_API_KEY", required=True)
DEEPGRAM_STT_MODEL: str = get_env_var("DEEPGRAM_STT_MODEL", required=False, default="nova-2-phonecall")
DEEPGRAM_TTS_MODEL: str = get_env_var("DEEPGRAM_TTS_MODEL", required=False, default="aura-asteria-en") # Example, check Deepgram for latest/best

# OpenRouter
OPENROUTER_API_KEY: Optional[str] = get_env_var("OPENROUTER_API_KEY", required=True)
OPENROUTER_DEFAULT_CONVERSATIONAL_MODEL: str = get_env_var("OPENROUTER_DEFAULT_CONVERSATIONAL_MODEL", required=False, default="google/gemini-flash-1.5") # Updated to a common model
OPENROUTER_DEFAULT_STRATEGY_MODEL: str = get_env_var("OPENROUTER_DEFAULT_STRATEGY_MODEL", required=False, default="google/gemini-flash-1.5")
OPENROUTER_DEFAULT_ANALYSIS_MODEL: str = get_env_var("OPENROUTER_DEFAULT_ANALYSIS_MODEL", required=False, default="google/gemini-1.5-pro-latest") # More powerful for analysis
OPENROUTER_DEFAULT_VISION_MODEL: str = get_env_var("OPENROUTER_DEFAULT_VISION_MODEL", required=False, default="google/gemini-pro-vision") # Common vision model
OPENROUTER_SITE_URL: Optional[str] = get_env_var("OPENROUTER_SITE_URL", required=False, default="https://boutique-ai-agency.com") # Example
OPENROUTER_APP_NAME: Optional[str] = get_env_var("OPENROUTER_APP_NAME", required=False, default="BoutiqueAIAgents")
OPENROUTER_CLIENT_MAX_RETRIES: int = get_int_env_var("OPENROUTER_CLIENT_MAX_RETRIES", required=False, default=2) # Reduced default
OPENROUTER_CLIENT_TIMEOUT_SECONDS: int = get_int_env_var("OPENROUTER_CLIENT_TIMEOUT_SECONDS", required=False, default=120) # Reduced default

# Proxies
PROXY_USERNAME: Optional[str] = get_env_var("PROXY_USERNAME", required=False)
PROXY_PASSWORD: Optional[str] = get_env_var("PROXY_PASSWORD", required=False)

# Clay.com
CLAY_API_KEY: Optional[str] = get_env_var("CLAY_API_KEY", required=False)
CLAY_API_BASE_URL: str = get_env_var("CLAY_API_BASE_URL", required=False, default="https://api.clay.com/v1")
CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY: Optional[str] = get_env_var("CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY", required=False) # AcquisitionAgent may attempt discovery if missing
CLAY_RESULTS_CALLBACK_SECRET_TOKEN: Optional[str] = get_env_var("CLAY_RESULTS_CALLBACK_SECRET_TOKEN", required=False) # Recommended for securing the callback
CLAY_CALLBACK_AUTH_HEADER_NAME: str = get_env_var("CLAY_CALLBACK_AUTH_HEADER_NAME", required=False, default="X-Callback-Auth-Token")


# Supabase
SUPABASE_URL: Optional[str] = get_env_var("SUPABASE_URL", required=True) # Required for core functionality
SUPABASE_KEY: Optional[str] = get_env_var("SUPABASE_KEY", required=True) # Must be Service Role Key for DB setup & operations
SUPABASE_ENABLED: bool = bool(SUPABASE_URL and SUPABASE_KEY)
SUPABASE_CALL_LOGS_TABLE: str = get_env_var("SUPABASE_CALL_LOGS_TABLE", required=False, default="call_logs")
SUPABASE_CONTACTS_TABLE: str = get_env_var("SUPABASE_CONTACTS_TABLE", required=False, default="contacts")
SUPABASE_RESOURCES_TABLE: str = get_env_var("SUPABASE_RESOURCES_TABLE", required=False, default="managed_resources")

# Agent Behavior
AGENT_TARGET_NICHE_DEFAULT: str = get_env_var("AGENT_TARGET_NICHE_DEFAULT", required=False, default="B2B SaaS Companies")
AGENT_MAX_CALL_DURATION_DEFAULT: int = get_int_env_var("AGENT_MAX_CALL_DURATION_DEFAULT", required=False, default=1800) # 30 minutes
ACQUISITION_AGENT_RUN_INTERVAL_SECONDS: int = get_int_env_var("ACQUISITION_AGENT_RUN_INTERVAL_SECONDS", required=False, default=3600) # 1 hour
ACQUISITION_AGENT_BATCH_SIZE: int = get_int_env_var("ACQUISITION_AGENT_BATCH_SIZE", required=False, default=10) # Smaller default batch
# ACQ_LEAD_SOURCE_TYPE: valid values "supabase_query", "local_csv"
ACQ_LEAD_SOURCE_TYPE: str = get_env_var("ACQ_LEAD_SOURCE_TYPE", required=False, default="supabase_query")
ACQ_LEAD_SOURCE_PATH: str = get_env_var("ACQ_LEAD_SOURCE_PATH", required=False, default="data/initial_leads.csv")
ACQ_LEAD_SOURCE_CSV_MAPPING: Dict = get_json_env_var("ACQ_LEAD_SOURCE_CSV_MAPPING", required=False, default={"company_name": "Company Name", "domain": "Website", "email": "Email Address"}) # Example mapping
ACQ_SUPABASE_PENDING_LEAD_STATUS: str = get_env_var("ACQ_SUPABASE_PENDING_LEAD_STATUS", required=False, default="New_Raw_Lead")
ACQ_QUALIFICATION_THRESHOLD: int = get_int_env_var("ACQ_QUALIFICATION_THRESHOLD", required=False, default=6) # Adjusted threshold

# Server & Webhooks
BASE_WEBHOOK_URL: Optional[str] = get_env_var("BASE_WEBHOOK_URL", required=True) # Must NOT have trailing slash
LOCAL_SERVER_PORT: int = get_int_env_var("LOCAL_SERVER_PORT", required=False, default=8080)
UVICORN_RELOAD: bool = get_bool_env_var("UVICORN_RELOAD", default=False)

# Browser Automation
# Default PLAYWRIGHT_HEADFUL_MODE should be False for server/Docker.
# server.py will use this.
PLAYWRIGHT_HEADFUL_MODE: bool = get_bool_env_var("PLAYWRIGHT_HEADFUL_MODE", default=False)
USE_REAL_BROWSER_AUTOMATOR: bool = get_bool_env_var("USE_REAL_BROWSER_AUTOMATOR", default=False) # Default to Mock unless explicitly enabled
PLAYWRIGHT_NAV_TIMEOUT_MS: int = get_int_env_var("PLAYWRIGHT_NAV_TIMEOUT_MS", default=60000)
PLAYWRIGHT_ACTION_TIMEOUT_MS: int = get_int_env_var("PLAYWRIGHT_ACTION_TIMEOUT_MS", default=30000)
PLAYWRIGHT_SELECTOR_TIMEOUT_MS: int = get_int_env_var("PLAYWRIGHT_SELECTOR_TIMEOUT_MS", default=15000)


# System Settings
LOG_LEVEL: str = get_env_var("LOG_LEVEL", required=False, default="INFO").upper()
LOG_FILE: Optional[str] = get_env_var("LOG_FILE", required=False) # Default to no file logging unless specified
LLM_CACHE_SIZE: int = get_int_env_var("LLM_CACHE_SIZE", required=False, default=100) # Reduced default
MAX_CONCURRENT_BROWSER_AUTOMATIONS: int = get_int_env_var("MAX_CONCURRENT_BROWSER_AUTOMATIONS", required=False, default=1) # Safety default

# Deployment Related (used by DockerDeploymentManager)
COOLIFY_APPLICATION_IMAGE: Optional[str] = get_env_var("COOLIFY_APPLICATION_IMAGE", required=False) # From Coolify if set

# --- Apply Logging Configuration ---
# This ensures logging is set up once, using the LOG_LEVEL from .env
# Uvicorn will use its own loggers for access logs, but application logs will follow this.

_root_logger = logging.getLogger() # Get the root logger
# Remove any handlers already attached to the root logger by basicConfig or other means
for _handler in _root_logger.handlers[:]:
    _root_logger.removeHandler(_handler)

_valid_log_levels = {
    "DEBUG": logging.DEBUG, "INFO": logging.INFO,
    "WARNING": logging.WARNING, "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}
_final_log_level_val = _valid_log_levels.get(LOG_LEVEL, logging.INFO)

if LOG_LEVEL not in _valid_log_levels:
    # Use print here as logger might not be fully set up if LOG_LEVEL itself is bad
    print(f"Warning: Invalid LOG_LEVEL '{LOG_LEVEL}' in .env. Defaulting to INFO.")

_root_logger.setLevel(_final_log_level_val)

# Configure console handler
_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(pathname)s:%(lineno)d %(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)
_console_handler.setLevel(_final_log_level_val) # Set level for handler too
_root_logger.addHandler(_console_handler)
# logger.info(f"Root logger configured. Console logging level: {LOG_LEVEL}") # Use specific logger

# Configure file handler if LOG_FILE is specified
if LOG_FILE:
    try:
        _log_dir = os.path.dirname(LOG_FILE)
        if _log_dir and not os.path.exists(_log_dir):
            os.makedirs(_log_dir, exist_ok=True)
            logger.info(f"Created log directory: {_log_dir}")

        from logging.handlers import RotatingFileHandler
        _file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=3, encoding='utf-8')
        _file_handler.setFormatter(_formatter)
        _file_handler.setLevel(_final_log_level_val) # Set level for file handler
        _root_logger.addHandler(_file_handler)
        logger.info(f"Rotating file logging configured to '{LOG_FILE}'. Level: {LOG_LEVEL}")
    except Exception as e:
        logger.error(f"Failed to configure file logging to '{LOG_FILE}': {e}", exc_info=True)
else:
    logger.info("File logging is not configured (LOG_FILE not set in .env).")


# --- Final Sanity Checks & Info ---
logger.info("--- Boutique AI Configuration Loaded ---")
if not BASE_WEBHOOK_URL:
    logger.critical("CRITICAL FAILURE: BASE_WEBHOOK_URL is not set in .env. External webhooks (Twilio, Clay callback) WILL FAIL.")
if not SUPABASE_ENABLED:
    logger.warning("Supabase persistence is DISABLED (SUPABASE_URL or SUPABASE_KEY missing). CRM and Resource Management will be affected.")
else:
    logger.info(f"Supabase ENABLED. URL: {SUPABASE_URL[:20]}...") # Don't log full sensitive URL always

if not CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY:
    logger.warning("CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY not set. AcquisitionAgent will attempt discovery if used, or Clay enrichment will fail.")
if CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY and not CLAY_RESULTS_CALLBACK_SECRET_TOKEN:
    logger.warning("CLAY_RESULTS_CALLBACK_SECRET_TOKEN not set. Clay results callback webhook is INSECURE. Please set this for production.")

logger.info(f"Default Niche: {AGENT_TARGET_NICHE_DEFAULT}")
logger.info(f"Uvicorn Reload: {UVICORN_RELOAD}")
logger.info(f"Use REAL Browser Automator: {USE_REAL_BROWSER_AUTOMATOR}")
logger.info(f"Playwright Headful Mode (config.py): {PLAYWRIGHT_HEADFUL_MODE}") # Log the config value
logger.info("--- Configuration loading process complete. Review warnings above. ---")

# Make sure all critical required variables were caught if missing.
# The get_env_var function with required=True should have raised ValueError already.
# --------------------------------------------------------------------------------
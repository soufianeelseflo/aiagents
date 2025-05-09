# /config.py: (Holistically Reviewed and Corrected for Deployment)
# --------------------------------------------------------------------------------
# boutique_ai_project/config.py

"""
Loads and validates environment variables from a single .env file (if present)
or system environment. Provides configuration constants for the Boutique AI system.
"""

import os
import logging
import json
from dotenv import load_dotenv, find_dotenv
from typing import Optional, Any, Dict, List, Union # ENSURES 'Union' IS IMPORTED

# --- Initial Logger Setup (before full config load) ---
# This initial setup ensures that issues during config loading itself can be logged.
# It will be reconfigured more comprehensively later in this file.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
try:
    dotenv_path = find_dotenv(raise_error_if_not_found=False, usecwd=True)
    if dotenv_path and os.path.exists(dotenv_path) and os.path.isfile(dotenv_path):
        logger.info(f"Loading environment variables from: {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path, override=True)
    else:
        # Attempt to load .env from /app/.env if it exists (common in containers if mounted)
        container_dotenv_path = "/app/.env"
        if os.path.exists(container_dotenv_path) and os.path.isfile(container_dotenv_path):
            logger.info(f"Loading environment variables from container path: {container_dotenv_path}")
            load_dotenv(dotenv_path=container_dotenv_path, override=True)
        else:
            logger.info(".env file not found in project path, parent directories, or /app/.env. Relying on system-set environment variables or defaults.")
except Exception as e:
     logger.error(f"An error occurred during .env file loading: {e}", exc_info=True)


# --- Helper Functions for Typed Environment Variable Retrieval ---
def get_env_var(var_name: str, required: bool = True, default: Optional[Any] = None) -> Optional[str]:
    value = os.getenv(var_name)
    if value is not None and value.strip() != "":
        is_secret = any(k in var_name.upper() for k in ["KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH", "SID"])
        log_display = "******" if is_secret else value
        if logger.isEnabledFor(logging.DEBUG) or not is_secret: # Log actual value for non-secrets if debug
             logger.debug(f"Found env var '{var_name}': {log_display}")
        else: # For secrets not in debug, just confirm presence
             logger.debug(f"Found env var '{var_name}': <present_and_redacted>")
        return value
    elif default is not None:
        logger.info(f"Env var '{var_name}' not found or empty. Using default value: '{default}'.")
        return str(default)
    elif required:
        msg = f"CRITICAL: Required environment variable '{var_name}' is missing or empty. This will likely cause application failure."
        logger.critical(msg)
        raise ValueError(msg)
    else:
        logger.debug(f"Optional env var '{var_name}' not found or empty. No default provided.")
        return None

def get_int_env_var(var_name: str, required: bool = True, default: Optional[int] = None) -> Optional[int]:
    value_str = get_env_var(var_name, required=required, default=str(default) if default is not None else None)
    if value_str is None: return None
    try:
        return int(value_str)
    except (ValueError, TypeError):
        msg = f"Invalid integer value for env var '{var_name}': '{value_str}'. Required={required}, Default={default}"
        logger.error(msg)
        if required and default is None: raise ValueError(msg)
        logger.warning(f"Falling back to default integer value {default} for '{var_name}' due to conversion error.")
        return default

def get_bool_env_var(var_name: str, default: bool = False) -> bool:
    value_str = get_env_var(var_name, required=False, default=str(default))
    if value_str is None: return default
    return value_str.lower() in ['true', '1', 'yes', 'y']

def get_json_env_var(var_name: str, required: bool = False, default: Optional[Union[Dict, List]] = None) -> Optional[Union[Dict, List]]:
    value_str = get_env_var(var_name, required=required, default=None)
    if value_str:
        try:
            return json.loads(value_str)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format for env var '{var_name}'. Error: {e}. Value (partial): '{value_str[:100]}...'")
            if required: raise ValueError(f"Invalid JSON for required env var '{var_name}'")
            return default
    else:
        if required:
             msg = f"CRITICAL: Required JSON env var '{var_name}' is missing or empty."
             logger.critical(msg)
             raise ValueError(msg)
        return default

# --- Configuration Constants (Loading from Environment or Defaults) ---
# Twilio (Required for core telephony)
TWILIO_ACCOUNT_SID: Optional[str] = get_env_var("TWILIO_ACCOUNT_SID", required=True)
TWILIO_AUTH_TOKEN: Optional[str] = get_env_var("TWILIO_AUTH_TOKEN", required=True)
TWILIO_PHONE_NUMBER: Optional[str] = get_env_var("TWILIO_PHONE_NUMBER", required=True)

# Deepgram (Required for STT/TTS in calls)
DEEPGRAM_API_KEY: Optional[str] = get_env_var("DEEPGRAM_API_KEY", required=True)
DEEPGRAM_STT_MODEL: str = get_env_var("DEEPGRAM_STT_MODEL", required=False, default="nova-2-phonecall")
DEEPGRAM_TTS_MODEL: str = get_env_var("DEEPGRAM_TTS_MODEL", required=False, default="aura-asteria-en")

# OpenRouter (Required for LLM capabilities)
OPENROUTER_API_KEY: Optional[str] = get_env_var("OPENROUTER_API_KEY", required=True)
OPENROUTER_DEFAULT_CONVERSATIONAL_MODEL: str = get_env_var("OPENROUTER_DEFAULT_CONVERSATIONAL_MODEL", required=False, default="google/gemini-flash-1.5")
OPENROUTER_DEFAULT_STRATEGY_MODEL: str = get_env_var("OPENROUTER_DEFAULT_STRATEGY_MODEL", required=False, default="google/gemini-flash-1.5")
OPENROUTER_DEFAULT_ANALYSIS_MODEL: str = get_env_var("OPENROUTER_DEFAULT_ANALYSIS_MODEL", required=False, default="google/gemini-flash-1.5")
OPENROUTER_DEFAULT_VISION_MODEL: str = get_env_var("OPENROUTER_DEFAULT_VISION_MODEL", required=False, default="google/gemini-pro-vision")
OPENROUTER_SITE_URL: str = get_env_var("OPENROUTER_SITE_URL", required=False, default="https://github.com/soufianeelseflo/aiagents")
OPENROUTER_APP_NAME: str = get_env_var("OPENROUTER_APP_NAME", required=False, default="BoutiqueAI")
OPENROUTER_CLIENT_MAX_RETRIES: int = get_int_env_var("OPENROUTER_CLIENT_MAX_RETRIES", required=False, default=2)
OPENROUTER_CLIENT_TIMEOUT_SECONDS: int = get_int_env_var("OPENROUTER_CLIENT_TIMEOUT_SECONDS", required=False, default=120)

# Proxies (Optional, used by Playwright if configured)
PROXY_HOST: Optional[str] = get_env_var("PROXY_HOST", required=False)
PROXY_PORT: Optional[int] = get_int_env_var("PROXY_PORT", required=False)
PROXY_USERNAME: Optional[str] = get_env_var("PROXY_USERNAME", required=False)
PROXY_PASSWORD: Optional[str] = get_env_var("PROXY_PASSWORD", required=False)

# Clay.com (Optional, for lead enrichment)
CLAY_API_KEY: Optional[str] = get_env_var("CLAY_API_KEY", required=False)
CLAY_API_BASE_URL: str = get_env_var("CLAY_API_BASE_URL", required=False, default="https://api.clay.com/v1")
CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY: Optional[str] = get_env_var("CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY", required=False)
CLAY_RESULTS_CALLBACK_SECRET_TOKEN: Optional[str] = get_env_var("CLAY_RESULTS_CALLBACK_SECRET_TOKEN", required=False)
CLAY_CALLBACK_AUTH_HEADER_NAME: str = get_env_var("CLAY_CALLBACK_AUTH_HEADER_NAME", required=False, default="X-Callback-Auth-Token")

# Supabase (Required for database persistence and initial setup)
SUPABASE_URL: Optional[str] = get_env_var("SUPABASE_URL", required=True)
SUPABASE_KEY: Optional[str] = get_env_var("SUPABASE_KEY", required=True) # This MUST be the Service Role Key
SUPABASE_ENABLED: bool = bool(SUPABASE_URL and SUPABASE_KEY)
SUPABASE_CALL_LOGS_TABLE: str = get_env_var("SUPABASE_CALL_LOGS_TABLE", required=False, default="call_logs")
SUPABASE_CONTACTS_TABLE: str = get_env_var("SUPABASE_CONTACTS_TABLE", required=False, default="contacts")
SUPABASE_RESOURCES_TABLE: str = get_env_var("SUPABASE_RESOURCES_TABLE", required=False, default="managed_resources")
SUPABASE_AGENTS_TABLE: str = get_env_var("SUPABASE_AGENTS_TABLE", required=False, default="deployed_agents") # From database_setup.py

# Agent Behavior
AGENT_TARGET_NICHE_DEFAULT: str = get_env_var("AGENT_TARGET_NICHE_DEFAULT", required=False, default="B2B Technology Scale-ups")
AGENT_MAX_CALL_DURATION_DEFAULT: int = get_int_env_var("AGENT_MAX_CALL_DURATION_DEFAULT", required=False, default=3600)
ACQUISITION_AGENT_RUN_INTERVAL_SECONDS: int = get_int_env_var("ACQUISITION_AGENT_RUN_INTERVAL_SECONDS", required=False, default=7200)
ACQUISITION_AGENT_BATCH_SIZE: int = get_int_env_var("ACQUISITION_AGENT_BATCH_SIZE", required=False, default=10)
ACQ_LEAD_SOURCE_TYPE: str = get_env_var("ACQ_LEAD_SOURCE_TYPE", required=False, default="supabase_query")
ACQ_LEAD_SOURCE_PATH: str = get_env_var("ACQ_LEAD_SOURCE_PATH", required=False, default="data/initial_leads.csv")
ACQ_LEAD_SOURCE_CSV_MAPPING: Dict = get_json_env_var("ACQ_LEAD_SOURCE_CSV_MAPPING", required=False, default={"company_name": "company_name", "domain": "domain", "email": "primary_contact_email_guess"}) or {}
ACQ_SUPABASE_PENDING_LEAD_STATUS: str = get_env_var("ACQ_SUPABASE_PENDING_LEAD_STATUS", required=False, default="New_Raw_Lead")
ACQ_QUALIFICATION_THRESHOLD: int = get_int_env_var("ACQ_QUALIFICATION_THRESHOLD", required=False, default=7)

# Server & Webhooks
BASE_WEBHOOK_URL: Optional[str] = get_env_var("BASE_WEBHOOK_URL", required=True) # e.g. https://yourapp.coolifyapp.com
LOCAL_SERVER_PORT: int = get_int_env_var("LOCAL_SERVER_PORT", required=False, default=8080)
UVICORN_RELOAD: bool = get_bool_env_var("UVICORN_RELOAD", default=False) # Should ALWAYS be False in production Docker

# System Settings
LOG_LEVEL: str = get_env_var("LOG_LEVEL", required=False, default="INFO").upper()
LOG_FILE: Optional[str] = get_env_var("LOG_FILE", required=False, default=None)
LLM_CACHE_SIZE: int = get_int_env_var("LLM_CACHE_SIZE", required=False, default=100)
MAX_CONCURRENT_BROWSER_AUTOMATIONS: int = get_int_env_var("MAX_CONCURRENT_BROWSER_AUTOMATIONS", required=False, default=1)

# Playwright Specific Settings
PLAYWRIGHT_NAV_TIMEOUT_MS: int = get_int_env_var("PLAYWRIGHT_NAV_TIMEOUT_MS", required=False, default=60000)
PLAYWRIGHT_ACTION_TIMEOUT_MS: int = get_int_env_var("PLAYWRIGHT_ACTION_TIMEOUT_MS", required=False, default=20000)
PLAYWRIGHT_SELECTOR_TIMEOUT_MS: int = get_int_env_var("PLAYWRIGHT_SELECTOR_TIMEOUT_MS", required=False, default=15000)
USE_REAL_BROWSER_AUTOMATOR: bool = get_bool_env_var("USE_REAL_BROWSER_AUTOMATOR", default=True)
PLAYWRIGHT_HEADFUL_MODE: bool = get_bool_env_var("PLAYWRIGHT_HEADFUL_MODE", default=False)


# --- Apply Logging Configuration (after all variables are loaded) ---
# Store the root logger instance
_root_logger_instance = logging.getLogger()

# Clear any handlers already attached to the root logger by basicConfig or previous runs
for _handler_instance in _root_logger_instance.handlers[:]:
    _root_logger_instance.removeHandler(_handler_instance)

_valid_log_levels = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}
_final_log_level_val = _valid_log_levels.get(LOG_LEVEL, logging.INFO)
if LOG_LEVEL not in _valid_log_levels:
    # Log to the initial basicConfig logger before it's reconfigured
    logging.getLogger(__name__).warning(f"Invalid LOG_LEVEL '{LOG_LEVEL}' specified in environment. Defaulting to INFO.")

_root_logger_instance.setLevel(_final_log_level_val) # Set level on root logger

# Define a more detailed formatter, using pathname for better context
_log_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(pathname)s:%(lineno)d %(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

_console_handler_instance = logging.StreamHandler()
_console_handler_instance.setFormatter(_log_formatter)
_root_logger_instance.addHandler(_console_handler_instance)
# Use the module-level logger to announce reconfiguration
logger.info(f"Console logging re-configured. Level: {LOG_LEVEL}.")

if LOG_FILE:
    try:
        _log_dir = os.path.dirname(LOG_FILE)
        # Create log directory if it doesn't exist and LOG_FILE path includes a directory
        if _log_dir and not os.path.exists(_log_dir):
            logger.info(f"Creating log directory for file logging: '{_log_dir}'")
            os.makedirs(_log_dir, exist_ok=True)
        
        if not _log_dir and LOG_FILE: # If LOG_FILE is just a filename
             logger.info(f"Log file '{LOG_FILE}' will be created in the current working directory ({os.getcwd()}). Ensure this is writable in Docker if not using a volume.")

        from logging.handlers import RotatingFileHandler
        _file_handler_instance = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        _file_handler_instance.setFormatter(_log_formatter)
        _root_logger_instance.addHandler(_file_handler_instance)
        logger.info(f"Rotating file logging configured to '{LOG_FILE}'. Level: {LOG_LEVEL}.")
    except Exception as e:
        logger.error(f"Failed to configure file logging to '{LOG_FILE}': {e}", exc_info=True)
else:
    logger.info("File logging is not configured (LOG_FILE environment variable not set or empty). Logs will go to console only.")

# --- Final Sanity Checks & Info ---
logger.info("--- Configuration Constants Loaded and Logging Initialized ---")
if not BASE_WEBHOOK_URL:
    logger.critical("CRITICAL FAILURE: BASE_WEBHOOK_URL environment variable is not set. Twilio and other callback webhooks WILL FAIL.")
if not SUPABASE_ENABLED:
    logger.warning("Supabase persistence is DISABLED (SUPABASE_URL or SUPABASE_KEY environment variables are missing/empty). Database setup and operations will be skipped.")
if CLAY_API_KEY and not CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY:
    logger.warning("CLAY_API_KEY is set, but CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY is not. AcquisitionAgent Clay enrichment may fail unless webhook URL is discovered by other means.")
if CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY and not CLAY_RESULTS_CALLBACK_SECRET_TOKEN:
    logger.warning("CLAY_RESULTS_CALLBACK_SECRET_TOKEN not set for Clay Webhook. This endpoint is INSECURE.")

logger.info(f"Playwright will use REAL automator: {USE_REAL_BROWSER_AUTOMATOR}. Headless mode for Docker: {not PLAYWRIGHT_HEADFUL_MODE}")
if PLAYWRIGHT_HEADFUL_MODE and os.getenv("DOCKER_CONTAINER") == "true": # A common way to detect Docker env
    logger.warning("PLAYWRIGHT_HEADFUL_MODE is True but appears to be running in a Docker container (DOCKER_CONTAINER=true). This is usually problematic unless Xvfb is perfectly configured and display is forwarded.")

logger.info("Configuration loading process complete. Review CRITICAL/WARNING messages above to ensure all services will function as expected.")
# --------------------------------------------------------------------------------
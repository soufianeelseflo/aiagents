# config.py

import os
import logging
from dotenv import load_dotenv, find_dotenv
from typing import Optional

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
try:
    dotenv_path = find_dotenv(raise_error_if_not_found=False)
    if dotenv_path:
        logger.info(f"Loading environment variables from: {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path, override=True)
    else:
        logger.warning(".env file not found. Relying solely on system environment variables.")
except Exception as e:
     logger.error(f"Error finding or loading .env file: {e}", exc_info=True)

# --- Helper Function to Get Environment Variable ---
def get_env_var(var_name: str, required: bool = True, default: Optional[str] = None) -> Optional[str]:
    """ Retrieves an environment variable, logs status, and handles requirement checks. """
    value = os.getenv(var_name)
    if value is not None:
        is_secret = "KEY" in var_name or "TOKEN" in var_name or "SID" in var_name or "SECRET" in var_name or "PASSWORD" in var_name
        log_value = f"{value[:4]}...{value[-4:]}" if is_secret and len(value) > 8 else value
        logger.debug(f"Found environment variable '{var_name}': {log_value}")
        return value
    elif default is not None:
        logger.warning(f"Environment variable '{var_name}' not found. Using default value: '{default}'")
        return default
    elif required:
        logger.error(f"CRITICAL: Required environment variable '{var_name}' not found and no default provided.")
        raise ValueError(f"Missing required environment variable: {var_name}")
    else:
        logger.info(f"Optional environment variable '{var_name}' not found. Proceeding without it.")
        return None

# --- Core Configuration Constants ---

# Twilio Configuration
TWILIO_ACCOUNT_SID: Optional[str] = get_env_var("TWILIO_ACCOUNT_SID", required=True)
TWILIO_AUTH_TOKEN: Optional[str] = get_env_var("TWILIO_AUTH_TOKEN", required=True)
TWILIO_PHONE_NUMBER: Optional[str] = get_env_var("TWILIO_PHONE_NUMBER", required=True)

# Deepgram Configuration
DEEPGRAM_API_KEY: Optional[str] = get_env_var("DEEPGRAM_API_KEY", required=True)
DEEPGRAM_STT_MODEL: str = get_env_var("DEEPGRAM_STT_MODEL", required=False, default="nova-2-phonecall") or "nova-2-phonecall"
DEEPGRAM_TTS_MODEL: str = get_env_var("DEEPGRAM_TTS_MODEL", required=False, default="aura-asteria-en") or "aura-asteria-en"

# OpenRouter Configuration
OPENROUTER_API_KEY: Optional[str] = get_env_var("OPENROUTER_API_KEY", required=True)
OPENROUTER_MODEL_NAME: str = get_env_var("OPENROUTER_MODEL_NAME", required=False, default="openai/gpt-4o") or "openai/gpt-4o"
OPENROUTER_SITE_URL: Optional[str] = get_env_var("OPENROUTER_SITE_URL", required=False)
OPENROUTER_APP_NAME: Optional[str] = get_env_var("OPENROUTER_APP_NAME", required=False, default="BoutiqueAI")

# Resource Management & Data Sources
PROXY_USERNAME: Optional[str] = get_env_var("PROXY_USERNAME", required=False) # Required if using proxies via ResourceManager
PROXY_PASSWORD: Optional[str] = get_env_var("PROXY_PASSWORD", required=False) # Required if using proxies via ResourceManager
CLAY_API_KEY: Optional[str] = get_env_var("CLAY_API_KEY", required=False) # Key for Clay table interactions (direct or via RM)
RESOURCE_MANAGER_DATA_PATH: str = get_env_var("RESOURCE_MANAGER_DATA_PATH", required=False, default="data/resource_manager_state.json") or "data/resource_manager_state.json" # Used ONLY if Supabase fails/not configured
CLAY_API_BASE_URL: str = get_env_var("CLAY_API_BASE_URL", required=False, default="https://api.clay.com/v1") or "https://api.clay.com/v1" # Verify this

# --- NEW: Supabase Configuration ---
SUPABASE_URL: Optional[str] = get_env_var("SUPABASE_URL", required=False) # Required if using Supabase persistence
SUPABASE_KEY: Optional[str] = get_env_var("SUPABASE_KEY", required=False) # Required if using Supabase persistence
# --- Define Supabase Table Names ---
SUPABASE_CALL_LOG_TABLE: str = get_env_var("SUPABASE_CALL_LOG_TABLE", required=False, default="call_logs") or "call_logs"
SUPABASE_CONTACTS_TABLE: str = get_env_var("SUPABASE_CONTACTS_TABLE", required=False, default="contacts") or "contacts"
SUPABASE_RESOURCES_TABLE: str = get_env_var("SUPABASE_RESOURCES_TABLE", required=False, default="managed_resources") or "managed_resources"

# Agent Configuration
TARGET_PHONE_NUMBER_DEFAULT: Optional[str] = get_env_var("TARGET_PHONE_NUMBER_DEFAULT", required=False)
AGENT_TARGET_NICHE_DEFAULT: str = get_env_var("AGENT_TARGET_NICHE_DEFAULT", required=False, default="High-Value SaaS Sales") or "High-Value SaaS Sales"
AGENT_MAX_CALL_DURATION_DEFAULT: int = int(get_env_var("AGENT_MAX_CALL_DURATION_DEFAULT", required=False, default="3600") or "3600")

# Infrastructure / Webhook Configuration
BASE_WEBHOOK_URL: Optional[str] = get_env_var("BASE_WEBHOOK_URL", required=True)
LOCAL_SERVER_PORT: int = int(get_env_var("LOCAL_SERVER_PORT", required=False, default="8080") or "8080")

# Logging Configuration
LOG_LEVEL: str = get_env_var("LOG_LEVEL", required=False, default="INFO").upper() or "INFO"
LOG_FILE: Optional[str] = get_env_var("LOG_FILE", required=False)

# Caching Configuration
LLM_CACHE_SIZE: int = int(get_env_var("LLM_CACHE_SIZE", required=False, default="100") or "100") # Max items in LLM response cache

# --- Update Root Logger Based on Config ---
valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
final_log_level = LOG_LEVEL if LOG_LEVEL in valid_log_levels else "INFO"
if final_log_level != LOG_LEVEL:
    logger.warning(f"Invalid LOG_LEVEL '{LOG_LEVEL}' specified. Defaulting to {final_log_level}.")

root_logger = logging.getLogger()
for handler in root_logger.handlers[:]: root_logger.removeHandler(handler) # Clear existing handlers
root_logger.setLevel(final_log_level)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)
logger.info(f"Console logging configured with level {final_log_level}.")
if LOG_FILE:
    try:
        log_dir = os.path.dirname(LOG_FILE)
        if log_dir: os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logger.info(f"File logging configured to '{LOG_FILE}' with level {final_log_level}.")
    except Exception as e:
        logger.error(f"Failed to configure file logging to '{LOG_FILE}': {e}", exc_info=True)

# --- Log Final Configuration Summary (Debug Level) ---
logger.debug("--- Configuration Summary ---")
logger.debug(f"TWILIO_ACCOUNT_SID: {'Loaded' if TWILIO_ACCOUNT_SID else 'MISSING!'}")
logger.debug(f"TWILIO_PHONE_NUMBER: {TWILIO_PHONE_NUMBER}")
logger.debug(f"DEEPGRAM_API_KEY: {'Loaded' if DEEPGRAM_API_KEY else 'MISSING!'}")
logger.debug(f"OPENROUTER_API_KEY: {'Loaded' if OPENROUTER_API_KEY else 'MISSING!'}")
logger.debug(f"OPENROUTER_MODEL_NAME: {OPENROUTER_MODEL_NAME}")
logger.debug(f"PROXY_USERNAME: {'Loaded' if PROXY_USERNAME else 'Not Set (Optional)'}")
logger.debug(f"CLAY_API_KEY: {'Loaded' if CLAY_API_KEY else 'Not Set (Use ResourceManager)'}")
logger.debug(f"CLAY_API_BASE_URL: {CLAY_API_BASE_URL}")
logger.debug(f"SUPABASE_URL: {'Loaded' if SUPABASE_URL else 'Not Set (Persistence Disabled)'}")
logger.debug(f"SUPABASE_KEY: {'Loaded' if SUPABASE_KEY else 'Not Set (Persistence Disabled)'}")
logger.debug(f"SUPABASE_CALL_LOG_TABLE: {SUPABASE_CALL_LOG_TABLE}")
logger.debug(f"SUPABASE_CONTACTS_TABLE: {SUPABASE_CONTACTS_TABLE}")
logger.debug(f"SUPABASE_RESOURCES_TABLE: {SUPABASE_RESOURCES_TABLE}")
logger.debug(f"RESOURCE_MANAGER_DATA_PATH: {RESOURCE_MANAGER_DATA_PATH} (Fallback only)")
logger.debug(f"TARGET_PHONE_NUMBER_DEFAULT: {TARGET_PHONE_NUMBER_DEFAULT or 'Not Set'}")
logger.debug(f"AGENT_TARGET_NICHE_DEFAULT: {AGENT_TARGET_NICHE_DEFAULT}")
logger.debug(f"AGENT_MAX_CALL_DURATION_DEFAULT: {AGENT_MAX_CALL_DURATION_DEFAULT}")
logger.debug(f"BASE_WEBHOOK_URL: {BASE_WEBHOOK_URL or 'MISSING!'}")
logger.debug(f"LOCAL_SERVER_PORT: {LOCAL_SERVER_PORT}")
logger.debug(f"LOG_LEVEL: {final_log_level}")
logger.debug(f"LOG_FILE: {LOG_FILE or 'Not Set'}")
logger.debug(f"LLM_CACHE_SIZE: {LLM_CACHE_SIZE}")
logger.debug("--- End Configuration Summary ---")

logger.info("Configuration loading process complete.")

# --- Validate critical combinations ---
if not SUPABASE_URL or not SUPABASE_KEY:
     logger.warning("Supabase URL or Key not configured. ResourceManager state and CRM logging will NOT be persistent via Supabase.")
     # Consider falling back to file-based logging/state or raising an error depending on requirements


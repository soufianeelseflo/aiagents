# boutique_ai_project/config.py
import os
import logging
import json
from dotenv import load_dotenv, find_dotenv
from typing import Optional, Any, Dict, List, Union # Added Union

# --- Configure Logging (Basic setup, can be overridden by server.py's more detailed setup) ---
# This initial logging setup is crucial for config.py itself to log warnings.
# It will be further refined by server.py's main logging config.
logging.basicConfig(
    level=logging.INFO, # Default to INFO for config loading
    format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__) # Logger for this module

# --- Load Environment Variables ---
try:
    dotenv_path = find_dotenv(raise_error_if_not_found=False, usecwd=True)
    if dotenv_path:
        logger.info(f"Config: Loading environment variables from: {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path, override=True)
    else:
        logger.info("Config: .env file not found. Relying on system environment variables or coded defaults.")
except Exception as e:
     logger.error(f"Config: Error finding or loading .env file: {e}", exc_info=True)

# --- Helper Functions for Typed Environment Variable Retrieval ---
def get_env_var(var_name: str, default: Optional[Any] = None, required: bool = False) -> Optional[str]:
    value = os.getenv(var_name)
    if value is not None and value.strip() != "":
        is_secret = any(k in var_name.upper() for k in ["KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH", "SID"])
        log_value = f"{value[:4]}...{value[-4:]}" if is_secret and len(value) > 8 else value
        # logger.debug(f"Config: Found env var '{var_name}': {log_value}") # Too verbose for DEBUG, use INFO for actual load
        return value
    elif default is not None:
        logger.warning(f"Config: Env var '{var_name}' not found or empty. Using default: '{default}'")
        return str(default)
    elif required:
        msg = f"CRITICAL CONFIG ERROR: Required env var '{var_name}' is missing or empty. Application cannot start."
        logger.critical(msg)
        raise ValueError(msg)
    else:
        logger.info(f"Config: Optional env var '{var_name}' not found or empty. No default set, will be None.")
        return None

def get_int_env_var(var_name: str, default: Optional[int] = None, required: bool = False) -> Optional[int]:
    value_str = get_env_var(var_name, default=str(default) if default is not None else None, required=required)
    if value_str is None: return None # If not required and not found
    try: return int(value_str)
    except (ValueError, TypeError):
        msg = f"Config: Invalid integer value for env var '{var_name}': '{value_str}'."
        logger.error(msg)
        if required and default is None: raise ValueError(msg)
        logger.warning(f"Config: Using default int value {default} for '{var_name}' due to conversion error.")
        return default

def get_bool_env_var(var_name: str, default: bool = False) -> bool:
     value_str = get_env_var(var_name, default=str(default), required=False) # Booleans are usually not "required" to be missing
     if value_str is None: return default # Should not happen if default is always provided to get_env_var
     return value_str.lower() in ['true', '1', 'yes', 'y']

def get_json_env_var(var_name: str, default: Optional[Union[Dict, List]] = None, required: bool = False) -> Optional[Union[Dict, List]]:
    value_str = get_env_var(var_name, default=None, required=required)
    if value_str:
        try: return json.loads(value_str)
        except json.JSONDecodeError:
            logger.error(f"Config: Invalid JSON format for env var '{var_name}'. Value: '{value_str[:100]}...'")
            if required: raise ValueError(f"Invalid JSON for required env var '{var_name}'")
            logger.warning(f"Config: Using default JSON value {default} for '{var_name}' due to parsing error.")
            return default
    return default


# --- Core Configuration Constants ---

# Twilio (Essential - Must be set in environment)
TWILIO_ACCOUNT_SID: Optional[str] = get_env_var("TWILIO_ACCOUNT_SID", required=True)
TWILIO_AUTH_TOKEN: Optional[str] = get_env_var("TWILIO_AUTH_TOKEN", required=True)
TWILIO_PHONE_NUMBER: Optional[str] = get_env_var("TWILIO_PHONE_NUMBER", required=True)

# Deepgram (Essential - Must be set in environment)
DEEPGRAM_API_KEY: Optional[str] = get_env_var("DEEPGRAM_API_KEY", required=True)
DEEPGRAM_STT_MODEL: str = get_env_var("DEEPGRAM_STT_MODEL", default="nova-2-phonecall")
DEEPGRAM_TTS_MODEL: str = get_env_var("DEEPGRAM_TTS_MODEL", default="aura-asteria-en") # Example Aura voice

# OpenRouter (Essential - Must be set in environment)
OPENROUTER_API_KEY: Optional[str] = get_env_var("OPENROUTER_API_KEY", required=True)
OPENROUTER_DEFAULT_CONVERSATIONAL_MODEL: str = get_env_var("OPENROUTER_DEFAULT_CONVERSATIONAL_MODEL", default="google/gemini-1.5-flash-latest")
OPENROUTER_DEFAULT_STRATEGY_MODEL: str = get_env_var("OPENROUTER_DEFAULT_STRATEGY_MODEL", default="google/gemini-1.5-flash-latest")
OPENROUTER_DEFAULT_ANALYSIS_MODEL: str = get_env_var("OPENROUTER_DEFAULT_ANALYSIS_MODEL", default="google/gemini-1.5-pro-latest")
OPENROUTER_DEFAULT_VISION_MODEL: str = get_env_var("OPENROUTER_DEFAULT_VISION_MODEL", default="google/gemini-pro-vision") # Or "google/gemini-1.5-flash-latest" if it has good vision
OPENROUTER_SITE_URL: str = get_env_var("OPENROUTER_SITE_URL", default="https://github.com/soufianeelseflo/aiagents") # Default to your repo
OPENROUTER_APP_NAME: str = get_env_var("OPENROUTER_APP_NAME", default="BoutiqueAIAgent")
OPENROUTER_CLIENT_MAX_RETRIES: int = get_int_env_var("OPENROUTER_CLIENT_MAX_RETRIES", default=2)
OPENROUTER_CLIENT_TIMEOUT_SECONDS: int = get_int_env_var("OPENROUTER_CLIENT_TIMEOUT_SECONDS", default=120)

# Proxies (Optional - Set in environment if used by ResourceManager)
PROXY_USERNAME: Optional[str] = get_env_var("PROXY_USERNAME", required=False)
PROXY_PASSWORD: Optional[str] = get_env_var("PROXY_PASSWORD", required=False)

# Clay.com (Key parts essential if Clay integration is used)
CLAY_API_KEY: Optional[str] = get_env_var("CLAY_API_KEY", required=False)
CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY: Optional[str] = get_env_var("CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY", required=False) # Crucial for AcquisitionAgent
CLAY_RESULTS_CALLBACK_SECRET_TOKEN: Optional[str] = get_env_var("CLAY_RESULTS_CALLBACK_SECRET_TOKEN", required=False) # Highly recommended for security
# Less likely to change, so stronger defaults:
CLAY_API_BASE_URL: str = get_env_var("CLAY_API_BASE_URL", default="https://api.clay.com/v1")
CLAY_CALLBACK_AUTH_HEADER_NAME: str = get_env_var("CLAY_CALLBACK_AUTH_HEADER_NAME", default="X-Callback-Auth-Token")
CLAY_ENRICHMENT_TABLE_NAME: str = get_env_var("CLAY_ENRICHMENT_TABLE_NAME", default="Primary Lead Enrichment")


# Supabase (Essential - Must be set in environment)
SUPABASE_URL: Optional[str] = get_env_var("SUPABASE_URL", required=True)
SUPABASE_KEY: Optional[str] = get_env_var("SUPABASE_KEY", required=True) # Service Role Key
SUPABASE_ENABLED: bool = bool(SUPABASE_URL and SUPABASE_KEY)
# Table names - defaults are fine, can be overridden if schema is different
SUPABASE_CONTACTS_TABLE: str = get_env_var("SUPABASE_CONTACTS_TABLE", default="contacts")
SUPABASE_CALL_LOGS_TABLE: str = get_env_var("SUPABASE_CALL_LOGS_TABLE", default="call_logs")
SUPABASE_RESOURCES_TABLE: str = get_env_var("SUPABASE_RESOURCES_TABLE", default="managed_resources")
# New tables for agent factory
SUPABASE_AGENT_TEMPLATES_TABLE: str = get_env_var("SUPABASE_AGENT_TEMPLATES_TABLE", default="agent_templates")
SUPABASE_CLIENT_AGENT_INSTANCES_TABLE: str = get_env_var("SUPABASE_CLIENT_AGENT_INSTANCES_TABLE", default="client_agent_instances")


# Agent Behavior Defaults (Can be overridden by environment variables for specific deployments)
AGENT_TARGET_NICHE_DEFAULT: str = get_env_var("AGENT_TARGET_NICHE_DEFAULT", default="Innovative Tech Startups")
AGENT_MAX_CALL_DURATION_DEFAULT: int = get_int_env_var("AGENT_MAX_CALL_DURATION_DEFAULT", default=2700) # 45 mins
ACQUISITION_AGENT_RUN_INTERVAL_SECONDS: int = get_int_env_var("ACQUISITION_AGENT_RUN_INTERVAL_SECONDS", default=1800) # 30 mins
ACQUISITION_AGENT_BATCH_SIZE: int = get_int_env_var("ACQUISITION_AGENT_BATCH_SIZE", default=15)
ACQ_LEAD_SOURCE_TYPE: str = get_env_var("ACQ_LEAD_SOURCE_TYPE", default="supabase_query") # "supabase_query" or "local_csv"
ACQ_LEAD_SOURCE_PATH: str = get_env_var("ACQ_LEAD_SOURCE_PATH", default="data/initial_leads.csv")
_default_csv_mapping_str = '{"company_name": "CompanyName", "domain": "Domain", "email": "Email"}'
ACQ_LEAD_SOURCE_CSV_MAPPING: Dict = get_json_env_var("ACQ_LEAD_SOURCE_CSV_MAPPING", default=json.loads(_default_csv_mapping_str))
ACQ_SUPABASE_PENDING_LEAD_STATUS: str = get_env_var("ACQ_SUPABASE_PENDING_LEAD_STATUS", default="New_Raw_Lead")
ACQ_QUALIFICATION_THRESHOLD: int = get_int_env_var("ACQ_QUALIFICATION_THRESHOLD", default=6) # Score 1-10

# Provisioning Agent
PROVISIONING_AGENT_INTERVAL_S: int = get_int_env_var("PROVISIONING_AGENT_INTERVAL_S", default=300) # 5 mins
PROVISIONING_AGENT_MAX_CONCURRENT: int = get_int_env_var("PROVISIONING_AGENT_MAX_CONCURRENT", default=2)

# Server & Webhooks (BASE_WEBHOOK_URL is critical and MUST be set in env)
BASE_WEBHOOK_URL: Optional[str] = get_env_var("BASE_WEBHOOK_URL", required=True)
LOCAL_SERVER_PORT: int = get_int_env_var("LOCAL_SERVER_PORT", default=8080)
UVICORN_RELOAD: bool = get_bool_env_var("UVICORN_RELOAD", default=False) # Should be False in production

# System Settings & Browser Automation (Defaults are generally fine, override if needed)
LOG_LEVEL: str = get_env_var("LOG_LEVEL", default="INFO").upper()
LOG_FILE: Optional[str] = get_env_var("LOG_FILE", required=False) # e.g., "logs/boutique_ai_app.log"
LLM_CACHE_SIZE: int = get_int_env_var("LLM_CACHE_SIZE", default=100)
USE_REAL_BROWSER_AUTOMATOR: bool = get_bool_env_var("USE_REAL_BROWSER_AUTOMATOR", default=False)
PLAYWRIGHT_HEADFUL_MODE: bool = get_bool_env_var("PLAYWRIGHT_HEADFUL_MODE", default=False) # For local debugging if real automator is used
PLAYWRIGHT_NAV_TIMEOUT_MS: int = get_int_env_var("PLAYWRIGHT_NAV_TIMEOUT_MS", default=60000)
PLAYWRIGHT_ACTION_TIMEOUT_MS: int = get_int_env_var("PLAYWRIGHT_ACTION_TIMEOUT_MS", default=30000)
PLAYWRIGHT_SELECTOR_TIMEOUT_MS: int = get_int_env_var("PLAYWRIGHT_SELECTOR_TIMEOUT_MS", default=15000)
MAX_CONCURRENT_BROWSER_AUTOMATIONS: int = get_int_env_var("MAX_CONCURRENT_BROWSER_AUTOMATIONS", default=1)


# --- Apply Final Logging Configuration (after all env vars are loaded) ---
# This part is primarily for ensuring the root logger is configured if not already done well by uvicorn/FastAPI.
# Uvicorn often takes over logging, so this might be redundant or just for non-uvicorn script runs.
_final_log_level_val = getattr(logging, LOG_LEVEL, logging.INFO)
# Reconfigure root logger if necessary, or just ensure module loggers inherit correctly.
# For now, assume uvicorn/FastAPI will manage handlers, and we're setting the level for our modules.
logging.getLogger().setLevel(_final_log_level_val) # Set root logger level
for handler in logging.getLogger().handlers: # Apply level to existing handlers
    handler.setLevel(_final_log_level_val)

logger.info("--- Boutique AI Configuration Loaded ---")
if not SUPABASE_ENABLED: logger.warning("Config: Supabase persistence is DISABLED (URL or KEY missing).")
else: logger.info(f"Config: Supabase ENABLED. URL: {SUPABASE_URL[:20]}...") # Don't log full URL if sensitive part of ref
if not BASE_WEBHOOK_URL: logger.critical("CRITICAL CONFIG: BASE_WEBHOOK_URL is not set. Twilio & Clay webhooks WILL FAIL.")
if not CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY and ACQ_LEAD_SOURCE_TYPE != 'local_csv':
    logger.warning("Config: CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY not set. AcquisitionAgent Clay enrichment may fail or attempt discovery.")
if CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY and not CLAY_RESULTS_CALLBACK_SECRET_TOKEN:
    logger.warning("Config: CLAY_RESULTS_CALLBACK_SECRET_TOKEN not set. Clay results webhook is INSECURE.")

logger.info(f"Config: Default Niche: {AGENT_TARGET_NICHE_DEFAULT}")
logger.info(f"Config: Uvicorn Reload (config.py perspective): {UVICORN_RELOAD}")
logger.info(f"Config: Use REAL Browser Automator: {USE_REAL_BROWSER_AUTOMATOR}")
logger.info(f"Config: Playwright Headful Mode (config.py perspective): {PLAYWRIGHT_HEADFUL_MODE}")
logger.info(f"Config: Default Log Level: {LOG_LEVEL}")
logger.info("--- Configuration loading process complete. Review any warnings above. ---")
# core/agents/resource_manager.py

import logging
import json
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

# Import dependencies from other modules
from core.services.proxy_manager_wrapper import ProxyManagerWrapper
# DataWrapper no longer needed here directly, API key is passed to it when used
# from core.services.data_wrapper import DataWrapper
from config import (
    RESOURCE_MANAGER_DATA_PATH,
    CLAY_API_KEY # Direct Clay API Key from config (can be None)
)

logger = logging.getLogger(__name__)

# --- WARNING ---
# AUTOMATING TRIAL SIGNUPS IS TECHNICALLY COMPLEX, HIGHLY PRONE TO BREAKING,
# AND VERY LIKELY VIOLATES THE TERMS OF SERVICE OF MOST PLATFORMS (LIKE CLAY.COM).
# THIS CAN LEAD TO ACCOUNT/IP BANS AND POTENTIAL LEGAL ISSUES.
# The `_execute_automated_signup` function below remains a STRUCTURAL PLACEHOLDER ONLY
# and is NOT functionally implemented due to these risks.
# This manager PRIMARILY handles proxy retrieval and API key management based on config/state.
# --- WARNING ---

class ResourceManager:
    """
    Manages access to external resources like residential proxies (via connection strings)
    and API keys (specifically Clay.com, sourced via config or conceptual trial state).
    Implements persistence for conceptual trial account state.
    """

    def __init__(self):
        """ Initializes the ResourceManager. """
        self.proxy_manager = ProxyManagerWrapper()
        self.trials_file_path = Path(RESOURCE_MANAGER_DATA_PATH)
        # Stores conceptual trial info (username, pass, api_key, expiry) if generated
        self.active_trials: Dict[str, List[Dict[str, Any]]] = self._load_trials_state()
        # Store the directly configured Clay API key, if provided
        self.direct_clay_api_key: Optional[str] = CLAY_API_KEY

        logger.info(f"ResourceManager initialized. State file: {self.trials_file_path}")
        if self.direct_clay_api_key:
             logger.info("Direct Clay API Key loaded from configuration.")

    # --- State Persistence ---
    def _load_trials_state(self) -> Dict[str, List[Dict[str, Any]]]:
        """Loads the state of tracked conceptual trial accounts from the JSON file."""
        try:
            if self.trials_file_path.exists():
                with open(self.trials_file_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    logger.info(f"Loaded conceptual trial state from {self.trials_file_path}")
                    return state if isinstance(state, dict) else {}
            else:
                logger.info("No existing conceptual trial state file found.")
                return {}
        except (json.JSONDecodeError, IOError, Exception) as e:
            logger.error(f"Error loading conceptual trial state from {self.trials_file_path}: {e}", exc_info=True)
            return {}

    def _save_trials_state(self):
        """Saves the current state of tracked conceptual trial accounts to the JSON file."""
        try:
            self.trials_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.trials_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.active_trials, f, indent=4, default=str)
            logger.debug(f"Saved conceptual trial state to {self.trials_file_path}")
        except (IOError, TypeError, Exception) as e:
            logger.error(f"Error saving conceptual trial state to {self.trials_file_path}: {e}", exc_info=True)

    # --- Proxy Management ---
    def get_proxy_connection_string(
        self,
        proxy_username: str, # Credentials must be provided per request
        proxy_password: str,
        location: str = "random",
        session_type: str = "rotating"
        ) -> Optional[str]:
        """
        Retrieves a proxy connection string using the ProxyManagerWrapper.
        Requires credentials to be passed in, as they might vary per task/trial account.
        """
        logger.info(f"Requesting proxy connection string for location '{location}', session '{session_type}'...")
        proxy_string = self.proxy_manager.get_proxy_string(
            proxy_username=proxy_username,
            proxy_password=proxy_password,
            location=location,
            session_type=session_type
        )
        if proxy_string:
            logger.info("Proxy connection string obtained successfully.")
        else:
            logger.error("Failed to obtain proxy connection string.")
        return proxy_string

    # --- Trial Account Management & API Key Retrieval ---
    def _is_trial_valid(self, trial_info: Dict[str, Any]) -> bool:
        """ Checks if a stored conceptual trial is still valid based on expiry. """
        expiry = trial_info.get("expiry_timestamp")
        if expiry and isinstance(expiry, (int, float)):
            is_valid = time.time() < expiry
            logger.debug(f"Checking trial validity: Expiry {expiry} vs Now {time.time()}. Valid: {is_valid}")
            return is_valid
        creation_time = trial_info.get("creation_timestamp", 0)
        default_lifespan = 7 * 24 * 3600 # Assume 7 days if no expiry
        is_valid_default = (time.time() - creation_time) < default_lifespan
        logger.debug(f"Checking trial validity (no expiry): Creation {creation_time}. Default Valid: {is_valid_default}")
        return is_valid_default

    def _find_valid_trial(self, service_name: str) -> Optional[Dict[str, Any]]:
        """ Finds the first valid conceptual trial for a service from stored state. """
        service_name_lower = service_name.lower()
        if service_name_lower in self.active_trials:
            valid_trials = [t for t in self.active_trials[service_name_lower] if self._is_trial_valid(t)]
            if valid_trials:
                logger.info(f"Found {len(valid_trials)} valid conceptual trial(s) for {service_name}. Using one.")
                return valid_trials[0]
            else:
                 logger.info(f"No valid existing conceptual trials for {service_name}. Cleaning up expired.")
                 self.active_trials[service_name_lower] = []
                 self._save_trials_state()
        return None

    def _execute_automated_signup(self, service_name: str, proxy: Optional[str]) -> Optional[Dict[str, Any]]:
        """ *** CONCEPTUAL PLACEHOLDER - HIGH RISK *** """
        proxy_info = f"via proxy {proxy[:20]}..." if proxy else "without proxy"
        logger.warning(f"Attempting CONCEPTUAL trial signup for {service_name} {proxy_info}")
        logger.error("Automated signup function (`_execute_automated_signup`) is NOT IMPLEMENTED due to high risk and complexity.")
        # Simulate success/failure
        if hash(service_name + str(time.time())) % 5 == 0: # Simulate success 1/5
             dummy_credentials = {
                 "service": service_name.lower(),
                 "username": f"dummy_{service_name.lower()}_{int(time.time())}",
                 "password": "dummy_password",
                 # Simulate finding an API key during signup (key for Clay retrieval)
                 "api_key": f"CLAY_DUMMY_KEY_{int(time.time())}" if service_name.lower() == "clay.com" else None,
                 "creation_timestamp": time.time(),
                 "expiry_timestamp": time.time() + (7 * 24 * 3600) # Assume 7 day trial
             }
             logger.info(f"CONCEPTUAL signup SUCCEEDED for {service_name}. Returning dummy credentials.")
             return dummy_credentials
        else:
             logger.error(f"CONCEPTUAL signup FAILED for {service_name}.")
             return None

    def _get_or_create_conceptual_trial(self, service_name: str) -> Optional[Dict[str, Any]]:
        """ Internal logic to get a valid conceptual trial or attempt creation. """
        service_name_lower = service_name.lower()
        valid_trial = self._find_valid_trial(service_name_lower)
        if valid_trial:
            return valid_trial

        logger.info(f"Attempting to create a new conceptual trial for {service_name_lower}.")
        # Conceptual signup doesn't need real proxy credentials here
        signup_proxy_string = None # Simulate not using proxy for conceptual signup

        new_trial_info = self._execute_automated_signup(service_name_lower, signup_proxy_string)

        if new_trial_info:
            if service_name_lower not in self.active_trials:
                self.active_trials[service_name_lower] = []
            self.active_trials[service_name_lower].append(new_trial_info)
            self._save_trials_state()
            logger.info(f"Successfully obtained and stored new conceptual trial for {service_name_lower}.")
            return new_trial_info
        else:
            logger.error(f"Failed to create new conceptual trial for {service_name_lower}.")
            return None

    # --- Public Resource Access Methods ---

    def get_clay_api_key(self) -> Optional[str]:
        """
        Retrieves the Clay.com API key.
        Prioritizes the directly configured key from .env (config.CLAY_API_KEY).
        If not found, attempts to get one from the conceptual trial management state.

        Returns:
            The Clay.com API key string or None if unavailable.
        """
        logger.info("Requesting Clay.com API key...")
        if self.direct_clay_api_key:
            logger.info("Using directly configured Clay.com API key.")
            return self.direct_clay_api_key

        logger.info("Direct Clay API key not found in config, attempting retrieval via conceptual trial management.")
        trial_info = self._get_or_create_conceptual_trial("clay.com")

        if trial_info and trial_info.get("api_key"):
            logger.info("Found conceptual Clay.com API key from trial management state.")
            return trial_info["api_key"]
        else:
            logger.error("Failed to obtain Clay.com API key via conceptual trial management.")
            return None

    def get_clay_credentials(self) -> Optional[Dict[str, str]]:
        """
        Retrieves conceptual username/password credentials for Clay.com
        from the trial management state. Needed for potential browser automation.

        Returns:
            Dict with 'username' and 'password', or None if unavailable.
        """
        logger.info("Requesting conceptual Clay.com credentials via trial management...")
        trial_info = self._get_or_create_conceptual_trial("clay.com")

        if trial_info and trial_info.get("username") and trial_info.get("password"):
            logger.info("Found conceptual Clay.com credentials from trial management state.")
            return {
                "username": trial_info["username"],
                "password": trial_info["password"]
            }
        else:
            logger.error("Failed to obtain conceptual Clay.com credentials via trial management.")
            return None

# (Conceptual Test Runner - Keep as is)
def main():
    print("Testing ResourceManager (v2)...")
    # ... (rest of test logic remains the same) ...
    try:
        manager = ResourceManager()
        # ... (rest of test logic remains the same) ...
        print("\n--- Testing Proxy Retrieval ---")
        proxy_user = "test_proxy_user"; proxy_pass = "test_proxy_pass"
        proxy_str = manager.get_proxy_connection_string(proxy_user, proxy_pass, location="de")
        if proxy_str: print(f"Got proxy string: http://{proxy_user}:*****@{proxy_str.split('@')[1]}")
        else: print("Failed to get proxy string.")
        print("\n--- Testing Clay.com API Key Retrieval ---")
        clay_key = manager.get_clay_api_key()
        if clay_key:
            source = "direct config" if manager.direct_clay_api_key else "conceptual trial"
            print(f"Got Clay API Key (from {source}): {clay_key[:15]}...")
        else: print("Failed to get Clay API Key.")
        print("\n--- Testing Clay.com Credential Retrieval ---")
        creds = manager.get_clay_credentials()
        if creds: print(f"Got Clay Credentials (conceptual trial): username={creds['username']}, password=*****")
        else: print("Failed to get Clay Credentials.")
        print("\n--- Current Conceptual Trials State ---")
        print(json.dumps(manager.active_trials, indent=2))
    except Exception as e: print(f"An error occurred during test: {e}")


if __name__ == "__main__":
    # import config # Ensure config is loaded
    # main() # Uncomment to run test
    print("ResourceManager structure defined (v2). Run test manually.")


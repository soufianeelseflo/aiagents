# core/agents/resource_manager.py

import logging
import json
import time
import asyncio
from datetime import datetime, timezone, timedelta # Added timedelta
from typing import Dict, Any, Optional, List

# Import Supabase client library and potential errors
from supabase import create_client, Client as SupabaseClient, PostgrestAPIError

# Import dependencies from other modules
from core.services.proxy_manager_wrapper import ProxyManagerWrapper
from core.services.fingerprint_generator import FingerprintGenerator # Import new generator
from core.services.llm_client import LLMClient # Needed to instantiate FingerprintGenerator
# Import configuration centrally
try:
    from config import (
        SUPABASE_URL, SUPABASE_KEY, SUPABASE_RESOURCES_TABLE, # Supabase config
        CLAY_API_KEY # Direct Clay API Key from config (can be None)
    )
    SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_KEY)
except ImportError:
    logger.error("Failed to import Supabase config. ResourceManager persistence disabled.")
    SUPABASE_URL, SUPABASE_KEY = None, None
    SUPABASE_RESOURCES_TABLE = "managed_resources" # Default won't matter
    SUPABASE_ENABLED = False
    CLAY_API_KEY = None # Assume None if config fails

logger = logging.getLogger(__name__)

# --- WARNING ---
# AUTOMATING TRIAL SIGNUPS IS TECHNICALLY COMPLEX, HIGHLY PRONE TO BREAKING,
# AND VERY LIKELY VIOLATES THE TERMS OF SERVICE OF MOST PLATFORMS (LIKE CLAY.COM).
# THIS CAN LEAD TO ACCOUNT/IP BANS AND POTENTIAL LEGAL ISSUES.
# The `_execute_automated_signup` function below remains a STRUCTURAL PLACEHOLDER ONLY
# and is NOT functionally implemented due to these risks.
# This manager PRIMARILY handles proxy retrieval and API key management based on config/Supabase state.
# --- WARNING ---

class ResourceManager:
    """
    Manages access to external resources like proxies and API keys (Clay.com focus).
    Uses Supabase for persistence of conceptual trial account state if configured.
    Incorporates FingerprintGenerator to associate realistic profiles with resources.
    """

    def __init__(self, llm_client: LLMClient): # Requires LLMClient for FingerprintGenerator
        """
        Initializes the ResourceManager and Supabase client if configured.

        Args:
            llm_client: An initialized LLMClient instance for the FingerprintGenerator.
        """
        self.proxy_manager = ProxyManagerWrapper()
        self.fingerprint_generator = FingerprintGenerator(llm_client) # Instantiate fingerprint generator
        self.supabase: Optional[SupabaseClient] = None
        self.resources_table: str = SUPABASE_RESOURCES_TABLE
        self.direct_clay_api_key: Optional[str] = CLAY_API_KEY

        if SUPABASE_ENABLED:
            try:
                self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
                logger.info(f"ResourceManager: Supabase client initialized for table '{self.resources_table}'.")
            except Exception as e:
                logger.error(f"ResourceManager: Failed to initialize Supabase client: {e}", exc_info=True)
                self.supabase = None
        else:
            logger.warning("ResourceManager: Supabase not configured. Conceptual trial state persistence is disabled.")

        if self.direct_clay_api_key:
             logger.info("ResourceManager: Direct Clay API Key loaded from configuration.")

    # --- Proxy Management ---
    def get_proxy_connection_string(
        self,
        proxy_username: str,
        proxy_password: str,
        location: str = "random",
        session_type: str = "rotating"
        ) -> Optional[str]:
        """ Retrieves a proxy connection string using the ProxyManagerWrapper. """
        logger.info(f"Requesting proxy connection string for location '{location}', session '{session_type}'...")
        # This method remains unchanged, as it relies on provided credentials
        return self.proxy_manager.get_proxy_string(
            proxy_username=proxy_username, proxy_password=proxy_password,
            location=location, session_type=session_type
        )

    # --- Conceptual Trial Account Management (State in Supabase) ---

    async def _find_valid_conceptual_trial(self, service_name: str) -> Optional[Dict[str, Any]]:
        """
        Finds a valid conceptual trial for a service by querying the Supabase table.
        Checks the 'expiry_timestamp' against the current time.
        Assumes table schema includes necessary fields like 'service', 'expiry_timestamp'.
        """
        if not self.supabase: return None
        service_name_lower = service_name.lower()
        current_time_iso = datetime.now(timezone.utc).isoformat()
        logger.debug(f"Querying Supabase '{self.resources_table}' for valid trial for '{service_name_lower}' expiring after {current_time_iso}")
        try:
            # Query for a trial that hasn't expired yet
            response = await asyncio.to_thread(
                self.supabase.table(self.resources_table)
                .select("*") # Select all columns for the resource
                .eq("service", service_name_lower)
                .gt("expiry_timestamp", current_time_iso) # expiry > now
                .order("creation_timestamp", desc=False) # Use oldest valid one first
                .limit(1)
                .execute
            )
            if response.data:
                valid_trial = response.data[0]
                logger.info(f"Found valid conceptual trial for {service_name_lower} in Supabase. ID: {valid_trial.get('id')}")
                return valid_trial
            else:
                logger.info(f"No valid (non-expired) conceptual trials found for {service_name_lower} in Supabase.")
                return None
        except PostgrestAPIError as e:
            logger.error(f"Supabase API error finding valid trial for '{service_name_lower}': {e.message}", exc_info=False)
            return None
        except Exception as e:
            logger.error(f"Unexpected error finding valid trial for '{service_name_lower}': {e}", exc_info=True)
            return None

    def _execute_automated_signup(self, service_name: str, proxy: Optional[str]) -> Optional[Dict[str, Any]]:
        """ *** CONCEPTUAL PLACEHOLDER - HIGH RISK - REMAINS UNIMPLEMENTED *** """
        proxy_info = f"via proxy {proxy[:20]}..." if proxy else "without proxy"
        logger.warning(f"Attempting CONCEPTUAL trial signup for {service_name} {proxy_info}")
        logger.error("Automated signup function (`_execute_automated_signup`) is NOT IMPLEMENTED due to high risk and complexity.")
        # Simulate success/failure
        if hash(service_name + str(time.time())) % 5 == 0: # Simulate success 1/5
             creation_ts = datetime.now(timezone.utc)
             expiry_ts = creation_ts + timedelta(days=7) # Simulate 7 day trial
             dummy_data = {
                 "service": service_name.lower(),
                 "username": f"dummy_{service_name.lower()}_{int(creation_ts.timestamp())}",
                 "password": "dummy_password",
                 "api_key": f"CLAY_DUMMY_KEY_{int(creation_ts.timestamp())}" if service_name.lower() == "clay.com" else None,
                 "creation_timestamp": creation_ts.isoformat(),
                 "expiry_timestamp": expiry_ts.isoformat(),
                 "status": "active",
                 # Store proxy user/pass if this trial needs specific ones, otherwise use global/passed-in creds
                 # "proxy_username": f"proxy_user_{int(creation_ts.timestamp())}",
                 # "proxy_password": "proxy_password_for_trial",
                 # Fingerprint profile could be generated here and stored as JSONB
                 # "fingerprint_profile": json.dumps(await self.fingerprint_generator.generate_profile(...)) # Needs async context
             }
             logger.info(f"CONCEPTUAL signup SUCCEEDED for {service_name}. Returning dummy data.")
             return dummy_data
        else:
             logger.error(f"CONCEPTUAL signup FAILED for {service_name}.")
             return None

    async def _create_conceptual_trial_record(self, service_name: str) -> Optional[Dict[str, Any]]:
        """
        Attempts the conceptual signup process and inserts the resulting record
        (if successful) into the Supabase table. Includes generating a fingerprint.
        """
        if not self.supabase: return None

        logger.info(f"Attempting to create and store a new conceptual trial record for {service_name}.")
        signup_proxy_string = None # Placeholder proxy for conceptual signup

        new_trial_data = self._execute_automated_signup(service_name.lower(), signup_proxy_string)

        if new_trial_data:
            try:
                # Generate a fingerprint profile to associate with this conceptual account
                logger.debug("Generating fingerprint profile for new conceptual trial...")
                fingerprint_profile = await self.fingerprint_generator.generate_profile(
                    role_context=f"User of {service_name}", # Basic context
                    os_type=random.choice(["Windows", "macOS"]) # Randomize OS
                )
                # Store the profile as JSONB in Supabase (assuming table has a jsonb column 'fingerprint_profile')
                new_trial_data['fingerprint_profile'] = fingerprint_profile # Store the dict directly, supabase client handles JSON

                logger.debug(f"Inserting conceptual trial data into Supabase table '{self.resources_table}': {new_trial_data}")
                response = await asyncio.to_thread(
                    self.supabase.table(self.resources_table)
                    .insert(new_trial_data)
                    .execute
                )
                if response.data:
                    logger.info(f"Successfully stored new conceptual trial record for {service_name} in Supabase. ID: {response.data[0].get('id')}")
                    return response.data[0] # Return the stored record
                else:
                    logger.error(f"Supabase insert for conceptual trial '{service_name}' completed but returned no data. Response: {response}")
                    return None
            except PostgrestAPIError as e:
                 logger.error(f"Supabase API error inserting conceptual trial record for '{service_name}': {e.message}", exc_info=False)
                 return None
            except Exception as e:
                 logger.error(f"Unexpected error storing conceptual trial record for '{service_name}': {e}", exc_info=True)
                 return None
        else:
            logger.error(f"Conceptual trial signup failed for {service_name}, cannot store record.")
            return None

    async def _get_or_create_conceptual_trial(self, service_name: str) -> Optional[Dict[str, Any]]:
        """ Internal logic to get a valid conceptual trial from Supabase or attempt creation. """
        valid_trial = await self._find_valid_conceptual_trial(service_name)
        if valid_trial:
            return valid_trial
        else:
            # If no valid trial exists, attempt to create one conceptually and store it
            return await self._create_conceptual_trial_record(service_name)


    # --- Public Resource Access Methods ---

    async def get_clay_api_key(self) -> Optional[str]:
        """
        Retrieves the Clay.com API key.
        1. Checks direct config (CLAY_API_KEY).
        2. If not found, queries Supabase for a valid conceptual trial key.
        3. If none found, attempts to create/store a new conceptual trial record.

        Returns:
            The Clay.com API key string or None if unavailable.
        """
        logger.info("Requesting Clay.com API key...")
        if self.direct_clay_api_key:
            logger.info("Using directly configured Clay.com API key.")
            return self.direct_clay_api_key

        if not self.supabase:
             logger.warning("Supabase not configured. Cannot retrieve conceptual Clay API key from database.")
             return None

        logger.info("Direct Clay API key not found, checking/creating conceptual trial via Supabase...")
        trial_info = await self._get_or_create_conceptual_trial("clay.com")

        if trial_info and trial_info.get("api_key"):
            logger.info("Found/obtained conceptual Clay.com API key via Supabase.")
            return trial_info["api_key"]
        else:
            logger.error("Failed to obtain Clay.com API key via conceptual trial management in Supabase.")
            return None

    async def get_resource_bundle(self, service_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a bundle of resources needed for a task involving a specific service,
        including conceptual credentials, API key, proxy string, and fingerprint profile.

        Args:
            service_name: The name of the service (e.g., "clay.com").

        Returns:
            A dictionary containing 'credentials', 'api_key', 'proxy_string',
            'fingerprint_profile', or None if essential resources cannot be obtained.
        """
        logger.info(f"Requesting resource bundle for service: {service_name}")

        # 1. Get Conceptual Trial Info (includes credentials, API key, fingerprint)
        trial_info = await self._get_or_create_conceptual_trial(service_name)
        if not trial_info:
             logger.error(f"Failed to get conceptual trial info for service '{service_name}'. Cannot provide bundle.")
             return None

        credentials = {
            "username": trial_info.get("username"),
            "password": trial_info.get("password")
        }
        api_key = trial_info.get("api_key")
        fingerprint_profile = trial_info.get("fingerprint_profile") # Assumes stored as JSON/dict

        # Check if essential parts are missing from trial info
        if not credentials["username"] or not credentials["password"]:
             logger.warning(f"Conceptual trial info for '{service_name}' missing username/password.")
             # Decide if bundle can proceed without credentials? Depends on use case.
             # For now, let's allow it but log warning.

        # 2. Get Proxy String (Requires providing credentials)
        # Determine which proxy credentials to use.
        # Option A: Use credentials associated with the specific trial (if stored)
        proxy_user = trial_info.get("proxy_username") # Assumes these fields exist in Supabase table
        proxy_pass = trial_info.get("proxy_password")
        # Option B: Use globally configured proxy credentials (from config.py)
        if not proxy_user or not proxy_pass:
             from config import PROXY_USERNAME, PROXY_PASSWORD # Import here if needed
             proxy_user = PROXY_USERNAME
             proxy_pass = PROXY_PASSWORD

        proxy_string = None
        if proxy_user and proxy_pass:
            # Get a proxy string using the determined credentials
            proxy_string = self.get_proxy_connection_string(
                proxy_username=proxy_user,
                proxy_password=proxy_pass,
                location="random" # Or make location configurable per service/task
            )
            if not proxy_string:
                 logger.error(f"Failed to obtain proxy string for service '{service_name}' bundle.")
                 # Decide if this is fatal for the bundle? Often yes for automation.
                 # return None # Make proxy essential
        else:
             logger.warning(f"Proxy credentials not available for service '{service_name}' bundle. Proxy not included.")


        # 3. Assemble the bundle
        bundle = {
            "service": service_name,
            "credentials": credentials if credentials["username"] else None, # Only return if valid
            "api_key": api_key,
            "proxy_string": proxy_string,
            "fingerprint_profile": fingerprint_profile # Can be None if not generated/stored
        }
        logger.info(f"Resource bundle assembled for service '{service_name}'.")
        logger.debug(f"Bundle details: Credentials={'Set' if bundle['credentials'] else 'None'}, APIKey={'Set' if bundle['api_key'] else 'None'}, Proxy={'Set' if bundle['proxy_string'] else 'None'}, Fingerprint={'Set' if bundle['fingerprint_profile'] else 'None'}")

        return bundle


# (Conceptual Test Runner - Requires Supabase Setup & LLMClient for Fingerprint)
async def main():
    print("Testing ResourceManager (v4 - Supabase & Fingerprint)...")
    if not SUPABASE_ENABLED: print("Skipping test: Supabase not configured."); return
    if not config.OPENROUTER_API_KEY: print("Skipping test: OpenRouter key needed for FingerprintGenerator."); return

    try:
        # Need LLMClient for FingerprintGenerator used by ResourceManager
        llm_client_for_rm = LLMClient()
        manager = ResourceManager(llm_client=llm_client_for_rm)
        if not manager.supabase: print("Skipping test: Supabase client failed init."); return

        print("\n--- Testing Clay.com Resource Bundle Retrieval ---")
        clay_bundle = await manager.get_resource_bundle("clay.com")
        if clay_bundle:
            print("Got Clay.com Resource Bundle:")
            print(f"  - Service: {clay_bundle.get('service')}")
            print(f"  - Credentials: {'Username Set' if clay_bundle.get('credentials') else 'None'}")
            print(f"  - API Key: {'Present' if clay_bundle.get('api_key') else 'None'}")
            print(f"  - Proxy String: {'Present' if clay_bundle.get('proxy_string') else 'None'}")
            print(f"  - Fingerprint: {'Present' if clay_bundle.get('fingerprint_profile') else 'None'}")
            if clay_bundle.get('fingerprint_profile'):
                 print(f"    - Example Header: User-Agent = {clay_bundle['fingerprint_profile'].get('headers', {}).get('User-Agent', 'N/A')}")
        else:
            print("Failed to get Clay.com resource bundle.")

    except Exception as e:
        print(f"An error occurred during test: {e}")

if __name__ == "__main__":
    # import asyncio
    # import config # Ensure config is loaded
    # from core.services.llm_client import LLMClient # Import for test
    # asyncio.run(main()) # Uncomment to run test (requires async context, Supabase, OpenRouter key)
    print("ResourceManager structure defined (v4 - Supabase & Fingerprint). Run test manually.")


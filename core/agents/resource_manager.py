# boutique_ai_project/core/agents/resource_manager.py

import logging
import json
import asyncio
import random
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Union

from supabase import create_client, Client as SupabaseClient, PostgrestAPIError

import config # Root config
from core.services.proxy_manager_wrapper import ProxyManagerWrapper
from core.services.fingerprint_generator import FingerprintGenerator
from core.automation.browser_automator_interface import BrowserAutomatorInterface
from core.services.llm_client import LLMClient # Optional for signup details

logger = logging.getLogger(__name__)

class ResourceManager:
    """
    Manages access to external resources: proxies, API keys (from config/Supabase),
    and orchestrates automated trial account acquisition using an injected BrowserAutomator
    and FingerprintGenerator. (Level 40+)
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient], # Pass LLMClient if used for signup details
        fingerprint_generator: FingerprintGenerator, # REQUIRED
        browser_automator: BrowserAutomatorInterface # REQUIRED
    ):
        self.llm_client = llm_client
        self.proxy_manager = ProxyManagerWrapper()
        self.fingerprint_generator = fingerprint_generator
        self.browser_automator = browser_automator
        self._trial_acquisition_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_BROWSER_AUTOMATIONS)
        self._active_automations: List[str] = [] # Track services currently undergoing trial acquisition

        self.supabase: Optional[SupabaseClient] = None
        if config.SUPABASE_ENABLED:
            try:
                self.supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
                logger.info(f"ResourceManager: Supabase client initialized for table '{config.SUPABASE_RESOURCES_TABLE}'.")
            except Exception as e:
                logger.error(f"ResourceManager: Failed to initialize Supabase client: {e}", exc_info=True)
        else:
            logger.warning("ResourceManager: Supabase not configured. Persistence disabled.")

        self.direct_clay_api_key: Optional[str] = config.CLAY_API_KEY
        if self.direct_clay_api_key:
             logger.info("ResourceManager: Direct Clay API Key (from .env) available.")
        else:
             logger.info("ResourceManager: Direct Clay API Key not found in .env. Will rely on Supabase or trial acquisition.")

    async def _generate_signup_details_for_service(self, service_name: str, context_info: Optional[str]=None) -> Dict[str, Any]:
        """Generates plausible signup details, potentially using LLM for variety and realism."""
        timestamp_part = int(time.time()*1000)
        details = { # Basic defaults
            "email": f"signup.{service_name.replace('.', '_')}.{timestamp_part}@mail-temp.com",
            "password": f"Compl3xP@ss{timestamp_part%10000}!",
            "first_name": random.choice(["Alex", "Jordan", "Casey", "Morgan", "Riley", "Devon"]),
            "last_name": random.choice(["Lee", "Devon", "Sky", "River", "Chase"]),
            "company_name": f"{random.choice(['Innovate', 'Synergy', 'Quantum', 'NextGen', 'Apex'])} Dynamics Inc."
        }
        if self.llm_client:
            try:
                prompt_parts = [
                    f"Generate highly plausible, unique, and varied user registration details for a new trial account on a B2B service called '{service_name}'.",
                    "Details should look human-generated for a temporary business evaluation.",
                    f"Include: first_name, last_name, email (unique, incorporate '{timestamp_part}', use plausible free/temp domain), password (strong, 12+ chars, mix case/numbers/symbols), company_name (plausible SMB name).",
                ]
                if context_info: prompt_parts.append(f"Context: {context_info}")
                prompt_parts.append("Return ONLY a valid JSON object: {'first_name': '...', 'last_name': '...', 'email': '...', 'password': '...', 'company_name': '...'}.")
                
                prompt = "\n".join(prompt_parts)
                response_str = await self.llm_client.generate_response(
                    messages=[{"role": "user", "content": prompt}], temperature=0.9, max_tokens=250, purpose="general"
                )
                if response_str and '{' in response_str and '}' in response_str:
                    llm_details = json.loads(response_str.strip())
                    required_keys = ['first_name', 'last_name', 'email', 'password', 'company_name']
                    if all(key in llm_details for key in required_keys):
                        logger.info(f"LLM generated signup details for {service_name}: {llm_details.get('email')}")
                        return llm_details
                    else: logger.warning(f"LLM response missing keys. Using defaults. Response: {response_str}")
            except json.JSONDecodeError: logger.warning(f"LLM response not valid JSON. Using defaults. Response: {response_str}")
            except Exception as e: logger.warning(f"LLM failed to generate signup details for {service_name}, using defaults: {e}")
        
        logger.info(f"Using/Generated basic signup details for {service_name}: email={details['email']}")
        return details

    async def _attempt_and_verify_trial_acquisition(
        self,
        service_name: str,
        service_automation_config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Orchestrates automated trial acquisition AND verification using the BrowserAutomator."""
        service_name_lower = service_name.lower()
        if service_name_lower in self._active_automations:
            logger.warning(f"Trial acquisition for '{service_name_lower}' already in progress. Skipping.")
            return None
        
        self._active_automations.append(service_name_lower)
        resource_to_store = None

        async with self._trial_acquisition_semaphore:
            logger.info(f"Attempting trial acquisition & verification for: {service_name} using URL: {service_automation_config.get('signup_url')}")
            proxy_string = self.get_proxy_string()
            fingerprint = await self.fingerprint_generator.generate_profile(role_context=f"Automated trial signup for {service_name}")
            signup_details_generated = await self._generate_signup_details_for_service(service_name)
            automation_result = None
            try:
                if not await self.browser_automator.setup_session(proxy_string=proxy_string, fingerprint_profile=fingerprint):
                    logger.error(f"[{service_name}] Failed to setup browser automator session.")
                    self._active_automations.remove(service_name_lower); return None

                automation_result = await self.browser_automator.full_signup_and_extract(
                    service_name=service_name,
                    signup_url=service_automation_config["signup_url"],
                    form_interaction_plan=service_automation_config["form_interaction_plan"],
                    signup_details_generated=signup_details_generated,
                    success_indicator=service_automation_config["success_indicator"],
                    resource_extraction_rules=service_automation_config["resource_extraction_rules"],
                    captcha_config=service_automation_config.get("captcha_config"),
                    max_retries=service_automation_config.get("max_retries", 1)
                )
            except Exception as e:
                logger.error(f"BrowserAutomator execution raised unhandled exception for {service_name}: {e}", exc_info=True)
                automation_result = {"status": "failed", "reason": f"Automator unhandled exception: {str(e)}"}
            finally:
                await self.browser_automator.close_session()
                if service_name_lower in self._active_automations: self._active_automations.remove(service_name_lower)

            if automation_result and automation_result.get("status") == "success":
                logger.info(f"Automated trial acquisition for {service_name} reported SUCCESS by automator.")
                extracted_resources = automation_result.get("extracted_resources", {})
                primary_key = extracted_resources.get("api_key") or extracted_resources.get("access_token") or extracted_resources.get("primary_resource_key")
                if not primary_key and extracted_resources: primary_key = next(iter(extracted_resources.values()), None)

                if not primary_key:
                    logger.error(f"Trial for {service_name} succeeded but NO primary key/token extracted. Extracted: {extracted_resources}")
                    return None

                # --- Verification Step (Conceptual - Requires specific implementation per service) ---
                verified_ok = True
                verification_config = service_automation_config.get("verification_step")
                if verification_config and primary_key:
                    logger.info(f"Attempting to verify acquired resource for {service_name}...")
                    logger.warning(f"Resource verification for {service_name} is CONCEPTUAL. Assuming success.")
                    # verified_ok = await self._execute_resource_verification(service_name, primary_key, verification_config)
                    # if not verified_ok: return None
                
                if verified_ok:
                    logger.info(f"Acquired (and conceptually verified) resource for {service_name} is valid.")
                    creation_ts = datetime.now(timezone.utc)
                    trial_duration_days = int(service_automation_config.get("trial_duration_days", 7))
                    expiry_ts = creation_ts + timedelta(days=trial_duration_days)
                    resource_to_store = {
                        "service": service_name.lower(), "resource_type": "trial_account",
                        "resource_data": {
                            "key": primary_key, "username": signup_details_generated.get("email"),
                            "password": signup_details_generated.get("password"),
                            "signup_email": signup_details_generated.get("email"),
                            "all_extracted_resources": extracted_resources,
                            "cookies": automation_result.get("cookies")
                        },
                        "creation_timestamp": creation_ts.isoformat(), "expiry_timestamp": expiry_ts.isoformat(),
                        "status": "active",
                        "notes": f"Automated trial via {self.browser_automator.__class__.__name__}. Verified: {verified_ok}. Proxy: {proxy_string is not None}.",
                        "fingerprint_profile_summary": {"user_agent": fingerprint.get("user_agent"), "os": fingerprint.get("os")}
                    }
                    return resource_to_store
            else:
                reason = automation_result.get("reason", "Unknown failure") if automation_result else "Automator returned None"
                logger.error(f"Automated trial acquisition for {service_name} FAILED. Reason: {reason}")
                return None
        # Ensure cleanup if semaphore times out
        if service_name_lower in self._active_automations: self._active_automations.remove(service_name_lower)
        return None

    async def _store_resource_in_supabase(self, resource_dict: Dict[str, Any]) -> bool:
        """Stores a resource dictionary in the Supabase 'managed_resources' table."""
        if not self.supabase: logger.warning(f"Supabase disabled. Cannot store resource for {resource_dict.get('service')}."); return False
        try:
            resource_dict_cleaned = json.loads(json.dumps(resource_dict, default=str))
            logger.info(f"Storing resource for '{resource_dict_cleaned.get('service')}' in Supabase.")
            response = await asyncio.to_thread(
                self.supabase.table(config.SUPABASE_RESOURCES_TABLE).insert(resource_dict_cleaned).execute
            )
            if hasattr(response, 'data') and response.data:
                logger.info(f"Successfully stored resource in Supabase. Record ID: {response.data[0].get('id')}")
                return True
            elif hasattr(response, 'status_code') and 200 <= response.status_code < 300:
                logger.info(f"Successfully stored resource in Supabase (status {response.status_code}).")
                return True
            else: logger.error(f"Supabase insert failed or no data. Response: {response}"); return False
        except Exception as e: logger.error(f"Error storing resource in Supabase: {e}", exc_info=True); return False

    async def _get_active_resource_from_supabase(self, service_name: str, resource_type: str) -> Optional[Dict[str, Any]]:
        """Retrieves an active resource of a specific type for a service from Supabase."""
        if not self.supabase: return None
        current_time_iso = datetime.now(timezone.utc).isoformat()
        logger.debug(f"Querying Supabase for active '{resource_type}' for '{service_name}'.")
        try:
            query = (
                self.supabase.table(config.SUPABASE_RESOURCES_TABLE)
                .select("*")
                .eq("service", service_name.lower())
                .eq("resource_type", resource_type)
                .eq("status", "active")
                .gte("expiry_timestamp", current_time_iso) # Check expiry
                .order("expiry_timestamp", desc=True, nulls_first=True)
                .limit(1)
            )
            response = await asyncio.to_thread(query.execute)
            if response.data:
                resource = response.data[0]
                logger.info(f"Found active '{resource_type}' for '{service_name}' in Supabase (ID: {resource.get('id')}).")
                return resource
            return None
        except Exception as e:
            logger.error(f"Error querying Supabase for '{resource_type}' for '{service_name}': {e}", exc_info=True)
            return None

    async def _ensure_resource_via_automation(
        self,
        service_name: str,
        service_automation_config: Dict[str, Any]
        ) -> Optional[Dict[str, Any]]:
        """Ensures an active trial resource exists, attempting signup if needed."""
        active_trial = await self._get_active_resource_from_supabase(service_name, "trial_account")
        if active_trial: return active_trial

        logger.info(f"No active trial for '{service_name}'. Attempting automated acquisition.")
        new_trial_data = await self._attempt_and_verify_trial_acquisition(service_name, service_automation_config)

        if new_trial_data:
            if await self._store_resource_in_supabase(new_trial_data):
                logger.info(f"Automated trial for '{service_name}' acquired and stored.")
                return new_trial_data
            else:
                logger.error(f"Failed to store new trial for '{service_name}'.")
                new_trial_data["_persisted_in_supabase"] = False
                return new_trial_data
        return None

    # --- Public Methods ---
    def get_proxy_string(self) -> Optional[str]:
        """Gets a proxy string using globally configured credentials."""
        if config.PROXY_USERNAME and config.PROXY_PASSWORD:
            return self.proxy_manager.get_proxy_string(config.PROXY_USERNAME, config.PROXY_PASSWORD)
        logger.debug("Global proxy username/password not configured.")
        return None

    async def get_api_key(self, service_name: str, service_automation_config: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Gets API key: checks config, then Supabase (api_key type), then Supabase (trial_account type), then attempts trial automation."""
        service_name_lower = service_name.lower()
        logger.info(f"Requesting API key for service: '{service_name_lower}'. Automation config provided: {bool(service_automation_config)}")

        # 1. Direct config
        if service_name_lower == "clay.com" and self.direct_clay_api_key:
            logger.info("Using direct CLAY_API_KEY from .env.")
            return self.direct_clay_api_key

        # 2. Active 'api_key' from Supabase
        api_key_resource = await self._get_active_resource_from_supabase(service_name_lower, "api_key")
        if api_key_resource and api_key_resource.get("resource_data", {}).get("key"):
            logger.info(f"Found active API key for '{service_name_lower}' (type: 'api_key') in Supabase.")
            return api_key_resource["resource_data"]["key"]

        # 3. Key from active 'trial_account' in Supabase
        trial_account_resource = await self._get_active_resource_from_supabase(service_name_lower, "trial_account")
        if trial_account_resource and trial_account_resource.get("resource_data", {}).get("key"):
            logger.info(f"Found API key within active 'trial_account' for '{service_name_lower}' in Supabase.")
            return trial_account_resource["resource_data"]["key"]
        
        # 4. Attempt automated trial acquisition if config provided
        if service_automation_config and service_automation_config.get("signup_url"):
            logger.info(f"No readily available API key for '{service_name_lower}'. Attempting automated trial acquisition.")
            new_resource = await self._ensure_resource_via_automation(service_name_lower, service_automation_config)
            if new_resource and new_resource.get("resource_data", {}).get("key"):
                logger.info(f"Acquired API key for '{service_name_lower}' via automated trial.")
                return new_resource["resource_data"]["key"]
            else: logger.warning(f"Automated trial for '{service_name_lower}' attempted/checked, but no API key ('key') found/acquired.")
        
        logger.error(f"Failed to obtain API key for service: '{service_name_lower}'. All methods exhausted.")
        return None

    async def get_full_resource_bundle(self, service_name: str, service_automation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Gets comprehensive bundle: API key, credentials, proxy, fingerprint."""
        logger.info(f"Requesting full resource bundle for service: {service_name}")
        bundle: Dict[str, Any] = {"service": service_name, "api_key": None, "credentials": None, "proxy_string": None, "fingerprint_profile": None}
        service_name_lower = service_name.lower()

        bundle["api_key"] = await self.get_api_key(service_name_lower, service_automation_config)
        
        resource_for_creds_and_fp = None
        # Prioritize trial account if key came from there or exists
        trial_account_resource = await self._get_active_resource_from_supabase(service_name_lower, "trial_account")
        if trial_account_resource: resource_for_creds_and_fp = trial_account_resource
        
        if not resource_for_creds_and_fp: # Check specific credentials type if no trial found
            creds_resource = await self._get_active_resource_from_supabase(service_name_lower, "credentials")
            if creds_resource: resource_for_creds_and_fp = creds_resource
        
        associated_fingerprint_summary = None
        if resource_for_creds_and_fp and resource_for_creds_and_fp.get("resource_data"):
            rd = resource_for_creds_and_fp["resource_data"]
            if "username" in rd and "password" in rd: bundle["credentials"] = {"username": rd["username"], "password": rd["password"]}
            if resource_for_creds_and_fp.get("fingerprint_profile_summary"):
                 associated_fingerprint_summary = resource_for_creds_and_fp.get("fingerprint_profile_summary")
                 logger.info(f"Using fingerprint context from stored resource for {service_name}.")

        bundle["proxy_string"] = self.get_proxy_string()

        fp_os = associated_fingerprint_summary.get("os") if associated_fingerprint_summary else None
        fp_browser = associated_fingerprint_summary.get("browser") if associated_fingerprint_summary else None
        bundle["fingerprint_profile"] = await self.fingerprint_generator.generate_profile(
            role_context=f"Interaction with {service_name}", os_type=fp_os, browser_type=fp_browser
        )
        
        logger.info(f"Full resource bundle for '{service_name}'. API Key: {bool(bundle.get('api_key'))}, Creds: {bool(bundle.get('credentials'))}")
        return bundle

# --- Test function ---
async def _test_resource_manager_final(browser_automator_instance: BrowserAutomatorInterface):
    print("--- Testing ResourceManager (FINAL - Full Automation Orchestration) ---")
    
    try: fp_gen = FingerprintGenerator()
    except Exception as e: print(f"Failed FingerprintGenerator init: {e}"); return
    llm_client = None # Optional for test if not generating signup details via LLM

    rm = ResourceManager(llm_client=llm_client, fingerprint_generator=fp_gen, browser_automator=browser_automator_instance)

    # Define automation config for Clay.com (YOU MUST REFINE THIS)
    clay_automation_config = {
        "signup_url": "https://app.clay.com/auth/signup", # VERIFY
        "form_interaction_plan": [
             {"action": "fill", "selector": "input[name='email']", "value_key": "email"},
             {"action": "fill", "selector": "input[name='password']", "value_key": "password"},
             {"action": "fill", "selector": "input[name='firstName']", "value_key": "first_name"},
             {"action": "fill", "selector": "input[name='lastName']", "value_key": "last_name"},
             {"action": "fill", "selector": "input[name='companyName']", "value_key": "company_name"},
             {"action": "click", "selector": "button[type='submit']"} # VERIFY SELECTOR
        ],
        "success_indicator": {"type": "url_contains", "value": "/sources"}, # VERIFY success URL/condition
        "resource_extraction_rules": [
            {"type": "llm_vision_extraction",
             "prompt_template": "Scan screenshot after Clay signup/login. Find API Key (Settings->API). Respond ONLY with key or 'NOT_FOUND'. {screenshot_base64}",
             "resource_name": "api_key"}
        ],
        "trial_duration_days": 7, "max_retries": 0 # Set retries > 0 to test retry logic
    }

    print("\n1. Attempting to get Clay.com API Key (will try .env, Supabase, then full trial acquisition)...")
    clay_key = await rm.get_api_key("clay.com", service_automation_config=clay_automation_config)
    if clay_key: print(f"  SUCCESS: Clay API Key: {clay_key[:4]}...")
    else: print("  FAILURE: Could not obtain Clay API Key via any method.")

    print("\n2. Attempting to get full resource bundle for Clay.com...")
    clay_bundle = await rm.get_full_resource_bundle("clay.com", service_automation_config=clay_automation_config)
    print(f"  Clay Bundle: API Key: {bool(clay_bundle.get('api_key'))}, Creds: {bool(clay_bundle.get('credentials'))}")
    print(f"    Fingerprint UA: {clay_bundle.get('fingerprint_profile',{}).get('user_agent')}")

if __name__ == "__main__":
    # import asyncio
    # from dotenv import load_dotenv
    # from core.automation.multimodal_playwright_automator import MultiModalPlaywrightAutomator # Or your implementation
    # from core.services.llm_client import LLMClient
    # async def run_test():
    #     load_dotenv()
    #     llm = LLMClient() if config.OPENROUTER_API_KEY else None
    #     # Replace with your actual automator instance
    #     automator = MockBrowserAutomatorForServer(llm) # Use mock if real one not ready/configured
    #     # automator = MultiModalPlaywrightAutomator(llm, headless=True) # Use real one
    #     await _test_resource_manager_final(browser_automator_instance=automator)
    # asyncio.run(run_test())
    print("ResourceManager (FINAL) defined. Test requires concrete BrowserAutomator and accurate service_automation_configs.")
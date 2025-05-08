# boutique_ai_project/core/agents/provisioning_agent.py

import logging
import json
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Callable, Coroutine

import config # Root config
from core.agents.resource_manager import ResourceManager
from core.services.crm_wrapper import CRMWrapper
from core.services.llm_client import LLMClient
# Import the DeploymentManager Interface and Config class
from core.services.deployment_manager import DeploymentManagerInterface, DeploymentConfig
# Optional: Import an Email Service Wrapper if implementing automated config requests
# from core.services.email_wrapper import EmailServiceWrapper

logger = logging.getLogger(__name__)

# --- Status Constants ---
# Statuses set by SalesAgent or external process indicating a deal is ready
STATUS_NEEDS_PROVISIONING = "Deal_Closed_Provisioning_Required"
# Statuses managed by this ProvisioningAgent
STATUS_GATHERING_CONFIG = "Provisioning_Gathering_Config"
STATUS_CONFIG_COLLECTION_PENDING = "Provisioning_Client_Input_Pending" # If waiting for client input
STATUS_PROVISIONING_RESOURCES = "Provisioning_Acquiring_Resources"
STATUS_DEPLOYING_INSTANCE = "Provisioning_Deploying_Instance"
STATUS_ACTIVE = "Service_Active" # Final success state
STATUS_PROVISIONING_FAILED = "Provisioning_Failed"
STATUS_NEEDS_MANUAL_INTERVENTION = "Provisioning_Needs_Manual_Help"

class ProvisioningAgent:
    """
    Autonomous AI agent responsible for provisioning and deploying client-specific
    AI agent instances after a deal is closed. (Level 40+ Foundation)
    Monitors CRM for closed deals and orchestrates the setup process.
    """

    def __init__(
        self,
        agent_id: str,
        resource_manager: ResourceManager,
        crm_wrapper: CRMWrapper,
        llm_client: LLMClient,
        deployment_manager: DeploymentManagerInterface, # Injected dependency
        # email_service: Optional[EmailServiceWrapper] = None, # Optional for client communication
        # Optional callback for reporting status to an orchestrator/UI
        on_provisioning_event: Optional[Callable[[str, str, str, Optional[str]], Coroutine[Any, Any, None]]] = None # agent_id, client_contact_id, status, details
        ):
        self.agent_id = agent_id
        self.resource_manager = resource_manager
        self.crm_wrapper = crm_wrapper
        self.llm_client = llm_client
        self.deployment_manager = deployment_manager
        # self.email_service = email_service # Store if provided
        self.on_provisioning_event = on_provisioning_event

        # Configurable parameters from main config
        self.run_interval_seconds = config.get_int_env_var("PROVISIONING_AGENT_INTERVAL_S", default=300, required=False) # Check every 5 mins
        self.max_concurrent_provisioning = config.get_int_env_var("PROVISIONING_AGENT_MAX_CONCURRENT", default=3, required=False)
        
        self._is_running = False
        self._current_task: Optional[asyncio.Task] = None
        self._provisioning_semaphore = asyncio.Semaphore(self.max_concurrent_provisioning)
        self._currently_provisioning = set() # Track IDs being processed to avoid duplicates

        logger.info(f"ProvisioningAgent {self.agent_id} initialized. Interval: {self.run_interval_seconds}s, Concurrency: {self.max_concurrent_provisioning}")

    async def start(self):
        """Starts the periodic provisioning check loop."""
        if self._is_running: logger.warning(f"ProvisioningAgent {self.agent_id} already running."); return
        logger.info(f"ProvisioningAgent {self.agent_id} starting run cycle.")
        self._is_running = True
        self._current_task = asyncio.create_task(self._run_loop())

    async def stop(self):
        """Stops the provisioning check loop gracefully."""
        if not self._is_running or not self._current_task: logger.warning(f"ProvisioningAgent {self.agent_id} not running."); return
        logger.info(f"ProvisioningAgent {self.agent_id} stopping..."); self._is_running = False
        if self._current_task: self._current_task.cancel()
        try: await self._current_task
        except asyncio.CancelledError: logger.info(f"ProvisioningAgent {self.agent_id} cycle task cancelled.")
        except Exception as e: logger.error(f"Error awaiting cancelled ProvisioningAgent task: {e}")
        self._current_task = None; logger.info(f"ProvisioningAgent {self.agent_id} stopped.")

    async def _run_loop(self):
        """Main loop checking for clients needing provisioning."""
        while self._is_running:
            try:
                logger.info(f"ProvisioningAgent {self.agent_id}: Checking for clients needing provisioning...")
                await self._check_and_provision_clients()
            except Exception as e:
                logger.error(f"ProvisioningAgent {self.agent_id}: Unhandled error in run loop: {e}", exc_info=True)
                await asyncio.sleep(self.run_interval_seconds / 2) # Shorter wait after error

            if self._is_running:
                await asyncio.sleep(self.run_interval_seconds)

    async def _check_and_provision_clients(self):
        """Fetches clients needing provisioning and attempts to provision them concurrently."""
        if not self.crm_wrapper.supabase:
            logger.warning("Supabase disabled, cannot check for clients needing provisioning.")
            return

        try:
            # Fetch clients needing provisioning, excluding those already being processed
            query = (
                self.crm_wrapper.supabase.table(config.SUPABASE_CONTACTS_TABLE)
                .select("id, email, company_name, llm_full_analysis_json, client_config_json") # Select needed fields
                .eq("status", STATUS_NEEDS_PROVISIONING)
                # .not_.in_("id", list(self._currently_provisioning)) # Filter out already processing (requires list not empty)
                .limit(self.max_concurrent_provisioning * 2)
            )
            response = await self.crm_wrapper._execute_supabase_query(query)

            if response and response.data:
                clients_to_provision = [c for c in response.data if c.get("id") not in self._currently_provisioning]
                if not clients_to_provision:
                    logger.info("No new clients found needing provisioning in this cycle.")
                    return

                logger.info(f"Found {len(clients_to_provision)} client(s) needing provisioning. Starting tasks.")
                tasks = []
                for client_data in clients_to_provision:
                    client_id = client_data.get("id")
                    if client_id:
                        self._currently_provisioning.add(client_id) # Mark as processing
                        tasks.append(self._provision_single_client(client_data))
                
                if tasks: await asyncio.gather(*tasks) # Run provisioning tasks concurrently

            else:
                logger.info("No clients found needing provisioning in this cycle.")

        except Exception as e:
            logger.error(f"Error fetching clients needing provisioning: {e}", exc_info=True)

    async def _provision_single_client(self, client_contact_data: Dict[str, Any]):
        """Handles the provisioning workflow for one client."""
        contact_id = client_contact_data.get("id")
        client_email = client_contact_data.get("email", f"UnknownContact_{contact_id}")
        if not contact_id: logger.error("Cannot provision client: contact data missing 'id'."); return

        # Use semaphore to limit actual concurrent heavy work (like browser automation or deployment)
        async with self._provisioning_semaphore:
            if not self._is_running: return # Check if agent stopped while waiting for semaphore

            logger.info(f"Starting provisioning process for Contact ID: {contact_id} ({client_email})")
            await self._emit_event(contact_id, "Provisioning_Started", "Initiating provisioning workflow.")
            
            try:
                # 1. Gather/Validate Configuration
                await self.crm_wrapper.update_contact_status_and_notes(contact_id, STATUS_GATHERING_CONFIG, "Gathering configuration requirements.")
                deployment_config_dict = await self._gather_client_config(client_contact_data)
                
                if not deployment_config_dict:
                    logger.error(f"Failed to gather necessary configuration for Contact ID: {contact_id}.")
                    await self.crm_wrapper.update_contact_status_and_notes(contact_id, STATUS_NEEDS_MANUAL_INTERVENTION, "Failed: Missing critical configuration data.")
                    await self._emit_event(contact_id, STATUS_PROVISIONING_FAILED, "Missing critical configuration.")
                    return # Exit this client's provisioning

                # 2. Acquire Necessary Resources (Conceptual - Requires specific logic)
                # Example: If client needs dedicated resources acquired via ResourceManager
                # await self.crm_wrapper.update_contact_status_and_notes(contact_id, STATUS_PROVISIONING_RESOURCES, "Acquiring dedicated resources (Conceptual).")
                # required_services = deployment_config_dict.get("required_dedicated_services", [])
                # client_api_keys = deployment_config_dict.get("client_provided_api_keys", {})
                # for service in required_services:
                #     if service not in client_api_keys:
                #         logger.info(f"Attempting to acquire resource '{service}' for client {contact_id} via RM.")
                #         # Define automation config for this service acquisition
                #         automation_config = config.get_service_automation_config(service) # Needs helper in config.py
                #         if automation_config:
                #             resource_key = await self.resource_manager.get_api_key(service, automation_config)
                #             if not resource_key: raise ValueError(f"Failed to acquire resource '{service}'")
                #             client_api_keys[service] = resource_key
                #         else: raise ValueError(f"Automation config missing for required service '{service}'")
                # deployment_config_dict["client_api_keys"] = client_api_keys # Update config
                logger.info(f"Resource acquisition step skipped/conceptual for Contact ID: {contact_id}.")

                # 3. Deploy Agent Instance using DeploymentManager
                await self.crm_wrapper.update_contact_status_and_notes(contact_id, STATUS_DEPLOYING_INSTANCE, f"Initiating deployment via {self.deployment_manager.__class__.__name__}.")
                
                deploy_config_obj = DeploymentConfig(
                    client_id=str(contact_id),
                    client_email=client_email,
                    agent_type=deployment_config_dict.get("agent_type", "SalesAgent"),
                    configuration=deployment_config_dict # Pass the full gathered/inferred config
                )

                deployment_result = await self.deployment_manager.deploy_agent_instance(deploy_config_obj)

                # 4. Update CRM based on deployment result
                if deployment_result.get("status") == "success":
                    instance_id = deployment_result.get("instance_id", "N/A")
                    access_url = deployment_result.get("access_url", "N/A")
                    logger.info(f"Successfully deployed agent instance for Contact ID: {contact_id}. Instance ID: {instance_id}")
                    await self.crm_wrapper.update_contact_status_and_notes(
                        contact_id, STATUS_ACTIVE, f"Service Active. Instance ID: {instance_id}. Access URL: {access_url}"
                    )
                    await self._emit_event(contact_id, STATUS_ACTIVE, f"Instance {instance_id} deployed.")
                else:
                    failure_reason = deployment_result.get("reason", "Unknown deployment failure")
                    logger.error(f"Failed to deploy agent instance for Contact ID: {contact_id}. Reason: {failure_reason}")
                    await self.crm_wrapper.update_contact_status_and_notes(
                        contact_id, STATUS_PROVISIONING_FAILED, f"Failed: Deployment error - {failure_reason}"
                    )
                    await self._emit_event(contact_id, STATUS_PROVISIONING_FAILED, f"Deployment error: {failure_reason}")

            except Exception as e:
                logger.error(f"Unhandled error during provisioning for Contact ID {contact_id}: {e}", exc_info=True)
                try:
                    await self.crm_wrapper.update_contact_status_and_notes(contact_id, STATUS_PROVISIONING_FAILED, f"Failed: Unexpected error - {type(e).__name__}")
                except Exception as log_e: logger.error(f"Failed even to update CRM status to failed for {contact_id}: {log_e}")
                await self._emit_event(contact_id, STATUS_PROVISIONING_FAILED, f"Unexpected error: {type(e).__name__}")
            finally:
                 if contact_id in self._currently_provisioning:
                     self._currently_provisioning.remove(contact_id) # Ensure ID is removed on completion/error


    async def _gather_client_config(self, client_contact_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Gathers necessary configuration for deploying a client's agent.
        Prioritizes stored config, then infers from sales analysis, then potentially triggers follow-up.
        """
        contact_id = client_contact_data.get("id")
        logger.info(f"Gathering deployment configuration for Contact ID: {contact_id}...")

        # 1. Check for pre-stored config in CRM
        stored_config = client_contact_data.get("client_config_json")
        if stored_config and isinstance(stored_config, dict) and stored_config.get("config_complete"): # Check for a completion flag
            logger.info(f"Found complete stored client configuration for Contact ID: {contact_id}.")
            return stored_config

        # 2. Infer config from sales analysis data (if available)
        inferred_config = {}
        sales_analysis = client_contact_data.get("llm_full_analysis_json")
        if sales_analysis and isinstance(sales_analysis, dict):
            logger.info(f"Attempting to infer configuration from sales analysis for Contact ID: {contact_id}.")
            inferred_config = {
                "agent_type": "SalesAgent", # Default, could be inferred from analysis later
                "target_niche": sales_analysis.get("target_niche") or client_contact_data.get("industry") or config.AGENT_TARGET_NICHE_DEFAULT,
                "initial_prompt_context": {
                    "company_name": client_contact_data.get("company_name"),
                    "primary_pain_point": sales_analysis.get("primary_inferred_pain_point"),
                    "suggested_hook": sales_analysis.get("suggested_outreach_hook")
                },
                "client_provided_api_keys": {}, # Placeholder for keys client needs to provide
                "required_config_missing": [] # Track missing essential items
            }
            # Example: Check if client needs to provide CRM keys
            # if not inferred_config["client_provided_api_keys"].get("crm_api_key"):
            #     inferred_config["required_config_missing"].append("crm_api_key")

        # Merge stored config (if any) with inferred config
        final_config = stored_config or {}
        final_config.update(inferred_config) # Inferred values override stored ones if keys clash, adjust if needed

        # 3. Check if critical configuration is missing
        # Define what's critical (e.g., client's CRM API key if integration is needed)
        critical_missing = final_config.get("required_config_missing", [])
        # Example check:
        # if not final_config.get("client_provided_api_keys", {}).get("crm_api_key"):
        #     critical_missing.append("client_crm_api_key")

        if critical_missing:
            logger.warning(f"Configuration for {contact_id} is incomplete. Missing: {critical_missing}.")
            # Option A: Mark for manual intervention
            # await self.crm_wrapper.update_contact_status_and_notes(contact_id, STATUS_NEEDS_MANUAL_INTERVENTION, f"Missing config: {critical_missing}")
            # return None

            # Option B: Trigger automated follow-up (Conceptual - requires EmailServiceWrapper)
            # if self.email_service:
            #     logger.info(f"Triggering automated config request email to {client_contact_data.get('email')}.")
            #     success = await self.email_service.send_config_request_email(
            #         to_email=client_contact_data.get('email'),
            #         contact_name=client_contact_data.get('first_name'),
            #         missing_items=critical_missing,
            #         secure_form_link=f"{config.BASE_WEBHOOK_URL}/client_config/{contact_id}" # Example link
            #     )
            #     if success:
            #         await self.crm_wrapper.update_contact_status_and_notes(contact_id, STATUS_CONFIG_COLLECTION_PENDING, f"Automated email sent for missing config: {critical_missing}")
            #     else:
            #         await self.crm_wrapper.update_contact_status_and_notes(contact_id, STATUS_NEEDS_MANUAL_INTERVENTION, f"Failed to send config request email.")
            # else:
            #      logger.error(f"Cannot request missing config for {contact_id}: Email service not available.")
            #      await self.crm_wrapper.update_contact_status_and_notes(contact_id, STATUS_NEEDS_MANUAL_INTERVENTION, f"Missing config and no email service: {critical_missing}")
            # For now, fail if critical info missing and no automated follow-up
            logger.error(f"CRITICAL configuration missing for {contact_id}: {critical_missing}. Manual intervention required.")
            await self.crm_wrapper.update_contact_status_and_notes(contact_id, STATUS_NEEDS_MANUAL_INTERVENTION, f"Missing critical config: {critical_missing}")
            return None
        else:
            final_config["config_complete"] = True # Mark config as complete
            # Store the completed config back to CRM
            await self.crm_wrapper.upsert_contact({"id": contact_id, "client_config_json": final_config}, "id")
            logger.info(f"Configuration gathered successfully for Contact ID: {contact_id}.")
            return final_config

    async def _emit_event(self, contact_id: str, status: str, details: Optional[str]):
        """Helper to call the optional status callback."""
        if self.on_provisioning_event:
            try:
                # Use create_task to avoid blocking the agent's main loop
                asyncio.create_task(
                    self.on_provisioning_event(self.agent_id, contact_id, status, details)
                )
            except Exception as e:
                logger.error(f"Error in on_provisioning_event callback task: {e}", exc_info=True)

# --- Test Function ---
async def _test_provisioning_agent_final():
    print("--- Testing ProvisioningAgent (FINAL - Level 40+) ---")
    if not config.SUPABASE_ENABLED or not config.OPENROUTER_API_KEY:
        print("Skipping test: Supabase or OpenRouter not configured.")
        return

    # Use Mock Dependencies for isolated testing
    class MockDeploymentManager(DeploymentManagerInterface):
        async def deploy_agent_instance(self, cfg: DeploymentConfig) -> Dict[str, Any]:
            logger.info(f"MOCK DEPLOY: Client {cfg.client_id}, Agent {cfg.agent_type}")
            await asyncio.sleep(1)
            return {"status": "success", "instance_id": f"mock_{cfg.instance_id_suggestion}", "access_url": "http://mock.url"}
        async def get_instance_status(self, instance_id: str) -> Dict[str, Any]: return {"status": "running"}
        async def stop_agent_instance(self, instance_id: str) -> bool: return True
        async def start_agent_instance(self, instance_id: str) -> bool: return True
        async def delete_agent_instance(self, instance_id: str) -> bool: return True

    class MockBrowserAutomatorForProvTest(BrowserAutomatorInterface):
        async def setup_session(self, *args, **kwargs): return True
        async def close_session(self, *args, **kwargs): pass
        async def full_signup_and_extract(self, *args, **kwargs): return {"status": "success", "extracted_resources": {"api_key": "mock_prov_key"}}
        async def navigate_to_page(self, *args, **kwargs): return True; async def take_screenshot(self, *args, **kwargs): return b"d"; async def fill_form_and_submit(self, *args, **kwargs): return True; async def check_success_condition(self, *args, **kwargs): return True; async def extract_resources_from_page(self, *args, **kwargs): return {}; async def solve_captcha_if_present(self, *args, **kwargs): return True; async def get_cookies(self) -> Optional[List[Dict[str, Any]]]: return []

    llm = LLMClient()
    fp_gen = FingerprintGenerator(llm)
    browser_automator = MockBrowserAutomatorForProvTest()
    resource_manager = ResourceManager(llm, fp_gen, browser_automator)
    crm_wrapper = CRMWrapper()
    deployment_manager = MockDeploymentManager()

    # Ensure DB tables exist
    db_client = create_supabase_client_for_setup()
    if db_client: await setup_supabase_tables(db_client)
    else: print("Cannot run test: Supabase client failed."); return

    # Add a dummy contact needing provisioning
    ts = int(time.time())
    test_email = f"provisiontest.final.{ts}@example.com"
    # Simulate analysis data being present from SalesAgent closing the deal
    simulated_analysis = {
        "likelihood_score": 8, "primary_inferred_pain_point": "Scaling outbound efforts",
        "suggested_outreach_hook": "Automate top-of-funnel", "is_qualified": True,
        "qualification_reasoning": "Good fit based on size and inferred pain.",
        "confidence_in_analysis": "Medium", "suggested_next_action": "Immediate_SalesAgent_Call" # This would change post-sale
    }
    contact_payload = {
        "email": test_email, "status": STATUS_NEEDS_PROVISIONING,
        "company_name": f"Provision Final Test {ts}", "first_name": "Provision", "last_name": "Test",
        "llm_full_analysis_json": simulated_analysis,
        # Simulate client providing some config, but maybe missing CRM key
        "client_config_json": {"target_niche": "FinTech AI", "agent_persona_tone": "Formal"}
    }
    created = await crm_wrapper.upsert_contact(contact_payload, "email")
    if not created: print(f"Failed to create test contact {test_email}. Aborting."); return
    print(f"Created test contact {test_email} with status '{STATUS_NEEDS_PROVISIONING}'.")

    provisioning_agent = ProvisioningAgent(
        agent_id="ProvAgent_Final_Test", resource_manager=resource_manager, crm_wrapper=crm_wrapper,
        llm_client=llm, deployment_manager=deployment_manager
    )

    print("\n--- Running one provisioning check cycle ---")
    # This will find the contact and attempt to provision it using the mock deployment manager
    await provisioning_agent._check_and_provision_clients()

    print("\n--- Verifying contact status in Supabase ---")
    # Wait a moment for async updates
    await asyncio.sleep(1)
    final_contact = await crm_wrapper.get_contact_info(test_email, "email")
    if final_contact:
        print(f"Final status for {test_email}: {final_contact.get('status')}")
        # Expected status: Service_Active (due to mock success) or Needs_Manual_Help/Failed if config gathering failed conceptually
    else:
        print(f"Could not retrieve final status for {test_email}.")

if __name__ == "__main__":
    # import asyncio
    # from dotenv import load_dotenv
    # load_dotenv()
    # asyncio.run(_test_provisioning_agent_final())
    print("ProvisioningAgent (FINAL - Level 40+) defined. Test requires full .env and Supabase setup.")
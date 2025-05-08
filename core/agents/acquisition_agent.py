# boutique_ai_project/core/agents/acquisition_agent.py

import logging
import json
import asyncio
import random
import time
import uuid # For generating unique correlation IDs
import os # For file path operations
import csv # For reading CSV lead source
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable, Coroutine, Tuple

import config # Root config
from core.agents.resource_manager import ResourceManager
from core.services.data_wrapper import DataWrapper, ClayWebhookError
from core.services.llm_client import LLMClient
from core.services.crm_wrapper import CRMWrapper
# Import the interface, not the concrete implementation directly
from core.automation.browser_automator_interface import BrowserAutomatorInterface, ResourceExtractionRule

logger = logging.getLogger(__name__)

# Example: Define a default structure for how Clay automation config might be passed if needed by RM
DEFAULT_CLAY_SERVICE_AUTOMATION_CONFIG = {
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
    "trial_duration_days": 7, "max_retries": 1, "captcha_config": None
}


class AcquisitionAgent:
    """
    Autonomous AI agent for "Level 40+" lead acquisition and qualification.
    Sources leads, orchestrates enrichment via Clay.com webhooks (attempts autonomous
    webhook discovery if needed), handles asynchronous results, performs deep
    LLM-based analysis, and meticulously tracks lead progression in CRM.
    """

    def __init__(
        self,
        agent_id: str,
        resource_manager: ResourceManager, # Requires a BrowserAutomator instance injected into RM
        data_wrapper: DataWrapper,
        llm_client: LLMClient,
        crm_wrapper: CRMWrapper,
        target_criteria: Optional[Dict[str, Any]] = None,
        on_cycle_complete_callback: Optional[Callable[[str, int, int, int], Coroutine[Any, Any, None]]] = None, # agent_id, sourced, sent_to_clay, qualified
        on_cycle_error_callback: Optional[Callable[[str, str], Coroutine[Any, Any, None]]] = None
        ):
        self.agent_id = agent_id
        self.resource_manager = resource_manager
        self.data_wrapper = data_wrapper
        self.llm_client = llm_client
        self.crm_wrapper = crm_wrapper
        
        self.target_criteria = target_criteria if target_criteria else self._get_default_target_criteria()
        self.run_interval_seconds = int(self.target_criteria.get("run_interval_seconds", config.ACQUISITION_AGENT_RUN_INTERVAL_SECONDS))
        self.batch_size = int(self.target_criteria.get("batch_size", config.ACQUISITION_AGENT_BATCH_SIZE))
        
        # Clay webhook URL: Try config first, then attempt discovery if needed
        self.clay_enrichment_webhook_url: Optional[str] = self.target_criteria.get("clay_enrichment_table_webhook_url")
        self.clay_enrichment_table_name: Optional[str] = self.target_criteria.get("clay_enrichment_table_name", "Primary Lead Enrichment") # Name to search for in Clay UI
        self._clay_webhook_url_verified_or_discovered = bool(self.clay_enrichment_webhook_url) # Track if we have a URL

        self.on_cycle_complete_callback = on_cycle_complete_callback
        self.on_cycle_error_callback = on_cycle_error_callback

        self._is_running = False
        self._current_task: Optional[asyncio.Task] = None
        self._processing_lock = asyncio.Lock() # To prevent race conditions

        logger.info(f"AcquisitionAgent {self.agent_id} (Level 40+) initialized. Target Niche: {self.target_criteria.get('niche', 'N/A')}.")
        if not self.clay_enrichment_webhook_url:
            logger.warning(f"Clay enrichment webhook URL not provided. Will attempt autonomous discovery for table named '{self.clay_enrichment_table_name}'.")
        else:
             logger.info(f"Using pre-configured Clay Webhook: {self.clay_enrichment_webhook_url[:50]}...")

    def _get_default_target_criteria(self) -> Dict[str, Any]:
        # Load from config/env vars for better management
        return {
            "niche": config.AGENT_TARGET_NICHE_DEFAULT,
            "clay_enrichment_table_webhook_url": config.CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY, # From .env
            "clay_enrichment_table_name": config.get_env_var("CLAY_ENRICHMENT_TABLE_NAME", default="Primary Lead Enrichment", required=False), # Name to find if URL missing
            "lead_source_type": config.ACQ_LEAD_SOURCE_TYPE,
            "lead_source_path": config.ACQ_LEAD_SOURCE_PATH,
            "lead_source_csv_field_mapping": config.ACQ_LEAD_SOURCE_CSV_MAPPING,
            "supabase_pending_lead_status": config.ACQ_SUPABASE_PENDING_LEAD_STATUS,
            "qualification_llm_score_threshold": config.ACQ_QUALIFICATION_THRESHOLD,
            "max_leads_to_process_per_cycle": config.ACQUISITION_AGENT_BATCH_SIZE,
            "run_interval_seconds": config.ACQUISITION_AGENT_RUN_INTERVAL_SECONDS,
            "batch_size": config.ACQUISITION_AGENT_BATCH_SIZE,
            "clay_service_automation_config": DEFAULT_CLAY_SERVICE_AUTOMATION_CONFIG # For RM if it needs to get Clay key
        }

    async def _autonomously_find_clay_webhook_url(self) -> Optional[str]:
        """Attempts to use BrowserAutomator to find a Clay table's input webhook URL."""
        logger.warning(f"Attempting autonomous discovery of Clay webhook URL for table named '{self.clay_enrichment_table_name}'. Requires robust BrowserAutomator.")
        
        clay_bundle = await self.resource_manager.get_full_resource_bundle(
            "clay.com", self.target_criteria.get("clay_service_automation_config")
        )
        if not clay_bundle or not clay_bundle.get("credentials"):
            logger.error("Webhook discovery failed: Could not obtain Clay login credentials via ResourceManager.")
            return None
            
        credentials = clay_bundle["credentials"]
        proxy = clay_bundle.get("proxy_string")
        fingerprint = await self.resource_manager.fingerprint_generator.generate_profile(role_context="Clay UI Login and Webhook Discovery")

        # Define the complex interaction plan for the automator
        table_name_to_find = self.clay_enrichment_table_name
        interaction_plan = [ # This plan is highly conceptual and needs real selectors/prompts
            {"action": "navigate", "url": "https://app.clay.com/auth/login"}, # VERIFY URL
            {"action": "wait", "duration": 2},
            {"action": "fill", "selector": "input[name='email']", "value_key": "username"},
            {"action": "fill", "selector": "input[name='password']", "value_key": "password"},
            {"action": "click", "selector": "button[type='submit']"}, # VERIFY SELECTOR
            {"action": "wait", "duration": 5, "comment": "Wait for dashboard/tables list"},
            {"action": "screenshot", "name": "tables_list_page"},
            {"action": "llm_find_and_click",
             "prompt": f"Find table named '{table_name_to_find}' in screenshot. Provide selector for its link/button.",
             "screenshot_ref": "tables_list_page"},
            {"action": "wait", "duration": 3},
            {"action": "screenshot", "name": "table_view_page"},
            {"action": "llm_find_and_click",
             "prompt": "Find '+ Add Source' or 'Import' button/link. Provide selector.",
             "screenshot_ref": "table_view_page"},
            {"action": "wait", "duration": 2},
            {"action": "screenshot", "name": "add_source_modal"},
            {"action": "llm_find_and_click",
             "prompt": "Find 'From webhook' or 'Monitor webhook' option. Provide selector.",
             "screenshot_ref": "add_source_modal"},
            {"action": "wait", "duration": 2},
            {"action": "screenshot", "name": "webhook_settings_page"}
        ]
        
        webhook_extraction_rule: ResourceExtractionRule = {
            "type": "llm_vision_extraction",
            "prompt_template": "Screenshot shows Clay's 'Import from Webhook' settings for table '{table_name}'. Extract the full Webhook URL (starts https://hooks.clay.com...). Respond ONLY with URL or 'NOT_FOUND'. {screenshot_base64}",
            "resource_name": "webhook_url"
        }

        automator = self.resource_manager.browser_automator
        webhook_url = None
        try:
            if not await automator.setup_session(proxy_string=proxy, fingerprint_profile=fingerprint): return None

            # Execute the plan (requires automator to implement 'llm_find_and_click' etc.)
            # This is the most complex part depending on the automator implementation.
            logger.warning("Executing conceptual interaction plan for Clay UI navigation via BrowserAutomator.")
            # interaction_success = await automator.execute_interaction_plan(interaction_plan, credentials) # Assumes method exists
            # For now, simulate failure of complex navigation
            interaction_success = False
            logger.error("Autonomous navigation within Clay UI to find webhook URL is NOT IMPLEMENTED/RELIABLE in this example automator.")

            if interaction_success:
                logger.info("Attempting to extract webhook URL using LLM vision...")
                webhook_page_screenshot = await automator.take_screenshot(full_page=True)
                if webhook_page_screenshot:
                    webhook_extraction_rule["prompt_template"] = webhook_extraction_rule["prompt_template"].format(
                        table_name=table_name_to_find, screenshot_base64="{screenshot_base64}" # Placeholder for actual injection if needed
                    )
                    extracted = await automator.extract_resources_from_page(
                        rules=[webhook_extraction_rule], page_screenshot_for_llm=webhook_page_screenshot
                    )
                    webhook_url = extracted.get("webhook_url")
                    if webhook_url and webhook_url != 'NOT_FOUND' and webhook_url.startswith("https://hooks.clay.com"):
                        logger.info(f"Successfully extracted Clay webhook URL autonomously: {webhook_url[:50]}...")
                        self.clay_enrichment_webhook_url = webhook_url
                        self._clay_webhook_url_verified_or_discovered = True
                        return webhook_url
                    else: logger.error(f"LLM vision failed to extract valid webhook URL. Extracted: {webhook_url}")
                else: logger.error("Failed to take screenshot of webhook settings page.")
            else: logger.error("Navigation to Clay webhook settings page failed.")
        except Exception as e: logger.error(f"Error during autonomous Clay webhook discovery: {e}", exc_info=True)
        finally: await automator.close_session()

        logger.error("Autonomous discovery of Clay webhook URL failed.")
        return None

    async def start(self):
        """Starts the agent's main processing loop."""
        if self._is_running: logger.warning(f"AcquisitionAgent {self.agent_id} already running."); return
        logger.info(f"AcquisitionAgent {self.agent_id} starting run cycle (Interval: {self.run_interval_seconds}s).")
        self._is_running = True
        self._current_task = asyncio.create_task(self._run_loop())

    async def stop(self):
        """Stops the agent's processing loop gracefully."""
        if not self._is_running or not self._current_task: logger.warning(f"AcquisitionAgent {self.agent_id} not running."); return
        logger.info(f"AcquisitionAgent {self.agent_id} stopping..."); self._is_running = False
        if self._current_task: self._current_task.cancel()
        try: await self._current_task
        except asyncio.CancelledError: logger.info(f"AcquisitionAgent {self.agent_id} cycle task cancelled.")
        except Exception as e: logger.error(f"Error awaiting cancelled task: {e}")
        self._current_task = None; logger.info(f"AcquisitionAgent {self.agent_id} stopped.")

    async def _run_loop(self):
        """Main loop: periodically runs the acquisition cycle."""
        while self._is_running:
            start_ts = time.monotonic()
            sourced, sent, qualified = 0, 0, 0
            try:
                logger.info(f"AcquisitionAgent {self.agent_id}: Starting cycle.")
                # --- Ensure Webhook URL ---
                if not self._clay_webhook_url_verified_or_discovered:
                    logger.info("Clay webhook URL not verified. Attempting discovery...")
                    await self._autonomously_find_clay_webhook_url() # This will set the URL if successful
                    if not self._clay_webhook_url_verified_or_discovered:
                        logger.error("Failed discovery. Skipping enrichment triggering this cycle.")
                        await asyncio.sleep(self.run_interval_seconds); continue
                
                # --- Run Cycle ---
                sourced, sent, qualified = await self._run_acquisition_cycle()
                if self.on_cycle_complete_callback:
                    await self.on_cycle_complete_callback(self.agent_id, sourced, sent, qualified)
            except Exception as e:
                logger.error(f"AcquisitionAgent {self.agent_id}: Unhandled error in cycle: {e}", exc_info=True)
                if self.on_cycle_error_callback: await self.on_cycle_error_callback(self.agent_id, str(e))
            
            elapsed = time.monotonic() - start_ts
            wait_time = max(10, self.run_interval_seconds - elapsed)
            if self._is_running:
                logger.info(f"AcquisitionAgent {self.agent_id}: Cycle took {elapsed:.2f}s. Sourced={sourced}, SentToClay={sent}, Qualified={qualified}. Waiting {wait_time:.2f}s.")
                await asyncio.sleep(wait_time)

    async def _source_initial_leads(self) -> List[Dict[str, Any]]:
        """Sources initial raw leads based on configured type."""
        source_type = self.target_criteria.get("lead_source_type", "supabase_query")
        max_leads = self.target_criteria.get("max_leads_to_process_per_cycle", self.batch_size)
        logger.info(f"Sourcing max {max_leads} initial leads via: {source_type}")
        if source_type == "local_csv": return await self._source_initial_leads_from_csv()
        if source_type == "supabase_query": return await self._source_leads_from_supabase()
        logger.warning(f"Unsupported lead source type '{source_type}'. No leads sourced.")
        return []

    async def _source_initial_leads_from_csv(self) -> List[Dict[str, Any]]:
        """Reads leads from a local CSV file."""
        leads = []
        path = self.target_criteria.get("lead_source_path", "data/initial_leads.csv")
        mapping = self.target_criteria.get("lead_source_csv_field_mapping", {"company_name": "company_name", "domain": "domain", "email": "primary_contact_email_guess"})
        data_dir = os.path.dirname(path)
        if data_dir and not os.path.exists(data_dir): os.makedirs(data_dir, exist_ok=True)
        if not os.path.exists(path):
            logger.warning(f"Lead source CSV not found at {path}. Creating dummy.")
            try:
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f); writer.writerow(mapping.keys())
                    writer.writerow(["Dummy CSV Co", "dummycsv.com", f"dummy.{int(time.time())}@examplecsv.com"])
            except OSError as e: logger.error(f"Failed to create dummy CSV at {path}: {e}"); return []

        try:
            with open(path, 'r', newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                max_leads = self.target_criteria.get("max_leads_to_process_per_cycle", self.batch_size)
                for i, row in enumerate(reader):
                    if i >= max_leads: break
                    lead_item = {target: row.get(source) for target, source in mapping.items()}
                    # Use email if present, otherwise generate placeholder based on domain/uuid
                    lead_item["primary_contact_email_guess"] = lead_item.get("email") or \
                                f"placeholder_{lead_item.get('domain', str(uuid.uuid4()))}@lead.placeholder"
                    if lead_item.get("domain") or lead_item.get("email"): # Need at least one identifier
                        leads.append(lead_item)
            logger.info(f"Loaded {len(leads)} leads from CSV: {path}")
        except FileNotFoundError: logger.error(f"CSV file not found at {path}.")
        except Exception as e: logger.error(f"Failed to load leads from CSV '{path}': {e}", exc_info=True)
        return leads

    async def _source_leads_from_supabase(self) -> List[Dict[str, Any]]:
        """Fetches leads from Supabase based on status."""
        if not self.crm_wrapper.supabase: logger.error("Supabase disabled, cannot source leads."); return []
        status = self.target_criteria.get("supabase_pending_lead_status", "New_Raw_Lead")
        limit = self.target_criteria.get("max_leads_to_process_per_cycle", self.batch_size)
        logger.info(f"Querying Supabase for up to {limit} contacts with status '{status}'.")
        query = self.crm_wrapper.supabase.table(config.SUPABASE_CONTACTS_TABLE).select("*").eq("status", status).limit(limit)
        response = await self.crm_wrapper._execute_supabase_query(query)
        if response and response.data:
            logger.info(f"Fetched {len(response.data)} leads from Supabase with status '{status}'.")
            return response.data
        logger.info(f"No leads found in Supabase with status '{status}'.")
        return []

    async def _trigger_clay_enrichment_for_leads(self, leads_to_enrich: List[Dict[str, Any]]) -> int:
        """Sends leads to Clay webhook, updates CRM status."""
        if not self.clay_enrichment_webhook_url: logger.error("Cannot enrich: Clay webhook URL unknown."); return 0
        sent_count = 0
        tasks = []
        logger.info(f"Preparing to send {len(leads_to_enrich)} leads to Clay webhook...")

        async def send_and_update(lead_data):
            nonlocal sent_count
            correlation_id = str(uuid.uuid4())
            clay_payload = { # Map fields your Clay table expects
                "domain": lead_data.get("domain"), "company_name": lead_data.get("company_name"),
                "email": lead_data.get("email"), "_correlation_id": correlation_id,
                "_source_agent_id": self.agent_id, "_timestamp_sent_utc": datetime.now(timezone.utc).isoformat()
            }
            clay_payload_cleaned = {k: v for k, v in clay_payload.items() if v is not None}
            if not clay_payload_cleaned.get("domain") and not clay_payload_cleaned.get("email"): return

            try:
                send_success = await self.data_wrapper.send_data_to_clay_webhook(self.clay_enrichment_webhook_url, clay_payload_cleaned)
                if send_success:
                    crm_lead_data = {"status": "Enrichment_Triggered_Clay", "last_activity_timestamp": datetime.now(timezone.utc).isoformat(), "correlation_id_clay": correlation_id}
                    unique_key = "id" if "id" in lead_data else "email"
                    key_value = lead_data.get("id") or lead_data.get("email")
                    if key_value:
                         crm_lead_data[unique_key] = key_value
                         update_success = await self.crm_wrapper.upsert_contact(crm_lead_data, unique_key_column=unique_key)
                         if update_success:
                             logger.info(f"Lead {key_value} sent to Clay. Correlation ID: {correlation_id}")
                             return True # Indicate success for counting
                         else:
                             logger.error(f"Sent lead {key_value} to Clay but failed to update CRM status.")
                    else: logger.warning(f"Lead missing ID/Email, cannot update status after sending. Data: {lead_data}")
            except Exception as e: logger.error(f"Error sending lead {lead_data.get('domain')} for Clay enrichment: {e}", exc_info=True)
            return False # Indicate failure

        for lead in leads_to_enrich:
            tasks.append(send_and_update(lead))
        
        results = await asyncio.gather(*tasks)
        sent_count = sum(1 for r in results if r is True)
        logger.info(f"Attempted to send {len(leads_to_enrich)} leads to Clay, {sent_count} succeeded.")
        return sent_count

    async def handle_clay_enrichment_result(self, clay_result_payload: Dict[str, Any]) -> bool:
        """Processes enriched data received from Clay, updates CRM, triggers analysis."""
        async with self._processing_lock:
            correlation_id = clay_result_payload.get("_correlation_id")
            if not correlation_id: logger.error(f"Clay result missing _correlation_id: {clay_result_payload}"); return False
            logger.info(f"Processing Clay enrichment result for correlation_id: {correlation_id}")
            
            contact_to_update = await self.crm_wrapper.get_contact_info(identifier=correlation_id, identifier_column="correlation_id_clay")
            if not contact_to_update: logger.error(f"No contact found for correlation_id_clay: {correlation_id}."); return False
            
            contact_id = contact_to_update.get("id")
            logger.info(f"Found contact ID {contact_id}. Updating with enrichment data.")

            update_payload = { # Map Clay output fields to CRM fields
                "id": contact_id, "status": "Enriched_From_Clay_Pending_Analysis",
                "last_activity_timestamp": datetime.now(timezone.utc).isoformat(),
                "clay_enrichment_data_json": clay_result_payload,
                "job_title": clay_result_payload.get("Title") or contact_to_update.get("job_title"),
                "linkedin_url": clay_result_payload.get("LinkedIn Personal URL") or contact_to_update.get("linkedin_url"),
                "company_linkedin_url": clay_result_payload.get("LinkedIn Company URL") or contact_to_update.get("company_linkedin_url"),
                "company_size_range": clay_result_payload.get("Company Size") or contact_to_update.get("company_size_range"),
                "industry": clay_result_payload.get("Industry") or contact_to_update.get("industry"),
                "company_description": clay_result_payload.get("Company Description") or contact_to_update.get("company_description"),
                "phone_number": clay_result_payload.get("Phone Number") or contact_to_update.get("phone_number"),
                "email": contact_to_update.get("email") # Preserve original email
            }
            # Clean payload of None values if necessary for upsert
            update_payload_cleaned = {k: v for k, v in update_payload.items() if v is not None}

            updated_contact = await self.crm_wrapper.upsert_contact(update_payload_cleaned, unique_key_column="id")
            if updated_contact:
                logger.info(f"Contact ID {contact_id} updated. Triggering analysis.")
                # Trigger analysis immediately for this lead
                asyncio.create_task(self._analyze_and_qualify_single_lead(updated_contact))
                return True
            else:
                logger.error(f"Failed to update contact ID {contact_id} after Clay enrichment.")
                return False

    async def _analyze_and_qualify_single_lead(self, lead_data: Dict[str, Any]):
         """Analyzes and qualifies a single lead, updating its CRM record."""
         async with self._processing_lock:
            lead_id_for_logs = lead_data.get("id") or lead_data.get("email", "Unknown Lead")
            logger.info(f"Analyzing single lead: {lead_id_for_logs}")
            # Reuse the batch analysis logic for a single lead
            qualified_count = await self._analyze_and_qualify_batch([lead_data])
            logger.info(f"Analysis complete for single lead {lead_id_for_logs}. Qualified: {qualified_count > 0}")


    async def _analyze_and_qualify_batch(self) -> int:
        """Fetches and qualifies a batch of leads pending analysis."""
        if not self.crm_wrapper.supabase: return 0
        logger.info("Querying for enriched leads pending analysis...")
        query = self.crm_wrapper.supabase.table(config.SUPABASE_CONTACTS_TABLE).select("*").eq("status", "Enriched_From_Clay_Pending_Analysis").limit(self.batch_size)
        response = await self.crm_wrapper._execute_supabase_query(query)
        if not response or not response.data: logger.info("No enriched leads found pending analysis."); return 0

        leads_to_analyze = response.data
        logger.info(f"Found {len(leads_to_analyze)} enriched leads for analysis.")
        qualified_in_batch = 0
        
        analysis_tasks = [self._analyze_and_update_single_lead_in_crm(lead) for lead in leads_to_analyze]
        results = await asyncio.gather(*analysis_tasks)
        
        qualified_in_batch = sum(1 for r in results if r is True)
        logger.info(f"Batch analysis complete. Newly qualified: {qualified_in_batch}/{len(leads_to_analyze)}")
        return qualified_in_batch

    async def _analyze_and_update_single_lead_in_crm(self, lead_data: Dict[str, Any]) -> bool:
        """Performs LLM analysis and updates CRM record for one lead."""
        lead_id_for_logs = lead_data.get("id") or lead_data.get("email", "Unknown Lead")
        logger.debug(f"Analyzing lead: {lead_id_for_logs}")
        clay_data = lead_data.get("clay_enrichment_data_json", {})
        context_for_llm = { # Selective context
            "company_name": lead_data.get("company_name"), "domain": lead_data.get("domain"),
            "industry": lead_data.get("industry"), "company_size_range": lead_data.get("company_size_range"),
            "job_title": lead_data.get("job_title"), "company_description": lead_data.get("company_description"),
            "primary_contact_name": f"{lead_data.get('first_name','')} {lead_data.get('last_name','')} ".strip(),
            "website_technologies": clay_data.get("BuiltWith - Tech Stack"),
            "funding_summary": clay_data.get("Crunchbase - Last Funding Round Amount & Date"),
        }
        context_for_llm_cleaned = {k:v for k,v in context_for_llm.items() if v is not None and v != ""}
        lead_context_str = json.dumps(context_for_llm_cleaned, indent=2, default=str)
        prompt = ( # Level 25+ Analysis Prompt
            f"You are an expert B2B sales development analyst for Boutique AI (sells AI Sales Agents for '{self.target_criteria.get('niche', 'B2B')}'). "
            f"Analyze lead data:\n```json\n{lead_context_str}\n```\n"
            f"Respond ONLY JSON: `likelihood_score` (int 1-10, needs/affords AI sales agents?), `primary_inferred_pain_point` (str, specific sales pain), `secondary_inferred_pain_point` (str, optional), `suggested_outreach_hook` (str, 1-2 compelling sentences), `is_qualified` (bool, score >= {self.target_criteria.get('qualification_llm_score_threshold', 6)}?), `qualification_reasoning` (str, brief reason for score/qual), `confidence_in_analysis` (Low/Medium/High), `suggested_next_action` (str: 'Immediate_SalesAgent_Call', 'Nurture_Sequence', 'Further_Research', 'Disqualify')."
        )
        messages = [{"role": "user", "content": prompt}]
        analysis_result_json_str = await self.llm_client.generate_response(messages=messages, purpose="analysis", temperature=0.1, max_tokens=800)

        if not analysis_result_json_str or "error" in analysis_result_json_str.lower():
            logger.error(f"LLM analysis failed for lead {lead_id_for_logs}. Response: {analysis_result_json_str}")
            await self.crm_wrapper.update_contact_status_and_notes(lead_data["id"], "Analysis_Failed_LLM", f"LLM Error: {analysis_result_json_str}")
            return False
        try:
            analysis = json.loads(analysis_result_json_str)
            is_qualified = analysis.get("is_qualified", False)
            logger.info(f"LLM Analysis for {lead_id_for_logs}: Score={analysis.get('likelihood_score')}, Qualified={is_qualified}")
            crm_update_payload = {
                "id": lead_data["id"],
                "status": f"Qualified_Sales_Ready" if is_qualified else f"Disqualified_Analysis",
                "llm_qualification_score": analysis.get("likelihood_score"), "llm_inferred_pain_point_1": analysis.get("primary_inferred_pain_point"),
                "llm_suggested_hook": analysis.get("suggested_outreach_hook"), "llm_qualification_reasoning": analysis.get("qualification_reasoning"),
                "llm_analysis_confidence": analysis.get("confidence_in_analysis"), "llm_suggested_next_action": analysis.get("suggested_next_action"),
                "llm_full_analysis_json": analysis, "last_activity_timestamp": datetime.now(timezone.utc).isoformat()
            }
            await self.crm_wrapper.upsert_contact(crm_update_payload, unique_key_column="id")
            return is_qualified
        except Exception as e:
            logger.error(f"Error processing LLM analysis for {lead_id_for_logs}: {e}", exc_info=True)
            await self.crm_wrapper.update_contact_status_and_notes(lead_data["id"], f"Analysis_Failed_Processing", str(e)[:200])
            return False

    async def _run_acquisition_cycle(self) -> Tuple[int, int, int]:
        """Main cycle: source, trigger enrichment, analyze enriched."""
        sourced_count, sent_to_clay_count, qualified_count = 0, 0, 0

        # --- Phase 1: Source & Upsert Raw Leads ---
        raw_leads = await self._source_initial_leads()
        leads_to_send_to_clay = []
        if raw_leads:
            logger.info(f"Sourced {len(raw_leads)} new raw leads.")
            for raw_lead_data in raw_leads:
                 unique_email = raw_lead_data.get("email") or raw_lead_data.get("primary_contact_email_guess") or \
                                f"placeholder_{raw_lead_data.get('domain', str(uuid.uuid4()))}@lead.placeholder"
                 crm_payload = {
                     "email": unique_email, "company_name": raw_lead_data.get("company_name"),
                     "domain": raw_lead_data.get("domain"), "status": "New_Raw_Lead",
                     "source_info": f"AcqAgent_{self.agent_id}_{self.target_criteria.get('lead_source_type', 'unknown')}",
                     "raw_lead_data_json": raw_lead_data, "last_activity_timestamp": datetime.now(timezone.utc).isoformat()
                 }
                 upserted_lead = await self.crm_wrapper.upsert_contact(crm_payload, unique_key_column="email")
                 if upserted_lead: leads_to_send_to_clay.append(upserted_lead)
                 else: logger.error(f"Failed to upsert raw lead: {raw_lead_data.get('domain')}")
            sourced_count = len(leads_to_send_to_clay)
        else: logger.info("No new raw leads sourced.")

        # --- Phase 2: Trigger Enrichment ---
        if leads_to_send_to_clay:
            sent_to_clay_count = await self._trigger_clay_enrichment_for_leads(leads_to_send_to_clay)
        
        # --- Phase 3: Analyze Previously Enriched Leads ---
        # This ensures leads enriched via webhook callback between cycles are processed.
        qualified_count = await self._analyze_and_qualify_batch()

        logger.info(f"Acquisition cycle summary: Sourced={sourced_count}, SentToClay={sent_to_clay_count}, NewlyQualified={qualified_count}")
        return sourced_count, sent_to_clay_count, qualified_count

# --- Test function ---
async def _test_acquisition_agent_final():
    # ... (Test function remains conceptually similar, ensuring it uses the updated agent and mocks/config) ...
    print("--- Testing AcquisitionAgent (FINAL - Level 40+) ---")
    # ... (rest of test setup as before) ...
    # Ensure TEST_CLAY_TABLE_WEBHOOK_URL_FROM_ENV is set in .env for the agent's criteria
    # ... (run cycle, simulate callback, run cycle again) ...

if __name__ == "__main__":
    print("AcquisitionAgent (FINAL - Level 40+) defined. Test requires full .env, Supabase, and Clay webhook setup.")
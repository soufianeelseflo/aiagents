# core/agents/acquisition_agent.py

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
# from core.automation.browser_automator_interface import BrowserAutomatorInterface # Not directly used by AcqAgent, RM uses it

logger = logging.getLogger(__name__)

# Example: Define a default structure for how Clay automation config might be passed if needed by RM
# This is more for RM's use if AcqAgent needed to tell RM to get a Clay key via automation.
# For now, AcqAgent primarily uses DataWrapper which uses webhooks.
DEFAULT_CLAY_SERVICE_AUTOMATION_CONFIG = {
    "signup_url": "https://app.clay.com/auth/signup", # VERIFY
    "form_interaction_plan": [
        {"action": "fill", "selector": "input[name='email']", "value_key": "email"},
        {"action": "fill", "selector": "input[name='password']", "value_key": "password"},
        {"action": "click", "selector": "button[type='submit']"}
    ],
    "success_indicator": {"type": "url_contains", "value": "/dashboard"},
    "resource_extraction_rules": [{"type": "llm_vision_extraction", "prompt_template": "Extract API key: {screenshot_base64}", "resource_name": "api_key"}],
    "trial_duration_days": 7
}


class AcquisitionAgent:
    """
    Autonomous AI agent for "Level 40" lead acquisition and qualification.
    Sources leads, orchestrates enrichment via Clay.com webhooks, handles asynchronous
    results, performs deep LLM-based analysis, and meticulously tracks lead progression.
    """

    def __init__(
        self,
        agent_id: str,
        resource_manager: ResourceManager,
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
        
        self.clay_enrichment_webhook_url = self.target_criteria.get("clay_enrichment_table_webhook_url")
        if not self.clay_enrichment_webhook_url:
            msg = f"AcquisitionAgent {self.agent_id}: CRITICAL - 'clay_enrichment_table_webhook_url' is REQUIRED in target_criteria for Clay enrichment."
            logger.critical(msg)
            raise ValueError(msg)

        self.on_cycle_complete_callback = on_cycle_complete_callback
        self.on_cycle_error_callback = on_cycle_error_callback

        self._is_running = False
        self._current_task: Optional[asyncio.Task] = None
        self._processing_lock = asyncio.Lock() # To prevent race conditions in handling Clay results / analysis

        logger.info(f"AcquisitionAgent {self.agent_id} (Level 40) initialized. Target Niche: {self.target_criteria.get('niche', 'N/A')}. Clay Webhook: {self.clay_enrichment_webhook_url[:50]}...")

    def _get_default_target_criteria(self) -> Dict[str, Any]:
        return {
            "niche": config.AGENT_TARGET_NICHE_DEFAULT,
            "clay_enrichment_table_webhook_url": os.getenv("CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY"),
            "lead_source_type": "local_csv",
            "lead_source_path": "data/initial_leads.csv", # CSV with headers: company_name,domain,primary_contact_email_guess
            "lead_source_csv_field_mapping": { # Maps CSV headers to internal lead fields
                "company_name": "company_name", "domain": "domain", "email": "primary_contact_email_guess"
            },
            "min_company_size": 20, "max_company_size": 1500,
            "industries_keywords": ["saas", "software", "technology", "b2b services", "ai", "automation"],
            "job_titles_keywords": ["ceo", "founder", "cto", "vp sales", "head of growth", "marketing director"],
            "qualification_llm_score_threshold": 6,
            "max_leads_to_process_per_cycle": self.batch_size, # Use batch_size from main config
            "run_interval_seconds": config.ACQUISITION_AGENT_RUN_INTERVAL_SECONDS,
            "batch_size": config.ACQUISITION_AGENT_BATCH_SIZE,
            # This config is for ResourceManager if it needs to acquire a Clay API key via automation
            "clay_service_automation_config": DEFAULT_CLAY_SERVICE_AUTOMATION_CONFIG
        }

    async def start(self): # Same as before
        if self._is_running: logger.warning(f"AcquisitionAgent {self.agent_id} already running."); return
        logger.info(f"AcquisitionAgent {self.agent_id} starting run cycle (Interval: {self.run_interval_seconds}s).")
        self._is_running = True
        self._current_task = asyncio.create_task(self._run_loop())

    async def stop(self): # Same as before
        if not self._is_running or not self._current_task: logger.warning(f"AcquisitionAgent {self.agent_id} not running."); return
        logger.info(f"AcquisitionAgent {self.agent_id} stopping..."); self._is_running = False
        if self._current_task: self._current_task.cancel()
        try: await self._current_task
        except asyncio.CancelledError: logger.info(f"AcquisitionAgent {self.agent_id} cycle task cancelled.")
        self._current_task = None; logger.info(f"AcquisitionAgent {self.agent_id} stopped.")

    async def _run_loop(self):
        while self._is_running:
            start_ts = time.monotonic()
            sourced_count, sent_to_clay_count, qualified_count = 0, 0, 0
            try:
                logger.info(f"AcquisitionAgent {self.agent_id}: Starting new acquisition cycle.")
                sourced_count, sent_to_clay_count, qualified_count = await self._run_acquisition_cycle()
                if self.on_cycle_complete_callback:
                    await self.on_cycle_complete_callback(self.agent_id, sourced_count, sent_to_clay_count, qualified_count)
            except Exception as e:
                logger.error(f"AcquisitionAgent {self.agent_id}: Unhandled error in acquisition cycle: {e}", exc_info=True)
                if self.on_cycle_error_callback: await self.on_cycle_error_callback(self.agent_id, str(e))
            
            elapsed = time.monotonic() - start_ts
            wait_time = max(0, self.run_interval_seconds - elapsed)
            if self._is_running:
                logger.info(f"AcquisitionAgent {self.agent_id}: Cycle took {elapsed:.2f}s. Sourced: {sourced_count}, SentToClay: {sent_to_clay_count}, Qualified: {qualified_count}. Waiting {wait_time:.2f}s.")
                await asyncio.sleep(wait_time)

    async def _source_initial_leads_from_csv(self) -> List[Dict[str, Any]]:
        """Reads leads from a local CSV file."""
        leads = []
        path = self.target_criteria.get("lead_source_path", "data/initial_leads.csv")
        mapping = self.target_criteria.get("lead_source_csv_field_mapping", {"company_name": "company_name", "domain": "domain", "email": "primary_contact_email_guess"})
        
        # Create dummy CSV if it doesn't exist for testing
        data_dir = os.path.dirname(path)
        if data_dir and not os.path.exists(data_dir): os.makedirs(data_dir, exist_ok=True)
        if not os.path.exists(path):
            logger.warning(f"Lead source CSV not found at {path}. Creating a dummy file.")
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(mapping.keys()) # Write headers based on mapping keys
                writer.writerow(["Test CSV Company A", "testcsvcompanya.com", f"contact.csv.{int(time.time())}@example.com"])
                writer.writerow(["Test CSV Company B", "testcsvcompanyb.io", f"info.csv.{int(time.time())}@example.com"])

        try:
            with open(path, 'r', newline='', encoding='utf-8-sig') as f: # utf-8-sig for BOM
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= self.target_criteria.get("max_leads_to_process_per_cycle", self.batch_size): break
                    lead_item = {}
                    for target_field, source_field in mapping.items():
                        lead_item[target_field] = row.get(source_field)
                    # Add a unique ID for initial CRM upsert if not present
                    if not lead_item.get("primary_contact_email_guess") and lead_item.get("domain"): # Fallback if no email
                        lead_item["primary_contact_email_guess"] = f"noemail_{lead_item['domain'].replace('.', '_')}@placeholder.com"
                    
                    # Basic validation: need at least a domain or an email to be useful
                    if lead_item.get("domain") or lead_item.get("primary_contact_email_guess"):
                        leads.append(lead_item)
            logger.info(f"Loaded {len(leads)} leads from CSV: {path}")
        except Exception as e:
            logger.error(f"Failed to load leads from CSV '{path}': {e}", exc_info=True)
        return leads

    async def _trigger_clay_enrichment_for_leads(self, raw_leads: List[Dict[str, Any]]) -> int:
        """Sends raw leads to Clay.com webhook for enrichment and updates CRM."""
        if not self.clay_enrichment_webhook_url: return 0
        sent_to_clay_count = 0
        
        # Get Clay API key if needed for webhook auth (DataWrapper might handle this if configured)
        # clay_auth_token = await self.resource_manager.get_api_key("clay.com", self.target_criteria.get("clay_service_automation_config"))

        for lead_data in raw_leads:
            correlation_id = str(uuid.uuid4())
            # Prepare payload for Clay. Keys must match what Clay table expects.
            clay_payload = {
                "domain": lead_data.get("domain"), # Must be a key your Clay table uses as input
                "company_name": lead_data.get("company_name"), # Optional, but good for context
                "email": lead_data.get("primary_contact_email_guess"), # Optional input
                "_correlation_id": correlation_id, # CRITICAL: Send this to Clay
                "_source_agent_id": self.agent_id,
                "_timestamp_sent_utc": datetime.now(timezone.utc).isoformat()
            }
            clay_payload_cleaned = {k: v for k, v in clay_payload.items() if v is not None}
            if not clay_payload_cleaned.get("domain") and not clay_payload_cleaned.get("email"):
                logger.warning(f"Skipping lead due to missing domain/email for Clay: {lead_data.get('company_name')}")
                continue

            try:
                send_success = await self.data_wrapper.send_data_to_clay_webhook(
                    self.clay_enrichment_webhook_url, clay_payload_cleaned #, clay_auth_token=clay_auth_token # If webhook needs auth
                )
                if send_success:
                    sent_to_clay_count += 1
                    crm_lead_data = {
                        "email": lead_data.get("primary_contact_email_guess") or f"noemail_{correlation_id}@placeholder.com",
                        "company_name": lead_data.get("company_name"), "domain": lead_data.get("domain"),
                        "status": "Enrichment_Triggered_Clay",
                        "last_activity_timestamp": datetime.now(timezone.utc).isoformat(),
                        "correlation_id_clay": correlation_id,
                        "source_info": f"AcqAgent_{self.agent_id}_CSV" if self.target_criteria.get("lead_source_type") == "local_csv" else f"AcqAgent_{self.agent_id}",
                        "raw_lead_data_json": lead_data
                    }
                    await self.crm_wrapper.upsert_contact(crm_lead_data, unique_key_column="email")
                    logger.info(f"Lead for {lead_data.get('domain') or lead_data.get('company_name')} sent to Clay. Correlation ID: {correlation_id}")
                else: # send_data_to_clay_webhook now raises ClayWebhookError on HTTP failure
                    logger.error(f"DataWrapper reported failure sending lead {lead_data.get('domain')} to Clay (should have raised).")
            except ClayWebhookError as e:
                logger.error(f"ClayWebhookError sending lead {lead_data.get('domain')}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error sending lead {lead_data.get('domain')} for Clay enrichment: {e}", exc_info=True)
        return sent_to_clay_count

    async def handle_clay_enrichment_result(self, clay_result_payload: Dict[str, Any]) -> bool:
        """Processes enriched data received from Clay (called by server webhook)."""
        async with self._processing_lock: # Ensure atomic processing for a given correlation_id
            correlation_id = clay_result_payload.get("_correlation_id")
            if not correlation_id:
                logger.error(f"Received Clay result without _correlation_id. Payload: {clay_result_payload}")
                return False

            logger.info(f"Processing Clay enrichment result for correlation_id: {correlation_id}")
            contact_to_update = await self.crm_wrapper.get_contact_info(identifier=correlation_id, identifier_column="correlation_id_clay")

            if not contact_to_update:
                logger.error(f"No contact found in CRM for correlation_id_clay: {correlation_id}. Clay result cannot be mapped.")
                return False
            
            contact_id = contact_to_update.get("id")
            logger.info(f"Found contact ID {contact_id} for correlation_id {correlation_id}. Updating with enrichment data.")

            # Map Clay result fields to your CRM contact fields
            # This mapping is CRITICAL and depends on your Clay table's output columns
            update_payload = {
                "id": contact_id, # Must use primary key for update
                "status": "Enriched_From_Clay_Pending_Analysis",
                "last_activity_timestamp": datetime.now(timezone.utc).isoformat(),
                "clay_enrichment_data_json": clay_result_payload, # Store the full raw enrichment
                "job_title": clay_result_payload.get("Title") or clay_result_payload.get("jobTitle") or contact_to_update.get("job_title"),
                "linkedin_url": clay_result_payload.get("LinkedIn Personal URL") or clay_result_payload.get("linkedinUrl") or contact_to_update.get("linkedin_url"),
                "company_linkedin_url": clay_result_payload.get("LinkedIn Company URL") or contact_to_update.get("company_linkedin_url"),
                "company_size_range": clay_result_payload.get("Company Size") or contact_to_update.get("company_size_range"), # Assuming Clay gives range
                "industry": clay_result_payload.get("Industry") or contact_to_update.get("industry"),
                "company_description": clay_result_payload.get("Company Description") or contact_to_update.get("company_description"),
                "phone_number": clay_result_payload.get("Phone Number") or clay_result_payload.get("Direct Dial") or contact_to_update.get("phone_number"),
                # Add more mappings as per your Clay table output and CRM schema
            }
            # Ensure email isn't accidentally changed if it's the unique upsert key for contacts
            update_payload["email"] = contact_to_update.get("email") 

            updated_contact = await self.crm_wrapper.upsert_contact(update_payload, unique_key_column="id")

            if updated_contact:
                logger.info(f"Contact ID {contact_id} updated with Clay enrichment. Ready for analysis.")
                # Optionally trigger immediate analysis or add to a queue
                # await self._analyze_and_qualify_batch([updated_contact])
                return True
            else:
                logger.error(f"Failed to update contact ID {contact_id} in CRM after Clay enrichment.")
                return False

    async def _analyze_and_qualify_batch(self) -> int:
        """Fetches a batch of 'Enriched_From_Clay_Pending_Analysis' leads and qualifies them."""
        if not self.crm_wrapper.supabase: return 0
        
        logger.info("Querying for enriched leads pending analysis...")
        query = (
            self.crm_wrapper.supabase.table(config.SUPABASE_CONTACTS_TABLE)
            .select("*")
            .eq("status", "Enriched_From_Clay_Pending_Analysis")
            .limit(self.batch_size)
        )
        response = await self.crm_wrapper._execute_supabase_query(query) # Use helper
        
        if not response or not response.data:
            logger.info("No enriched leads found pending analysis in this batch.")
            return 0

        leads_to_analyze = response.data
        logger.info(f"Found {len(leads_to_analyze)} enriched leads for analysis.")
        
        qualified_in_batch = 0
        for lead_data in leads_to_analyze:
            async with self._processing_lock: # Ensure one analysis at a time if it modifies shared state or for rate limiting
                lead_id_for_logs = lead_data.get("id") or lead_data.get("email", "Unknown Lead")
                logger.debug(f"Analyzing lead: {lead_id_for_logs}")

                # Prepare context for LLM (more selective than dumping all of clay_enrichment_data_json)
                clay_data = lead_data.get("clay_enrichment_data_json", {})
                context_for_llm = {
                    "company_name": lead_data.get("company_name") or clay_data.get("companyNameEnriched"),
                    "domain": lead_data.get("domain") or clay_data.get("domain"), # Domain from Clay if enriched
                    "industry": lead_data.get("industry") or clay_data.get("Industry"),
                    "company_size_range": lead_data.get("company_size_range") or clay_data.get("Company Size"),
                    "job_title": lead_data.get("job_title") or clay_data.get("Title"),
                    "company_description": lead_data.get("company_description") or clay_data.get("Company Description"),
                    "primary_contact_name": f"{lead_data.get('first_name','')} {lead_data.get('last_name','')} ".strip() or clay_data.get("Full Name"),
                    # Add a few more key fields from your Clay output that are relevant for qualification
                    "website_technologies": clay_data.get("BuiltWith - Tech Stack"), # Example
                    "funding_summary": clay_data.get("Crunchbase - Last Funding Round Amount & Date"), # Example
                }
                context_for_llm_cleaned = {k:v for k,v in context_for_llm.items() if v is not None and v != ""}
                lead_context_str = json.dumps(context_for_llm_cleaned, indent=2, default=str)

                prompt = ( # Same "Level 25" prompt as before
                    f"You are an expert B2B sales development analyst for Boutique AI..." # Truncated for brevity
                    # ... (full prompt from previous AcquisitionAgent version) ...
                    f"Based ONLY on the provided data:\n```json\n{lead_context_str}\n```\n\n"
                    f"Respond with a single, valid JSON object with keys: `likelihood_score` (int 1-10), `primary_inferred_pain_point` (str), `secondary_inferred_pain_point` (str, optional), `suggested_outreach_hook` (str), `is_qualified` (bool, score >= {self.target_criteria.get('qualification_llm_score_threshold', 6)}), `qualification_reasoning` (str), `confidence_in_analysis` (Low/Medium/High), `suggested_next_action` (str)."
                )
                messages = [{"role": "user", "content": prompt}]
                
                analysis_result_json_str = await self.llm_client.generate_response(
                    messages=messages, temperature=0.2, max_tokens=800 # Increased tokens for detailed reasoning
                )

                if not analysis_result_json_str or "error" in analysis_result_json_str.lower():
                    logger.error(f"LLM analysis failed for lead {lead_id_for_logs}. Response: {analysis_result_json_str}")
                    await self.crm_wrapper.update_contact_status_and_notes(lead_data["id"], "Analysis_Failed_LLM", f"LLM Error: {analysis_result_json_str}")
                    continue

                try:
                    analysis = json.loads(analysis_result_json_str)
                    logger.info(f"LLM Analysis for {lead_id_for_logs}: Score={analysis.get('likelihood_score')}, Qualified={analysis.get('is_qualified')}")

                    crm_update_payload = {
                        "id": lead_data["id"],
                        "status": f"Qualified_Sales_Ready" if analysis.get("is_qualified") else f"Disqualified_Analysis",
                        "llm_qualification_score": analysis.get("likelihood_score"),
                        "llm_inferred_pain_point_1": analysis.get("primary_inferred_pain_point"),
                        "llm_inferred_pain_point_2": analysis.get("secondary_inferred_pain_point"),
                        "llm_suggested_hook": analysis.get("suggested_outreach_hook"),
                        "llm_qualification_reasoning": analysis.get("qualification_reasoning"),
                        "llm_analysis_confidence": analysis.get("confidence_in_analysis"),
                        "llm_suggested_next_action": analysis.get("suggested_next_action"),
                        "last_activity_timestamp": datetime.now(timezone.utc).isoformat(),
                        "llm_full_analysis_json": analysis
                    }
                    await self.crm_wrapper.upsert_contact(crm_update_payload, unique_key_column="id")

                    if analysis.get("is_qualified"):
                        qualified_in_batch += 1
                        logger.info(f"Lead {lead_id_for_logs} QUALIFIED. Next action: {analysis.get('suggested_next_action')}")
                    else:
                        logger.info(f"Lead {lead_id_for_logs} NOT QUALIFIED. Reason: {analysis.get('qualification_reasoning')}")
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse LLM analysis JSON for {lead_id_for_logs}: {analysis_result_json_str[:200]}...")
                    await self.crm_wrapper.update_contact_status_and_notes(lead_data["id"], "Analysis_Failed_JSONParse", analysis_result_json_str[:200])
                except Exception as e:
                    logger.error(f"Error processing LLM analysis for {lead_id_for_logs}: {e}", exc_info=True)
                    await self.crm_wrapper.update_contact_status_and_notes(lead_data["id"], f"Analysis_Failed_Exception", str(e)[:200])
                
                await asyncio.sleep(config.get_int_env_var("ACQ_AGENT_LLM_DELAY_S", default=1, required=False)) # Configurable delay
        return qualified_in_batch

    async def _run_acquisition_cycle(self) -> Tuple[int, int, int]:
        """Main operational cycle: source, trigger enrichment, analyze."""
        total_sourced_this_cycle = 0
        total_sent_to_clay_this_cycle = 0
        total_qualified_this_cycle = 0

        # 1. Source new raw leads and upsert them to CRM with "New_Raw_Lead" status
        raw_leads = await self._source_initial_leads_from_csv() # Or other configured source
        if raw_leads:
            logger.info(f"Sourced {len(raw_leads)} new raw leads.")
            # Upsert these raw leads to CRM to give them an ID and initial status
            for raw_lead_data in raw_leads:
                # Ensure a unique key for upsert, e.g., email or a generated placeholder if email is missing
                unique_email = raw_lead_data.get("primary_contact_email_guess") or \
                               f"placeholder_{raw_lead_data.get('domain', str(uuid.uuid4()))}@lead.placeholder"
                
                crm_payload = {
                    "email": unique_email,
                    "company_name": raw_lead_data.get("company_name"),
                    "domain": raw_lead_data.get("domain"),
                    "status": "New_Raw_Lead", # Initial status before sending to Clay
                    "source_info": f"AcqAgent_{self.agent_id}_{self.target_criteria.get('lead_source_type', 'unknown_source')}",
                    "raw_lead_data_json": raw_lead_data,
                    "last_activity_timestamp": datetime.now(timezone.utc).isoformat()
                }
                await self.crm_wrapper.upsert_contact(crm_payload, unique_key_column="email")
            total_sourced_this_cycle = len(raw_leads)
        else:
            logger.info("No new raw leads sourced in this cycle.")

        # 2. Fetch "New_Raw_Lead" (or similar) leads from CRM and trigger Clay enrichment
        if self.crm_wrapper.supabase:
            query = (
                self.crm_wrapper.supabase.table(config.SUPABASE_CONTACTS_TABLE)
                .select("*")
                .eq("status", "New_Raw_Lead") # Or whatever status indicates ready for Clay
                .limit(self.batch_size)
            )
            response = await self.crm_wrapper._execute_supabase_query(query)
            if response and response.data:
                leads_for_clay = response.data
                logger.info(f"Found {len(leads_for_clay)} leads in CRM to send for Clay enrichment.")
                sent_count = await self._trigger_clay_enrichment_for_leads(leads_for_clay)
                total_sent_to_clay_this_cycle = sent_count
            else:
                logger.info("No leads in CRM found needing to be sent to Clay for enrichment.")
        
        # 3. Analyze leads that have been enriched (status "Enriched_From_Clay_Pending_Analysis")
        # This status is set by `handle_clay_enrichment_result` when Clay calls back.
        qualified_count = await self._analyze_and_qualify_batch()
        total_qualified_this_cycle = qualified_count
        
        # Log cycle summary
        logger.info(f"Acquisition cycle summary: Sourced={total_sourced_this_cycle}, SentToClay={total_sent_to_clay_this_cycle}, NewlyQualified={total_qualified_this_cycle}")
        return total_sourced_this_cycle, total_sent_to_clay_this_cycle, total_qualified_this_cycle

# --- Test function (more comprehensive) ---
async def _test_acquisition_agent_level40():
    print("--- Testing AcquisitionAgent (Level 40) ---")
    if not all([config.OPENROUTER_API_KEY, config.SUPABASE_ENABLED, os.getenv("CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY")]):
        print("Skipping AcquisitionAgent L40 test: Missing OpenRouter key, Supabase config, or CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY in .env.")
        return

    class MockBrowserAutomatorForAcqTest(BrowserAutomatorInterface): # Basic Mock
        async def setup_session(self, *args, **kwargs): return True
        async def close_session(self, *args, **kwargs): pass
        async def navigate_to_page(self, *args, **kwargs): return True
        async def take_screenshot(self, *args, **kwargs): return b"dummy_bytes"
        async def fill_form_and_submit(self, *args, **kwargs): return True
        async def check_success_condition(self, *args, **kwargs): return True
        async def extract_resources_from_page(self, *args, **kwargs): return {"api_key": "mock_acq_test_key"}
        async def solve_captcha_if_present(self, *args, **kwargs): return True
        async def full_signup_and_extract(self, service_name, *args, **kwargs):
            if service_name == "clay.com": return {"status": "success", "extracted_resources": {"api_key": "mock_clay_key_acq_agent"}}
            return {"status": "failed", "reason": "Mocked failure"}

    llm = LLMClient()
    fp_gen = FingerprintGenerator(llm_client=llm)
    browser_automator = MockBrowserAutomatorForAcqTest() # Use mock for this test
    
    resource_manager = ResourceManager(llm_client=llm, fingerprint_generator=fp_gen, browser_automator=browser_automator)
    data_wrapper = DataWrapper()
    crm_wrapper = CRMWrapper()

    # Create dummy CSV for testing
    test_csv_path = "data/test_acq_leads.csv"
    if not os.path.exists(os.path.dirname(test_csv_path)): os.makedirs(os.path.dirname(test_csv_path))
    with open(test_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["company_name", "domain", "primary_contact_email_guess"]) # Match default mapping
        ts = int(time.time())
        writer.writerow([f"CSV Lead Alpha {ts}", f"csvleadalpha{ts}.com", f"alpha.{ts}@examplecsv.com"])
        writer.writerow([f"CSV Lead Beta {ts+1}", f"csvleadbeta{ts+1}.io", f"beta.{ts+1}@examplecsv.com"])

    agent_criteria = {
        "clay_enrichment_table_webhook_url": os.getenv("CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY"),
        "lead_source_type": "local_csv",
        "lead_source_path": test_csv_path, # Use the test CSV
        "max_leads_to_process_per_cycle": 5, # Small batch for test
        "qualification_llm_score_threshold": 7,
        "clay_service_automation_config": DEFAULT_CLAY_SERVICE_AUTOMATION_CONFIG
    }

    acq_agent = AcquisitionAgent(
        agent_id="AcqAgent_L40_Test", resource_manager=resource_manager, data_wrapper=data_wrapper,
        llm_client=llm, crm_wrapper=crm_wrapper, target_criteria=agent_criteria
    )

    print("\n--- Running one acquisition cycle (sourcing & sending to Clay) ---")
    sourced, sent_to_clay, _ = await acq_agent._run_acquisition_cycle()
    print(f"Cycle 1: Sourced={sourced}, SentToClay={sent_to_clay}")

    # Simulate Clay calling back for one of the leads
    if crm_wrapper.supabase and sent_to_clay > 0:
        print("\n--- Simulating Clay Webhook Callback for one lead ---")
        # Find a lead that was just sent
        pending_response = await asyncio.to_thread(
            crm_wrapper.supabase.table(config.SUPABASE_CONTACTS_TABLE)
            .select("id, correlation_id_clay, email, company_name")
            .eq("status", "Enrichment_Triggered_Clay")
            .order("created_at", desc=True) # Get one of the recent ones
            .limit(1).execute
        )
        if pending_response.data:
            lead_for_callback = pending_response.data[0]
            corr_id = lead_for_callback["correlation_id_clay"]
            print(f"Simulating callback for contact: {lead_for_callback['company_name']} (Correlation ID: {corr_id})")
            
            dummy_clay_result = { # This structure MUST match what your Clay table actually outputs
                "_correlation_id": corr_id,
                "Company Name (from Clay)": lead_for_callback['company_name'] + " (Enriched by Clay)",
                "Industry": "AI Automation Solutions", "Company Size": "51-200 employees",
                "Website": lead_for_callback.get('domain', 'domainfromclay.com'),
                "LinkedIn Company URL": f"https://linkedin.com/company/{lead_for_callback['company_name'].lower().replace(' ','-')}-clay",
                "Key People": [{"name": "Jane Doe", "title": "VP of Innovation", "email": f"jane.doe.{int(time.time())}@example.com"}],
                "Funding Stage": "Series A",
                "Recent News": "Launched new AI product last quarter."
            }
            await acq_agent.handle_clay_enrichment_result(dummy_clay_result)
            
            print("\n--- Running analysis part of acquisition cycle ---")
            # This will pick up the lead whose status was changed by handle_clay_enrichment_result
            _, _, qualified_after_callback = await acq_agent._run_acquisition_cycle() # This will run sourcing again (0 new), then analysis
            print(f"Cycle 2 (Analysis): Qualified={qualified_after_callback}")
            print(f"Check Supabase for contact {lead_for_callback['email']} - should be analyzed and qualified/disqualified.")
        else:
            print("Could not find a lead in 'Enrichment_Triggered_Clay' status to simulate callback for.")
    
    await DataWrapper.close_session()
    if os.path.exists(test_csv_path): os.remove(test_csv_path) # Clean up dummy CSV

if __name__ == "__main__":
    # import asyncio
    # from dotenv import load_dotenv
    # load_dotenv()
    # asyncio.run(_test_acquisition_agent_level40())
    print("AcquisitionAgent (Level 40) defined. Test requires full .env, Supabase, and Clay webhook setup.")
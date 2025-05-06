# core/agents/acquisition_agent.py

import asyncio
import logging
import random
import time
from typing import Dict, Any, Optional, List, Callable, Coroutine

# Import dependencies from other modules
from core.agents.resource_manager import ResourceManager # Manages proxies, conceptual trials, keys
from core.services.data_wrapper import DataWrapper # Interacts with Clay tables via API
from core.services.llm_client import LLMClient # For lead analysis, scoring, angle generation
from core.services.crm_wrapper import CRMWrapper # For storing qualified leads in Supabase
# Import configuration centrally if needed for defaults
import config

logger = logging.getLogger(__name__)

class AcquisitionAgent:
    """
    Autonomous AI agent responsible for identifying, enriching, analyzing,
    and qualifying potential B2B leads using resourceful methods, primarily
    leveraging Clay.com data (via table interactions) and AI analysis.
    Stores qualified leads in the Supabase database via CRMWrapper.
    """

    def __init__(
        self,
        agent_id: str,
        # --- Dependency Injection ---
        resource_manager: ResourceManager,
        data_wrapper: DataWrapper, # Clay table interaction
        llm_client: LLMClient,
        crm_wrapper: CRMWrapper, # Supabase interaction
        # --- Configuration ---
        target_criteria: Optional[Dict[str, Any]] = None, # Specific criteria for this run
        run_interval_seconds: int = 3600 * 4, # How often to run the cycle (e.g., 4 hours)
        batch_size: int = 50, # How many leads to process per cycle
        # --- Callbacks (Optional - for Orchestrator interaction) ---
        on_cycle_complete_callback: Optional[Callable[[str, int], Coroutine[Any, Any, None]]] = None, # Args: agent_id, leads_added
        on_cycle_error_callback: Optional[Callable[[str, str], Coroutine[Any, Any, None]]] = None # Args: agent_id, error_message
        ):
        """
        Initializes the Acquisition Agent instance.

        Args:
            agent_id: Unique identifier for this agent.
            resource_manager: Instance of ResourceManager.
            data_wrapper: Instance of DataWrapper (already configured with API key if possible).
            llm_client: Instance of LLMClient.
            crm_wrapper: Instance of CRMWrapper (connected to Supabase).
            target_criteria: Dictionary defining the ideal customer profile for this run.
            run_interval_seconds: Frequency at which the acquisition cycle should run.
            batch_size: Number of leads to attempt to process in one cycle.
            on_cycle_complete_callback: Async callback on successful cycle completion.
            on_cycle_error_callback: Async callback on cycle failure.
        """
        self.agent_id = agent_id
        self.resource_manager = resource_manager
        self.data_wrapper = data_wrapper
        self.llm_client = llm_client
        self.crm_wrapper = crm_wrapper
        self.target_criteria = target_criteria if target_criteria else self._get_default_target_criteria()
        self.run_interval_seconds = run_interval_seconds
        self.batch_size = batch_size
        self.on_cycle_complete_callback = on_cycle_complete_callback
        self.on_cycle_error_callback = on_cycle_error_callback

        self._is_running = False
        self._current_task: Optional[asyncio.Task] = None

        logger.info(f"AcquisitionAgent {self.agent_id} initialized. Run interval: {self.run_interval_seconds}s.")

    def _get_default_target_criteria(self) -> Dict[str, Any]:
        """ Defines default lead targeting criteria if none provided. """
        # TODO: Load this from a config file or database for flexibility
        logger.warning("Using default target criteria for lead acquisition.")
        return {
            "clay_table_source_id": "t_YOUR_PREBUILT_LEAD_TABLE_ID", # ** MUST BE SET ** Table built in Clay UI
            "min_company_size": 50,
            "max_company_size": 1000,
            "industries": ["Software", "Technology", "Financial Services"], # Example list
            "job_titles": ["VP Sales", "Head of Sales", "Sales Director", "Chief Revenue Officer"],
            "required_keywords": ["growth", "efficiency", "automation"], # Keywords in profile/company description
            "geo_target": ["US", "CA"] # Target countries/regions
        }

    async def start(self):
        """ Starts the periodic acquisition cycle. """
        if self._is_running:
            logger.warning(f"AcquisitionAgent {self.agent_id} start requested but already running.")
            return
        logger.info(f"AcquisitionAgent {self.agent_id} starting run cycle (Interval: {self.run_interval_seconds}s).")
        self._is_running = True
        # Create task that runs the cycle repeatedly
        self._current_task = asyncio.create_task(self._run_loop())

    async def stop(self):
        """ Stops the acquisition cycle gracefully. """
        if not self._is_running or not self._current_task:
            logger.warning(f"AcquisitionAgent {self.agent_id} stop requested but not running.")
            return
        logger.info(f"AcquisitionAgent {self.agent_id} stopping run cycle...")
        self._is_running = False
        if self._current_task:
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                logger.info(f"AcquisitionAgent {self.agent_id} run cycle task cancelled.")
            except Exception as e:
                 logger.error(f"Error during AcquisitionAgent stop/task cleanup: {e}", exc_info=True)
        self._current_task = None
        logger.info(f"AcquisitionAgent {self.agent_id} stopped.")

    async def _run_loop(self):
        """ The main loop that periodically executes the acquisition cycle. """
        while self._is_running:
            start_time = time.monotonic()
            leads_added = 0
            error_msg = None
            try:
                logger.info(f"AcquisitionAgent {self.agent_id}: Starting new acquisition cycle.")
                leads_added = await self._run_acquisition_cycle()
                logger.info(f"AcquisitionAgent {self.agent_id}: Cycle finished. Added {leads_added} qualified leads.")
                if self.on_cycle_complete_callback:
                     await self.on_cycle_complete_callback(self.agent_id, leads_added)

            except Exception as e:
                error_msg = f"Error in acquisition cycle: {e}"
                logger.error(f"AcquisitionAgent {self.agent_id}: {error_msg}", exc_info=True)
                if self.on_cycle_error_callback:
                     await self.on_cycle_error_callback(self.agent_id, error_msg)

            finally:
                # Wait for the next interval, accounting for execution time
                elapsed_time = time.monotonic() - start_time
                wait_time = max(0, self.run_interval_seconds - elapsed_time)
                logger.info(f"AcquisitionAgent {self.agent_id}: Cycle took {elapsed_time:.2f}s. Waiting {wait_time:.2f}s for next cycle.")
                if self._is_running: # Check again before sleeping
                    await asyncio.sleep(wait_time)

    async def _run_acquisition_cycle(self) -> int:
        """ Executes one full cycle of finding, enriching, qualifying, and storing leads. """
        qualified_leads_added = 0

        # 1. Ensure DataWrapper has API Key
        # (Re-check/set key each cycle in case ResourceManager state changes)
        clay_api_key = await self.resource_manager.get_clay_api_key()
        if not clay_api_key:
             logger.error(f"AcquisitionAgent {self.agent_id}: Cannot run cycle, failed to get Clay API Key.")
             return 0
        self.data_wrapper.set_api_key(clay_api_key)

        # 2. Find Initial Leads (e.g., from a pre-built Clay table)
        logger.info(f"AcquisitionAgent {self.agent_id}: Finding initial leads...")
        initial_leads = await self._find_initial_leads(self.target_criteria.get("clay_table_source_id"))
        if not initial_leads:
             logger.info(f"AcquisitionAgent {self.agent_id}: No initial leads found in source table. Cycle ending early.")
             return 0
        logger.info(f"AcquisitionAgent {self.agent_id}: Found {len(initial_leads)} initial leads.")

        # 3. Enrich Leads (Conceptual / High-Risk - Requires external implementation)
        logger.info(f"AcquisitionAgent {self.agent_id}: Enriching {len(initial_leads)} leads (conceptual)...")
        enriched_leads = await self._enrich_leads(initial_leads) # This currently returns input as placeholder
        logger.info(f"AcquisitionAgent {self.agent_id}: Enrichment step completed (conceptual).")

        # 4. Analyze and Qualify Leads using LLM
        logger.info(f"AcquisitionAgent {self.agent_id}: Analyzing and qualifying {len(enriched_leads)} leads...")
        qualified_leads = await self._analyze_and_qualify_leads(enriched_leads)
        logger.info(f"AcquisitionAgent {self.agent_id}: Found {len(qualified_leads)} qualified leads after analysis.")

        # 5. Store Qualified Leads in Supabase
        if qualified_leads:
            logger.info(f"AcquisitionAgent {self.agent_id}: Storing {len(qualified_leads)} qualified leads...")
            qualified_leads_added = await self._store_qualified_leads(qualified_leads)
            logger.info(f"AcquisitionAgent {self.agent_id}: Successfully stored {qualified_leads_added} leads in Supabase.")
        else:
             logger.info(f"AcquisitionAgent {self.agent_id}: No leads met qualification criteria.")

        # 6. Conceptual: Exploit Social Algorithms / Generate SEO Content
        # await self._exploit_social_algorithms(qualified_leads)
        # await self._generate_seo_content(self.target_criteria)

        return qualified_leads_added


    async def _find_initial_leads(self, source_table_id: Optional[str]) -> List[Dict[str, Any]]:
        """
        Fetches initial lead data, likely by reading rows from a specified Clay table
        that was populated via Clay's UI or other means.
        """
        if not source_table_id:
             logger.error("Cannot find initial leads: Clay source table ID not specified in target criteria.")
             return []

        # This assumes the DataWrapper's lookup can be adapted or a 'read_table' method exists.
        # For now, simulate reading some rows conceptually.
        # In reality, you'd need to implement pagination or targeted lookups.
        logger.warning(f"_find_initial_leads needs specific implementation for reading from Clay table '{source_table_id}'. Simulating.")
        # Example: Simulate reading a few rows that might match criteria conceptually
        simulated_leads = []
        for i in range(self.batch_size // 5): # Simulate finding a fraction of the batch size
             simulated_leads.append({
                 "id_from_clay": f"clay_lead_{int(time.time()*1000 + i)}",
                 "Company Name": f"Potential Client Inc {i}",
                 "Company Domain": f"potential{i}.com",
                 "Contact Name": f"Decision Maker {i}",
                 "Contact Email": f"decision.maker{i}@potential{i}.com",
                 "Contact Title": random.choice(self.target_criteria.get("job_titles", ["VP Sales"])),
                 "Industry": random.choice(self.target_criteria.get("industries", ["Software"])),
                 # Add other fields available in your source Clay table
             })
        # Replace simulation with actual DataWrapper calls to read from the table
        # Example (requires DataWrapper method):
        # initial_leads = await self.data_wrapper.read_table_rows(source_table_id, limit=self.batch_size)
        return simulated_leads


    async def _enrich_leads(self, leads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enriches lead data using resourceful methods.
        **CONCEPTUAL / HIGH-RISK PLACEHOLDER:** Actual implementation would likely
        involve browser automation to interact with Clay trial accounts or other
        non-standard data sources, using proxies and generated fingerprints.

        Args:
            leads: List of lead dictionaries (potentially with basic info).

        Returns:
            List of lead dictionaries, potentially enriched with more data.
            (This placeholder currently returns the input unmodified).
        """
        logger.warning("Executing CONCEPTUAL lead enrichment step.")
        enriched_leads_list = []
        for lead in leads:
            logger.debug(f"Conceptually enriching lead: {lead.get('Contact Email') or lead.get('Company Name')}")
            # --- ### START CONCEPTUAL BROWSER AUTOMATION / SCRAPING ### ---
            # 1. Get resource bundle (proxy, conceptual creds, fingerprint)
            #    bundle = await self.resource_manager.get_resource_bundle("clay.com") # Or other service
            #    if not bundle or not bundle.get('credentials') or not bundle.get('proxy_string'):
            #        logger.error("Failed to get resources for enrichment. Skipping lead.")
            #        enriched_leads_list.append(lead) # Keep original lead
            #        continue

            # 2. Instantiate/Use a Browser Automation Wrapper (requires separate implementation)
            #    browser_wrapper = BrowserAutomationWrapper(proxy=bundle['proxy_string'], fingerprint=bundle['fingerprint_profile'])
            #    try:
            #        # Login to conceptual Clay trial account
            #        await browser_wrapper.login("clay.com", bundle['credentials']['username'], bundle['credentials']['password'])
            #        # Navigate to enrichment tools, input lead data (e.g., email, domain)
            #        # Run enrichments, scrape results from the UI
            #        enrichment_data = await browser_wrapper.enrich_lead_via_clay_ui(lead)
            #        # Merge enrichment_data back into the lead dictionary
            #        lead.update(enrichment_data)
            #        logger.debug("Conceptual enrichment successful.")
            #    except Exception as e:
            #         logger.error(f"Conceptual enrichment failed for lead: {e}", exc_info=False)
            #    finally:
            #         await browser_wrapper.close()
            # --- ### END CONCEPTUAL BROWSER AUTOMATION / SCRAPING ### ---

            # Placeholder: Just add a dummy enrichment field
            lead['enrichment_status'] = "Conceptual Enrichment Placeholder"
            enriched_leads_list.append(lead)
            await asyncio.sleep(0.05) # Simulate some processing time

        return enriched_leads_list

    async def _analyze_and_qualify_leads(self, leads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ Uses LLM to analyze enriched leads, score them, and identify angles. """
        qualified_leads = []
        for lead in leads:
            logger.debug(f"Analyzing lead: {lead.get('Contact Email') or lead.get('Company Name')}")
            # Prepare context for LLM analysis
            lead_context = json.dumps(lead, indent=2, default=str) # Present lead data clearly
            prompt = (f"Analyze the following B2B lead data for Boutique AI (specializing in autonomous AI sales agents for {self.target_niche}):\n\n"
                      f"```json\n{lead_context}\n```\n\n"
                      f"Based ONLY on the provided data:\n"
                      f"1. Score the lead's fit (1-10) for needing sales automation/efficiency (Likelihood_Score).\n"
                      f"2. Briefly state the primary inferred pain point related to sales (Inferred_Pain_Point).\n"
                      f"3. Suggest a compelling 1-sentence outreach hook tailored to this lead (Outreach_Hook).\n"
                      f"4. Qualify the lead (True/False) based on a Likelihood_Score >= 6 and presence of relevant job title/industry (Is_Qualified).\n"
                      f"Return ONLY a valid JSON object with keys: Likelihood_Score (int), Inferred_Pain_Point (str), Outreach_Hook (str), Is_Qualified (bool)."
                     )
            messages = [{"role": "user", "content": prompt}]
            analysis_result_text = await self.llm_client.generate_response(messages=messages, temperature=0.3, use_cache=False) # Low temp for consistent analysis

            if analysis_result_text:
                try:
                    analysis = json.loads(analysis_result_text)
                    if analysis.get("Is_Qualified") is True:
                         # Add analysis results back to the lead dictionary
                         lead['likelihood_score'] = analysis.get('Likelihood_Score')
                         lead['inferred_pain_point'] = analysis.get('Inferred_Pain_Point')
                         lead['suggested_hook'] = analysis.get('Outreach_Hook')
                         lead['qualification_status'] = 'Qualified'
                         qualified_leads.append(lead)
                         logger.info(f"Lead Qualified: {lead.get('Contact Email') or lead.get('Company Name')}. Score: {lead.get('likelihood_score')}")
                    else:
                         logger.debug(f"Lead Not Qualified: {lead.get('Contact Email') or lead.get('Company Name')}. Reason: Low score or failed criteria.")
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                     logger.error(f"Failed to parse LLM analysis JSON for lead: {e}. Response: {analysis_result_text[:100]}...")
            else:
                 logger.error("LLM analysis failed for lead.")
            await asyncio.sleep(0.1) # Avoid hitting LLM rate limits too hard

        return qualified_leads

    async def _store_qualified_leads(self, leads: List[Dict[str, Any]]) -> int:
        """ Stores qualified leads into the Supabase 'contacts' table via CRMWrapper. """
        added_count = 0
        for lead in leads:
            # Map lead data to the expected Supabase 'contacts' table schema
            # Assumes CRMWrapper handles the actual insert logic
            contact_data = {
                "phone_number": lead.get("Contact Phone") or lead.get("Phone Number"), # Find phone number field
                "email": lead.get("Contact Email") or lead.get("Email"),
                "first_name": lead.get("Contact Name", "").split(" ")[0] if lead.get("Contact Name") else None,
                "last_name": " ".join(lead.get("Contact Name", "").split(" ")[1:]) if lead.get("Contact Name") and " " in lead.get("Contact Name") else None,
                "company_name": lead.get("Company Name"),
                "title": lead.get("Contact Title"),
                "source": f"ClayTable_{self.target_criteria.get('clay_table_source_id', 'Unknown')}",
                "status": "New Qualified Lead",
                # Store analysis results if table schema supports it (e.g., in a JSONB column)
                "qualification_details": json.dumps({
                     "likelihood_score": lead.get('likelihood_score'),
                     "inferred_pain_point": lead.get('inferred_pain_point'),
                     "suggested_hook": lead.get('suggested_hook')
                })
            }
            # Remove None values before insertion if table requires it
            contact_data = {k: v for k, v in contact_data.items() if v is not None}

            # Use CRMWrapper to insert/update contact (CRMWrapper needs implementation for this)
            # This assumes crm_wrapper has an 'upsert_contact' or similar method.
            # Using log_call_outcome is incorrect here. Need a dedicated contact creation method.
            # --- Placeholder for actual contact storage ---
            logger.warning(f"Storing lead requires a dedicated upsert/insert method in CRMWrapper. Simulating storage for: {contact_data.get('email')}")
            # success = await self.crm_wrapper.upsert_contact(contact_data, lookup_key='email') # Example
            success = True # Simulate success
            # --- End Placeholder ---

            if success:
                added_count += 1
            else:
                 logger.error(f"Failed to store qualified lead {contact_data.get('email')} in Supabase.")
            await asyncio.sleep(0.05) # Slight delay between inserts

        return added_count

    # --- Conceptual Methods for Advanced Tactics ---
    async def _exploit_social_algorithms(self, leads: List[Dict[str, Any]]):
        """ CONCEPTUAL: Analyze social platforms and attempt algorithm exploitation. """
        logger.warning("_exploit_social_algorithms is conceptual and requires specific implementation.")
        # 1. Use ResourceManager for proxies/fingerprints.
        # 2. Use scraping/automation wrapper to analyze target profiles/content on X.com/LinkedIn.
        # 3. Use LLMClient to determine optimal engagement strategy/timing/content.
        # 4. Execute engagement via automation wrapper.
        pass

    async def _generate_seo_content(self, criteria: Dict[str, Any]):
        """ CONCEPTUAL: Generate niche SEO content targeting AI algorithm understanding. """
        logger.warning("_generate_seo_content is conceptual and requires specific implementation.")
        # 1. Use LLMClient with prompts focused on answering complex user queries related to niche/pain points.
        # 2. Structure content for "answer engine optimization".
        # 3. Requires separate mechanism for publishing content.
        pass

    async def _manage_trial_accounts(self, service_name: str):
        """ CONCEPTUAL: Orchestrate conceptual trial creation/usage. """
        logger.warning("_manage_trial_accounts is conceptual. Calls high-risk placeholder in ResourceManager.")
        # This would involve calling resource_manager._get_or_create_conceptual_trial()
        # and handling the resulting credentials/keys, potentially rotating them.
        pass


# (Conceptual Test Runner)
async def main():
    print("Testing AcquisitionAgent (Structure)...")
    # Requires full dependency setup (Supabase, OpenRouter key, etc.)
    if not config.SUPABASE_ENABLED or not config.OPENROUTER_API_KEY:
         print("Skipping test: Supabase or OpenRouter not configured.")
         return

    try:
        # Instantiate dependencies
        llm = LLMClient()
        crm = CRMWrapper() # Uses Supabase
        rm = ResourceManager(llm_client=llm) # Uses Supabase, needs LLM for fingerprints
        dw = DataWrapper() # Needs key set

        # Configure DataWrapper (needs key from RM or config)
        clay_key = await rm.get_clay_api_key()
        if clay_key: dw.set_api_key(clay_key)
        else: print("Warning: Could not get Clay API key for DataWrapper in test."); return

        # Create agent
        agent = AcquisitionAgent(
            agent_id="AcqAgent_Test01",
            resource_manager=rm,
            data_wrapper=dw,
            llm_client=llm,
            crm_wrapper=crm,
            run_interval_seconds=10 # Short interval for testing
        )

        print("Starting agent cycle (will run once then stop)...")
        # Run one cycle for testing instead of the full loop
        await agent._run_acquisition_cycle()
        print("Agent cycle finished.")

    except Exception as e:
        print(f"An error occurred during test: {e}")

if __name__ == "__main__":
    # import asyncio
    # import config # Ensure config is loaded
    # asyncio.run(main()) # Uncomment to run test (requires async context and all services configured)
    print("AcquisitionAgent structure defined. Run test manually in an async context.")


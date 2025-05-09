# /server.py:
# --------------------------------------------------------------------------------
# boutique_ai_project/server.py

import logging
import asyncio
import base64
import json
import os
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone # Added imports
from urllib.parse import urlparse # Added for robust URL handling

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Response as FastAPIResponse, Header, Query, Depends
from fastapi.responses import PlainTextResponse
# from fastapi.routing import APIRoute # Unused
from starlette.websockets import WebSocketState # Import for checking state

from twilio.twiml.voice_response import VoiceResponse, Start

import config # Root config
from core.services.llm_client import LLMClient
from core.services.fingerprint_generator import FingerprintGenerator
from core.services.telephony_wrapper import TelephonyWrapper
from core.services.crm_wrapper import CRMWrapper
from core.services.data_wrapper import DataWrapper # ClayWebhookError not directly handled here
from core.communication.voice_handler import VoiceHandler
from core.agents.resource_manager import ResourceManager
from core.agents.acquisition_agent import AcquisitionAgent, DEFAULT_CLAY_SERVICE_AUTOMATION_CONFIG
from core.agents.sales_agent import SalesAgent
from core.agents.provisioning_agent import ProvisioningAgent # Added import
from core.automation.browser_automator_interface import BrowserAutomatorInterface
# --- Import the REAL Automator Implementation ---
from core.automation.multimodal_playwright_automator import MultiModalPlaywrightAutomator
# --- OR Import a Mock for testing if Playwright isn't ready ---
# from core.automation.mock_browser_automator import MockBrowserAutomator

from core.services.deployment_manager import DeploymentManagerInterface, DockerDeploymentManager # Added imports
from core.database_setup import setup_supabase_tables, create_client as create_supabase_client_for_setup


# --- Mock Automator (Defined here for simplicity if not using separate file) ---
class MockBrowserAutomatorForServer(BrowserAutomatorInterface): # Renamed to avoid conflict if mock file exists
    def __init__(self, llm_client: Optional[LLMClient] = None): self.llm_client = llm_client; logger.info("MockBrowserAutomatorForServer initialized.")
    async def setup_session(self, *args, **kwargs): logger.info("MockAutomator(Server): setup_session"); return True
    async def close_session(self, *args, **kwargs): logger.info("MockAutomator(Server): close_session")
    async def navigate_to_page(self, url: str, *args, **kwargs): logger.info(f"MockAutomator(Server): navigate_to_page {url}"); return True
    async def take_screenshot(self, *args, **kwargs): logger.info("MockAutomator(Server): take_screenshot"); return b"dummy_screenshot_bytes_content"
    
    # Adjusted to match updated interface in multimodal_playwright_automator.py
    async def fill_form_and_submit(
        self,
        form_interaction_plan: List[Dict[str, Any]], # Changed from form_selectors_and_values
        submit_button_selector: str, # This parameter is less relevant if submit is part of the plan
        page_screenshot_for_llm: Optional[bytes] = None,
        llm_form_fill_prompt: Optional[str] = None,
        signup_details_generated: Optional[Dict[str, Any]] = None # Added
    ) -> bool:
        logger.info(f"MockAutomator(Server): fill_form_and_submit. Plan: {form_interaction_plan}, Details: {signup_details_generated}")
        return True

    async def check_success_condition(self, indicator: Dict[str, Any], *args, **kwargs): logger.info(f"MockAutomator(Server): check_success_condition with indicator: {indicator}"); return True
    async def extract_resources_from_page(self, rules: List[Dict[str, Any]], *args, **kwargs): logger.info(f"MockAutomator(Server): extract_resources_from_page with rules: {rules}"); return {"api_key": "mock_server_extracted_key_123"}
    async def solve_captcha_if_present(self, *args, **kwargs): logger.info("MockAutomator(Server): solve_captcha_if_present"); return True
    async def get_cookies(self) -> Optional[List[Dict[str, Any]]]: logger.info("MockAutomator(Server): get_cookies"); return [{"name": "mock_cookie", "value": "mock_value"}]
    
    async def full_signup_and_extract(
        self,
        service_name: str,
        signup_url: str,
        form_interaction_plan: List[Dict[str, Any]], # Corrected argument name
        signup_details_generated: Dict[str, Any],
        success_indicator: Dict[str, Any], # Corrected type
        resource_extraction_rules: List[Dict[str, Any]], # Corrected type
        captcha_config: Optional[Dict[str, Any]] = None, # Corrected type
        max_retries: int = 0
    ) -> Dict[str, Any]:
        logger.info(f"MockAutomator(Server): full_signup_and_extract for {service_name} at {signup_url}")
        if service_name == "clay.com": return {"status": "success", "extracted_resources": {"api_key": "mock_clay_trial_key_from_server_automator"}, "cookies": []}
        return {"status": "failed", "reason": "Mocked failure by MockAutomator(Server)"}

logger = logging.getLogger("boutique_ai_server") # Consistent logger name

# --- Application State ---
class AppState:
    llm_client: LLMClient
    fingerprint_generator: FingerprintGenerator
    browser_automator: BrowserAutomatorInterface
    resource_manager: ResourceManager
    data_wrapper: DataWrapper
    crm_wrapper: CRMWrapper
    telephony_wrapper: TelephonyWrapper
    acquisition_agent: AcquisitionAgent
    provisioning_agent: ProvisioningAgent # Added
    deployment_manager: DeploymentManagerInterface # Added
    active_sales_agents: Dict[str, SalesAgent] = {}
    supabase_client_for_setup: Optional[Any] = None # SupabaseClient type hint if possible
    is_shutting_down: bool = False

# --- FastAPI Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    logger.info("Boutique AI Server (Orchestrator) initializing...")
    current_app_state = AppState() # Create instance of AppState
    app.state = current_app_state # Attach to app instance immediately

    # --- Database Setup ---
    if config.SUPABASE_ENABLED:
        logger.info("Attempting Supabase database schema setup/verification...")
        try:
            # Ensure this client uses SERVICE_ROLE_KEY for DDL
            current_app_state.supabase_client_for_setup = create_supabase_client_for_setup()
            if current_app_state.supabase_client_for_setup:
                await setup_supabase_tables(current_app_state.supabase_client_for_setup)
                logger.info("Supabase database schema setup/verification complete.")
            else:
                logger.error("Failed to create Supabase client for database setup. Tables may not be created.")
        except Exception as db_setup_e:
            logger.error(f"Error during Supabase database setup: {db_setup_e}", exc_info=True)
            logger.critical("CRITICAL: Database tables might not be correctly set up. Check logs and Supabase dashboard.")
    else:
        logger.warning("Supabase is not enabled (SUPABASE_URL or SUPABASE_KEY not set). Skipping database table setup.")

    # --- Initialize Shared Dependencies ---
    try:
        current_app_state.llm_client = LLMClient()
        current_app_state.fingerprint_generator = FingerprintGenerator(llm_client=current_app_state.llm_client)

        if config.USE_REAL_BROWSER_AUTOMATOR:
            try:
                current_app_state.browser_automator = MultiModalPlaywrightAutomator(
                    llm_client=current_app_state.llm_client,
                    headless=not config.PLAYWRIGHT_HEADFUL_MODE # Use config var
                )
                logger.info("Lifespan Startup: Using REAL MultiModalPlaywrightAutomator.")
            except ImportError:
                 logger.error("Failed to import MultiModalPlaywrightAutomator. Ensure 'playwright' is installed and browser binaries via 'playwright install'. Falling back to Mock.")
                 current_app_state.browser_automator = MockBrowserAutomatorForServer(llm_client=current_app_state.llm_client)
            except Exception as auto_err:
                 logger.error(f"Error initializing MultiModalPlaywrightAutomator: {auto_err}. Falling back to Mock.", exc_info=True)
                 current_app_state.browser_automator = MockBrowserAutomatorForServer(llm_client=current_app_state.llm_client)
        else:
            current_app_state.browser_automator = MockBrowserAutomatorForServer(llm_client=current_app_state.llm_client)
            logger.warning("Lifespan Startup: Using MOCK BrowserAutomator (set USE_REAL_BROWSER_AUTOMATOR=true in .env to enable real one).")

        current_app_state.resource_manager = ResourceManager(
            llm_client=current_app_state.llm_client,
            fingerprint_generator=current_app_state.fingerprint_generator,
            browser_automator=current_app_state.browser_automator
        )
        current_app_state.data_wrapper = DataWrapper()
        current_app_state.crm_wrapper = CRMWrapper()
        current_app_state.telephony_wrapper = TelephonyWrapper()

        # --- Initialize Deployment Manager ---
        # Using DockerDeploymentManager as the concrete implementation
        if config.COOLIFY_APPLICATION_IMAGE or os.getenv("DOCKER_HOST"): # Check if in Docker-like env
            current_app_state.deployment_manager = DockerDeploymentManager(base_image_name=config.COOLIFY_APPLICATION_IMAGE)
            logger.info("DockerDeploymentManager initialized for client agent deployments.")
        else:
            # Fallback or raise error if Docker environment not detected and it's required
            logger.warning("DockerDeploymentManager not initialized (COOLIFY_APPLICATION_IMAGE not set or Docker env unclear). Client provisioning may fail.")
            # If provisioning is critical, you might want to raise an error or use a mock that logs errors.
            # For now, let it proceed, ProvisioningAgent will log errors if manager is not functional.
            # As a dummy placeholder if no Docker env:
            class LoggingDeploymentManager(DeploymentManagerInterface): # Basic mock for non-docker envs
                async def deploy_agent_instance(self, dc) -> Dict[str, Any]: logger.error("LoggingDM: deploy called"); return {"status": "failed", "reason":"Not implemented"}
                async def get_instance_status(self, id) -> Dict[str, Any]: logger.error("LoggingDM: status called"); return {"status":"unknown"}
                async def stop_agent_instance(self, id) -> bool: logger.error("LoggingDM: stop called"); return False
                async def start_agent_instance(self, id) -> bool: logger.error("LoggingDM: start called"); return False
                async def delete_agent_instance(self, id) -> bool: logger.error("LoggingDM: delete called"); return False
            current_app_state.deployment_manager = LoggingDeploymentManager()


        # --- Initialize and Start Background Agents ---
        acq_criteria = {
            "niche": config.AGENT_TARGET_NICHE_DEFAULT,
            "clay_enrichment_table_webhook_url": config.CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY,
            "clay_enrichment_table_name": config.get_env_var("CLAY_ENRICHMENT_TABLE_NAME", default="Primary Lead Enrichment", required=False),
            "lead_source_type": config.ACQ_LEAD_SOURCE_TYPE,
            "lead_source_path": config.ACQ_LEAD_SOURCE_PATH,
            "lead_source_csv_field_mapping": config.ACQ_LEAD_SOURCE_CSV_MAPPING,
            "supabase_pending_lead_status": config.ACQ_SUPABASE_PENDING_LEAD_STATUS,
            "qualification_llm_score_threshold": config.ACQ_QUALIFICATION_THRESHOLD,
            "max_leads_to_process_per_cycle": config.ACQUISITION_AGENT_BATCH_SIZE,
            "run_interval_seconds": config.ACQUISITION_AGENT_RUN_INTERVAL_SECONDS,
            "batch_size": config.ACQUISITION_AGENT_BATCH_SIZE,
            "clay_service_automation_config": DEFAULT_CLAY_SERVICE_AUTOMATION_CONFIG # Pass the default config
        }
        if not acq_criteria["clay_enrichment_table_webhook_url"]:
            logger.warning("CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY not set. AcquisitionAgent will attempt discovery if needed.")
        
        current_app_state.acquisition_agent = AcquisitionAgent(
            agent_id="GlobalAcquisitionAgent_001",
            resource_manager=current_app_state.resource_manager,
            data_wrapper=current_app_state.data_wrapper,
            llm_client=current_app_state.llm_client,
            crm_wrapper=current_app_state.crm_wrapper,
            target_criteria=acq_criteria
        )
        await current_app_state.acquisition_agent.start()
        logger.info("AcquisitionAgent started in background.")

        # Initialize and start ProvisioningAgent
        current_app_state.provisioning_agent = ProvisioningAgent(
            agent_id="GlobalProvisioningAgent_001",
            resource_manager=current_app_state.resource_manager,
            crm_wrapper=current_app_state.crm_wrapper,
            llm_client=current_app_state.llm_client,
            deployment_manager=current_app_state.deployment_manager # Pass the initialized manager
        )
        await current_app_state.provisioning_agent.start()
        logger.info("ProvisioningAgent started in background.")


    except Exception as init_err:
        logger.critical(f"CRITICAL ERROR during application initialization: {init_err}", exc_info=True)
        # Depending on severity, you might want to exit or raise to prevent Uvicorn from starting
        # For now, it will raise, causing startup to fail, which is good for unrecoverable errors.
        raise RuntimeError(f"Failed to initialize core application components: {init_err}") from init_err

    # --- Application is Ready ---
    current_app_state.is_shutting_down = False
    logger.info("Boutique AI Server initialization complete. Ready for requests.")
    yield # Application runs here

    # --- Shutdown Logic ---
    logger.info("Boutique AI Server shutting down...")
    current_app_state.is_shutting_down = True

    if hasattr(current_app_state, 'acquisition_agent') and current_app_state.acquisition_agent:
        logger.info("Stopping AcquisitionAgent...")
        await current_app_state.acquisition_agent.stop()
    
    if hasattr(current_app_state, 'provisioning_agent') and current_app_state.provisioning_agent:
        logger.info("Stopping ProvisioningAgent...")
        await current_app_state.provisioning_agent.stop()

    active_calls = list(current_app_state.active_sales_agents.values())
    if active_calls:
        logger.info(f"Signaling end for {len(active_calls)} active sales calls...")
        stop_tasks = [agent.stop_call("Server Shutdown Signal") for agent in active_calls] # Pass reason
        try:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error during sales agent stop tasks: {e}", exc_info=True)
        logger.info("Waiting briefly for sales agent cleanup...")
        await asyncio.sleep(3) # Increased wait for graceful cleanup

    logger.info("Closing shared resources...")
    # Ensure DataWrapper.close_session is called
    if hasattr(DataWrapper, 'close_session'): # Check if method exists
        await DataWrapper.close_session()
        logger.info("DataWrapper session closed.")

    if hasattr(current_app_state, 'browser_automator') and current_app_state.browser_automator and \
       hasattr(current_app_state.browser_automator, 'close_session'):
        logger.info("Closing BrowserAutomator session...")
        await current_app_state.browser_automator.close_session()
    
    # Close Supabase client for setup if it was created
    if current_app_state.supabase_client_for_setup and hasattr(current_app_state.supabase_client_for_setup, 'close'):
        # Supabase client typically doesn't need explicit close unless it's holding persistent connections
        # httpx client used by supabase-py is managed within its requests
        pass

    logger.info("Boutique AI Server shutdown complete.")

# --- FastAPI App Instance ---
app = FastAPI(
    title="Boutique AI Server & Orchestrator",
    version="1.5.0", # Incremented version
    description="Autonomous AI Agent Factory & Sales Engine Backend",
    lifespan=lifespan # Use the new lifespan context manager
)

# --- Dependency Injection Helper ---
async def get_app_state() -> AppState:
    """Dependency injector to get the application state."""
    # Add robust check in case state is not AppState (e.g., during tests or misconfiguration)
    if not hasattr(app, "state") or not isinstance(app.state, AppState):
         logger.critical("Application state (app.state) is not initialized or not an AppState instance!")
         raise HTTPException(status_code=503, detail="Server critical state not ready. Please check logs.")
    return app.state

# --- Twilio Call Webhook ---
@app.post("/call_webhook", tags=["Twilio Webhooks"], response_class=PlainTextResponse)
async def handle_twilio_call_webhook(request: Request, state: AppState = Depends(get_app_state)):
    """Handles Twilio webhook for incoming/answered calls. Responds with TwiML <Stream>."""
    if not state.telephony_wrapper: # Should be initialized in lifespan
        logger.error("/call_webhook: Telephony service unavailable at request time.")
        raise HTTPException(status_code=503, detail="Telephony service temporarily unavailable.")

    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    call_status = form_data.get("CallStatus") # e.g., "initiated", "ringing", "in-progress", "completed", "failed"
    to_number = form_data.get("To")
    from_number = form_data.get("From") # Prospect's number for outbound, Twilio number for inbound
    logger.info(f"[{call_sid}] Received /call_webhook. Status: {call_status}, From: {from_number}, To: {to_number}")

    if not call_sid:
        logger.error("CallSid missing in Twilio webhook request.")
        raise HTTPException(status_code=400, detail="CallSid missing from Twilio request.")

    # Determine prospect's phone number. For outbound, 'To' is prospect. For inbound, 'From' is prospect.
    # This logic assumes TWILIO_PHONE_NUMBER is correctly set for your agency.
    prospect_phone_raw = to_number if from_number == config.TWILIO_PHONE_NUMBER else from_number
    
    if not prospect_phone_raw:
        logger.error(f"[{call_sid}] Could not determine prospect phone. From='{from_number}', To='{to_number}', ConfiguredTwilioNum='{config.TWILIO_PHONE_NUMBER}'")
        # Respond with Hangup if we can't process
        response_hangup = VoiceResponse(); response_hangup.hangup()
        return PlainTextResponse(str(response_hangup), media_type="application/xml")

    # Ensure prospect_phone is cleaned and E.164 for consistency before encoding
    # Basic cleaning: remove leading/trailing spaces. More robust cleaning might be needed.
    prospect_phone_cleaned = prospect_phone_raw.strip()
    # Add '+' if missing for E.164, assuming US numbers if no '+' (simplification)
    if not prospect_phone_cleaned.startswith('+') and prospect_phone_cleaned.isdigit() and len(prospect_phone_cleaned) >= 10:
        prospect_phone_cleaned = f"+1{prospect_phone_cleaned}" # Assuming US if no country code

    # Construct WebSocket URL for Twilio Media Stream
    # Prefer config.BASE_WEBHOOK_URL for host to ensure public accessibility
    base_url_parsed = urlparse(config.BASE_WEBHOOK_URL)
    ws_scheme = "wss" if base_url_parsed.scheme == "https" else "ws"
    ws_host = base_url_parsed.netloc # Use host from BASE_WEBHOOK_URL

    # URL-encode the prospect_phone for the query parameter
    encoded_prospect_phone = prospect_phone_cleaned.replace('+', '%2B') # Ensure + is encoded
    ws_path = f"/call_ws?call_sid={call_sid}&prospect_phone={encoded_prospect_phone}"
    ws_url_absolute = f"{ws_scheme}://{ws_host}{ws_path}"

    response = VoiceResponse()
    if call_status in ["ringing", "in-progress", "initiated"]: # Only start stream if call is active
        start_verb = Start()
        start_verb.stream(url=ws_url_absolute)
        response.append(start_verb)
        # response.pause(length=1) # Optional brief pause
        logger.info(f"[{call_sid}] Responding to Twilio with TwiML <Stream> to: {ws_url_absolute} for prospect {prospect_phone_cleaned}")
    elif call_status in ["completed", "failed", "busy", "no-answer"]:
        logger.info(f"[{call_sid}] Call ended with status: {call_status}. Not starting stream. Responding with Hangup or allowing Twilio default.")
        # Optionally log this termination if not already handled by SalesAgent ending
        # response.hangup() # Or let Twilio handle its flow for these states
    else: # Unknown status
        logger.warning(f"[{call_sid}] Unknown call status '{call_status}'. Not starting stream.")
        # response.hangup()

    return PlainTextResponse(str(response), media_type="application/xml")

# --- Twilio Media Stream WebSocket ---
@app.websocket("/call_ws")
async def websocket_call_endpoint(
    websocket: WebSocket,
    call_sid: str = Query(...),
    prospect_phone: str = Query(...) # Already URL decoded by FastAPI
):
    """Handles bidirectional audio stream and manages SalesAgent lifecycle for the call."""
    state: AppState = websocket.app.state # Correct way to get app state
    await websocket.accept()
    logger.info(f"[{call_sid}] WebSocket accepted for prospect: {prospect_phone}. Initializing SalesAgent.")

    if not all([state.llm_client, state.telephony_wrapper, state.crm_wrapper, state.data_wrapper, state.resource_manager]): # Checked RM
        logger.error(f"[{call_sid}] Core services not ready (LLM, Telephony, CRM, Data, RM). Closing WebSocket.")
        await websocket.close(code=1011, reason="Server critical services not ready"); return

    current_stream_sid: Optional[str] = None # To store the Twilio streamSid from 'start' event

    async def send_audio_to_twilio_ws_via_websocket(csid_local: str, audio_chunk: bytes):
        # Ensure this websocket instance is still active and matches call_sid
        if csid_local == call_sid and current_stream_sid and websocket.client_state == WebSocketState.CONNECTED:
            try:
                media_payload = base64.b64encode(audio_chunk).decode('utf-8')
                await websocket.send_json({"event": "media", "streamSid": current_stream_sid, "media": {"payload": media_payload}})
            except WebSocketDisconnect:
                logger.warning(f"[{call_sid}] WebSocket disconnected while trying to send audio.")
            except Exception as e:
                logger.warning(f"[{call_sid}] WS Error sending audio: {type(e).__name__} - {e}")

    async def send_mark_to_twilio_ws_via_websocket(csid_local: str, mark_name: str):
        if csid_local == call_sid and current_stream_sid and websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json({"event": "mark", "streamSid": current_stream_sid, "mark": {"name": mark_name}})
                logger.info(f"[{call_sid}] Sent mark '{mark_name}' to Twilio.")
            except WebSocketDisconnect:
                logger.warning(f"[{call_sid}] WebSocket disconnected while trying to send mark.")
            except Exception as e:
                logger.warning(f"[{call_sid}] WS Error sending mark: {type(e).__name__} - {e}")

    # Initialize VoiceHandler here, specific to this call/WebSocket session
    voice_handler = VoiceHandler(
        transcript_callback=None, # Will be set by SalesAgent if it uses this instance's methods
        error_callback=None       # Will be set by SalesAgent
    )

    # Fetch or create contact info
    # Ensure prospect_phone is in a consistent format for CRM query, e.g., E.164
    prospect_phone_e164 = prospect_phone # Assuming it's already cleaned by /call_webhook
    pre_call_contact_info = await state.crm_wrapper.get_contact_info(identifier=prospect_phone_e164, identifier_column="phone_number")
    if not pre_call_contact_info:
        logger.info(f"[{call_sid}] No existing contact for {prospect_phone_e164}. Creating shell contact.")
        shell_data = {"phone_number": prospect_phone_e164, "status": "IncomingCall_NewContact_WS", "source_info": f"Call_{call_sid}"}
        # Use upsert which handles creation and returns the record
        pre_call_contact_info = await state.crm_wrapper.upsert_contact(shell_data, "phone_number")
        if not pre_call_contact_info: # If upsert failed critically
            logger.error(f"[{call_sid}] Failed to create shell contact for {prospect_phone_e164}. Cannot proceed with SalesAgent.")
            await websocket.close(code=1011, reason="CRM contact creation failed"); return
    
    # Create SalesAgent instance
    sales_agent = SalesAgent(
        agent_id=f"SalesAgent_{call_sid}", target_phone_number=prospect_phone_e164,
        voice_handler=voice_handler, llm_client=state.llm_client,
        telephony_wrapper=state.telephony_wrapper, crm_wrapper=state.crm_wrapper,
        initial_prospect_data=pre_call_contact_info,
        send_audio_callback=send_audio_to_twilio_ws_via_websocket, # Pass the specific websocket sender
        send_mark_callback=send_mark_to_twilio_ws_via_websocket,
        # Callbacks for server to know agent's final state
        on_call_complete_callback=lambda aid, status, hist, summary_obj: logger.info(f"SERVER: Agent {aid} call ended. Status: {status}. Outcome: {summary_obj.get('final_call_outcome_category','N/A')}"),
        on_call_error_callback=lambda aid, err_msg: logger.error(f"SERVER: Agent {aid} reported call error: {err_msg}")
    )
    state.active_sales_agents[call_sid] = sales_agent

    try:
        while True: # Main loop for receiving messages from Twilio media stream
            if websocket.client_state != WebSocketState.CONNECTED:
                logger.warning(f"[{call_sid}] WebSocket no longer connected in receive loop.")
                sales_agent.signal_call_ended_externally("WebSocket Not Connected in Loop")
                break
            
            message = await websocket.receive_json() # Blocks until a message is received
            event = message.get("event")

            if event == "start":
                current_stream_sid = message.get("start", {}).get("streamSid")
                if not current_stream_sid:
                    logger.error(f"[{call_sid}] WS 'start' event missing streamSid: {message}")
                    await sales_agent._handle_fatal_error("Missing streamSid in Twilio 'start' event"); break
                logger.info(f"[{call_sid}] WS 'start' event received. Stream SID: {current_stream_sid}. Activating SalesAgent.")
                # Start the SalesAgent's main call logic in a background task
                asyncio.create_task(sales_agent.start_sales_call(call_sid, current_stream_sid))
            elif event == "media" and current_stream_sid:
                payload = message.get("media", {}).get("payload")
                if payload:
                    # Pass audio to SalesAgent, which then passes to VoiceHandler
                    await sales_agent.handle_incoming_audio(base64.b64decode(payload))
            elif event == "stop":
                logger.info(f"[{call_sid}] WS 'stop' event received from Twilio. Signaling agent to end.")
                sales_agent.signal_call_ended_externally("Twilio WebSocket Stop Event"); break # Exit loop
            elif event == "mark":
                mark_name = message.get('mark', {}).get('name')
                logger.debug(f"[{call_sid}] WS 'mark' event: {mark_name}")
                # SalesAgent's VoiceHandler might use this if it needs to react to its own marks
            elif event == "connected": # Initial confirmation, not the 'start' with streamSid
                logger.info(f"[{call_sid}] WS 'connected' event (protocol confirmation).")
            else:
                logger.debug(f"[{call_sid}] Unknown WS event type: {event}. Message: {message}")

    except WebSocketDisconnect as e:
        logger.warning(f"[{call_sid}] WebSocket disconnected by client/Twilio. Code: {e.code}, Reason: {e.reason}")
        if sales_agent.is_call_active:
            sales_agent.signal_call_ended_externally(f"WebSocket Disconnected (Code: {e.code})")
    except json.JSONDecodeError:
        logger.error(f"[{call_sid}] Invalid JSON received on WebSocket. Terminating connection.")
        if sales_agent.is_call_active:
            sales_agent.signal_call_ended_externally("WebSocket Invalid JSON")
    except Exception as e:
        logger.error(f"[{call_sid}] Error in WebSocket handler: {type(e).__name__} - {e}", exc_info=True)
        if sales_agent.is_call_active:
            sales_agent.signal_call_ended_externally(f"WebSocket Handler Exception: {type(e).__name__}")
    finally:
        logger.info(f"[{call_sid}] Cleaning up WebSocket & SalesAgent resources for prospect {prospect_phone}...")
        # Ensure agent cleanup is robustly called if not already handled by agent's own lifecycle
        if sales_agent.is_call_active: # If agent thinks it's still active but WS is closing
            logger.warning(f"[{call_sid}] WebSocket closing while agent {sales_agent.agent_id} still active. Forcing stop.")
            await sales_agent.stop_call("WebSocket Final Cleanup") # This will trigger its internal cleanup

        if call_sid in state.active_sales_agents:
            del state.active_sales_agents[call_sid]
            logger.info(f"[{call_sid}] SalesAgent removed from active agents dictionary.")
        
        # Ensure voice_handler is disconnected if it was connected by the agent
        if voice_handler.is_connected:
            await voice_handler.disconnect()
            logger.info(f"[{call_sid}] VoiceHandler disconnected during WebSocket cleanup.")

        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close(code=1000) # Graceful close
                logger.info(f"[{call_sid}] WebSocket connection explicitly closed.")
            except Exception as e_close:
                logger.warning(f"[{call_sid}] Error explicitly closing WebSocket: {e_close}")
        
        logger.info(f"[{call_sid}] WebSocket resources cleaned up from server perspective for {prospect_phone}.")


# --- Webhook for Clay.com Enrichment Results ---
@app.post("/webhooks/clay/enrichment_results", tags=["Webhooks"])
async def handle_clay_enrichment_webhook(
    request: Request,
    state: AppState = Depends(get_app_state),
    x_callback_auth_token: Optional[str] = Header(None, alias=config.CLAY_CALLBACK_AUTH_HEADER_NAME.lower()) # Ensure alias matches header standard (lowercase)
):
    """Receives enriched data from Clay, validates, and passes to AcquisitionAgent."""
    configured_secret = config.CLAY_RESULTS_CALLBACK_SECRET_TOKEN
    if configured_secret: # Only enforce auth if secret is configured
        if not x_callback_auth_token:
            logger.warning(f"Clay webhook: Missing auth token header '{config.CLAY_CALLBACK_AUTH_HEADER_NAME}'. Rejecting.")
            raise HTTPException(status_code=401, detail=f"Unauthorized: Missing {config.CLAY_CALLBACK_AUTH_HEADER_NAME} token")
        if x_callback_auth_token != configured_secret:
            logger.warning(f"Clay webhook: Invalid auth token provided. Rejecting.")
            raise HTTPException(status_code=403, detail="Forbidden: Invalid authentication token")
    elif not configured_secret and config.CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY: # Warn if using Clay but not securing callback
        logger.warning("Clay webhook: CLAY_RESULTS_CALLBACK_SECRET_TOKEN not configured in .env. This endpoint is INSECURE.")

    logger.info("Received POST on /webhooks/clay/enrichment_results")
    if not state.acquisition_agent:
        logger.error("Clay webhook: AcquisitionAgent not available! Cannot process enrichment.")
        raise HTTPException(status_code=503, detail="Acquisition service is temporarily unavailable.")

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        logger.error("Clay webhook: Invalid JSON payload received.")
        raise HTTPException(status_code=400, detail="Invalid JSON payload provided by Clay.")

    # Log summary of payload for debugging, avoid logging full sensitive data if large
    logger.debug(f"Clay webhook payload (summary): {json.dumps(payload)[:500]}" + ("..." if len(json.dumps(payload)) > 500 else ""))
    
    correlation_id = payload.get("_correlation_id") # Ensure this key is sent by Clay HTTP API step
    if not correlation_id:
        logger.warning("Clay webhook: Payload missing '_correlation_id'. Cannot process. Payload: %s", payload)
        # Return 400 Bad Request to Clay if correlation_id is essential
        return FastAPIResponse(content={"status": "error", "message": "Payload missing _correlation_id. Ensure your Clay HTTP API step includes it."}, status_code=400)

    # Offload processing to a background task to respond to Clay quickly
    asyncio.create_task(state.acquisition_agent.handle_clay_enrichment_result(payload))
    logger.info(f"Clay enrichment result for correlation_id '{correlation_id}' queued for asynchronous processing.")
    return FastAPIResponse(content={"status": "received", "message": "Result queued for processing."}, status_code=202) # Accepted

# --- Admin/Test Endpoint to Trigger Outbound Call ---
@app.post("/admin/actions/initiate_call", tags=["Admin Actions"])
async def trigger_outbound_call_admin(
    target_number: str = Query(..., description="E.164 formatted phone number to call."), # Make target_number required
    crm_contact_id: Optional[str] = Query(None, description="Optional CRM Contact ID (UUID) to associate with this call."),
    state: AppState = Depends(get_app_state)
):
    """Initiates an outbound call via Twilio for testing/admin purposes."""
    if not state.telephony_wrapper:
        logger.error("ADMIN initiate_call: Telephony service unavailable.")
        raise HTTPException(status_code=503, detail="Telephony service is currently unavailable.")
    if not target_number: # Already handled by Query(...) but good practice
        logger.error("ADMIN initiate_call: Missing 'target_number'.")
        raise HTTPException(status_code=400, detail="Required query parameter 'target_number' is missing.")
    
    logger.info(f"ADMIN: API request to initiate outbound call to: {target_number}. CRM Contact ID (if any): {crm_contact_id}")
    
    # Custom parameters for Twilio webhook. Twilio will prefix these with "Custom_".
    custom_params_for_twilio_webhook: Dict[str, str] = {}
    if crm_contact_id:
        custom_params_for_twilio_webhook["crm_contact_id"] = crm_contact_id
        # You could add more, e.g., "admin_initiated_by": "user_xyz"
    
    call_sid = await state.telephony_wrapper.initiate_call(
        target_number=target_number,
        custom_parameters=custom_params_for_twilio_webhook if custom_params_for_twilio_webhook else None
    )
    if call_sid:
        logger.info(f"ADMIN: Outbound call to {target_number} initiated successfully. Call SID: {call_sid}")
        return {"message": "Outbound call initiated successfully.", "call_sid": call_sid, "target_number": target_number}
    
    logger.error(f"ADMIN: Failed to initiate outbound call to {target_number} via telephony provider.")
    raise HTTPException(status_code=500, detail="Failed to initiate call via telephony provider. Check server logs.")

# --- Health Check Endpoint ---
@app.get("/health", tags=["System"])
async def health_check():
    """Basic health check endpoint. Add more checks as needed (e.g., DB connectivity)."""
    # Could add checks for Supabase, LLM client basic ping if available
    return {
        "status": "ok",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "service_version": app.version # Assuming app.version is set
    }

# --- Main Execution Block (for direct run, e.g. `python server.py`) ---
if __name__ == "__main__":
    import uvicorn
    # Log essential URLs for easy access during startup/debugging
    logger.info(f"--- Starting Boutique AI Server Orchestrator (Version: {app.version}) ---")
    logger.info(f"Host: 0.0.0.0, Port: {config.LOCAL_SERVER_PORT}, Reload: {config.UVICORN_RELOAD}")
    logger.info(f"Log Level: {config.LOG_LEVEL}")
    logger.info(f"Public Base URL for Webhooks: {config.BASE_WEBHOOK_URL}")
    if config.BASE_WEBHOOK_URL:
        logger.info(f"Expected Twilio Call Webhook URL: {config.BASE_WEBHOOK_URL.rstrip('/')}/call_webhook")
        logger.info(f"Expected Clay Enrichment Results Webhook URL: {config.BASE_WEBHOOK_URL.rstrip('/')}/webhooks/clay/enrichment_results")
    else:
        logger.critical("CRITICAL: BASE_WEBHOOK_URL is not set. Webhooks will fail!")
    
    logger.info(f"Supabase Enabled: {config.SUPABASE_ENABLED}")
    logger.info(f"Using REAL Browser Automator: {config.USE_REAL_BROWSER_AUTOMATOR}")
    if config.USE_REAL_BROWROWSER_AUTOMATOR:
        logger.info(f"Playwright Headful Mode: {config.PLAYWRIGHT_HEADFUL_MODE}")
    logger.info("--- Server Starting ---")
    
    uvicorn.run(
        "server:app", # app object in server.py
        host="0.0.0.0",
        port=config.LOCAL_SERVER_PORT,
        log_level=config.LOG_LEVEL.lower(), # Uvicorn log levels are lowercase
        reload=config.UVICORN_RELOAD,
        # Consider adding access_log=False if Uvicorn access logs are too noisy and you have other means
        # uvicorn.run already uses the root logger's configuration if log_config is None.
        # If you want uvicorn to use a specific dictConfig, you can pass it to log_config.
        # However, the logging setup in config.py should already cover application logs.
    )
# --------------------------------------------------------------------------------
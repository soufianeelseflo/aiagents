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
from datetime import datetime, timezone # Added for health check timestamp

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Response as FastAPIResponse, Header, Query, Depends
from fastapi.responses import PlainTextResponse
# from fastapi.routing import APIRoute # APIRoute not used, can remove if desired
from starlette.websockets import WebSocketState # Import for checking state

from twilio.twiml.voice_response import VoiceResponse, Start

import config # Root config - Ensure config.py loads correctly first!

# --- Graceful handling if core modules aren't ready ---
try:
    from core.services.llm_client import LLMClient
except ImportError as e: logger.error(f"Failed to import LLMClient: {e}"); LLMClient = None
try:
    from core.services.fingerprint_generator import FingerprintGenerator
except ImportError as e: logger.error(f"Failed to import FingerprintGenerator: {e}"); FingerprintGenerator = None
try:
    from core.services.telephony_wrapper import TelephonyWrapper
except ImportError as e: logger.error(f"Failed to import TelephonyWrapper: {e}"); TelephonyWrapper = None
try:
    from core.services.crm_wrapper import CRMWrapper
except ImportError as e: logger.error(f"Failed to import CRMWrapper: {e}"); CRMWrapper = None
try:
    from core.services.data_wrapper import DataWrapper, ClayWebhookError
except ImportError as e: logger.error(f"Failed to import DataWrapper: {e}"); DataWrapper = None; ClayWebhookError = Exception
try:
    from core.communication.voice_handler import VoiceHandler
except ImportError as e: logger.error(f"Failed to import VoiceHandler: {e}"); VoiceHandler = None
try:
    from core.agents.resource_manager import ResourceManager
except ImportError as e: logger.error(f"Failed to import ResourceManager: {e}"); ResourceManager = None
try:
    from core.agents.acquisition_agent import AcquisitionAgent, DEFAULT_CLAY_SERVICE_AUTOMATION_CONFIG
except ImportError as e: logger.error(f"Failed to import AcquisitionAgent: {e}"); AcquisitionAgent = None; DEFAULT_CLAY_SERVICE_AUTOMATION_CONFIG = {}
try:
    from core.agents.sales_agent import SalesAgent
except ImportError as e: logger.error(f"Failed to import SalesAgent: {e}"); SalesAgent = None
try:
    from core.automation.browser_automator_interface import BrowserAutomatorInterface
except ImportError as e: logger.error(f"Failed to import BrowserAutomatorInterface: {e}"); BrowserAutomatorInterface = None
try:
    # Decide which automator to use *before* lifespan
    use_real_automator = config.get_bool_env_var("USE_REAL_BROWSER_AUTOMATOR", default=False)
    if use_real_automator and BrowserAutomatorInterface:
        try:
            from core.automation.multimodal_playwright_automator import MultiModalPlaywrightAutomator
            AutomatorImplementation = MultiModalPlaywrightAutomator
            logger.info("Will attempt to use REAL MultiModalPlaywrightAutomator.")
        except ImportError as e:
            logger.error(f"Could not import MultiModalPlaywrightAutomator: {e}. Check if Playwright is installed via requirements.txt and Dockerfile.")
            logger.error("Falling back to MOCK automator.")
            AutomatorImplementation = None # Mark as unavailable
    else:
         AutomatorImplementation = None # Use Mock or indicate none if mock isn't defined/needed
         if use_real_automator:
             logger.warning("USE_REAL_BROWSER_AUTOMATOR is true, but BrowserAutomatorInterface or implementation failed import.")
         else:
              logger.warning("USE_REAL_BROWSER_AUTOMATOR is false. No real browser automation will be used.")

except ImportError as e: # Catch general import error for automator section
     logger.error(f"Error importing Browser Automator: {e}")
     AutomatorImplementation = None


# Only import DB setup if Supabase is potentially enabled
if config.SUPABASE_ENABLED:
    try:
        from core.database_setup import setup_supabase_tables, create_client as create_supabase_client_for_setup
    except ImportError as e:
        logger.error(f"Failed to import database_setup: {e}. Supabase setup will be skipped.")
        setup_supabase_tables = None
        create_supabase_client_for_setup = None
else:
    setup_supabase_tables = None
    create_supabase_client_for_setup = None

# --- Define Mock Browser Automator Directly Here if needed fallback ---
# This avoids needing a separate mock file if the real one fails import
# But only define it if the real one couldn't be imported
if AutomatorImplementation is None and BrowserAutomatorInterface:
    logger.info("Defining MockBrowserAutomator directly in server.py as fallback.")
    class MockBrowserAutomator(BrowserAutomatorInterface):
        def __init__(self, llm_client: Optional[LLMClient] = None): logger.info("MockBrowserAutomator initialized.")
        async def setup_session(self, *args, **kwargs): logger.info("MockAutomator: setup_session"); return True
        async def close_session(self, *args, **kwargs): logger.info("MockAutomator: close_session")
        async def navigate_to_page(self, url: str, *args, **kwargs): logger.info(f"MockAutomator: navigate_to_page {url}"); return True
        async def take_screenshot(self, *args, **kwargs): logger.info("MockAutomator: take_screenshot"); return b"mock_screenshot_bytes"
        async def fill_form_and_submit(self, *args, **kwargs): logger.warning("MockAutomator: fill_form_and_submit called (deprecated)"); return True # Deprecated form
        async def _execute_interaction_plan(self, plan: List[Dict[str, Any]], details: Dict[str, Any]) -> bool: logger.info(f"MockAutomator: execute_interaction_plan with {len(plan)} steps."); return True # Add if needed by RM test/logic
        async def check_success_condition(self, *args, **kwargs): logger.info("MockAutomator: check_success_condition"); return True
        async def extract_resources_from_page(self, *args, **kwargs): logger.info("MockAutomator: extract_resources_from_page"); return {"mock_resource": "mock_value"}
        async def solve_captcha_if_present(self, *args, **kwargs): logger.info("MockAutomator: solve_captcha_if_present (assuming none)"); return True
        async def get_cookies(self) -> Optional[List[Dict[str, Any]]]: logger.info("MockAutomator: get_cookies"); return []
        async def full_signup_and_extract(self, *args, **kwargs) -> Dict[str, Any]: logger.info("MockAutomator: full_signup_and_extract"); return {"status": "success", "extracted_resources": {"mock_api_key": "mock_key_123"}, "cookies": []}
    AutomatorImplementation = MockBrowserAutomator # Assign the mock as the implementation to use


logger = logging.getLogger(__name__) # Use standard logger name

# --- Application State ---
# Using Optional for components that might fail initialization
class AppState:
    llm_client: Optional[LLMClient]
    fingerprint_generator: Optional[FingerprintGenerator]
    browser_automator: Optional[BrowserAutomatorInterface]
    resource_manager: Optional[ResourceManager]
    data_wrapper: Optional[DataWrapper]
    crm_wrapper: Optional[CRMWrapper]
    telephony_wrapper: Optional[TelephonyWrapper]
    acquisition_agent: Optional[AcquisitionAgent]
    # SalesAgent instances are created per call
    active_sales_agents: Dict[str, SalesAgent] = {}
    supabase_client_for_setup: Optional[Any] = None
    is_shutting_down: bool = False

# Global state object (alternatively, manage within FastAPI app state)
# Using app.state is generally preferred in FastAPI
# app_state = AppState() # Remove this global instance

# --- FastAPI Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    logger.info("Boutique AI Server starting lifespan...")
    # Create state instance and attach to app
    app.state = AppState()
    app.state.is_shutting_down = False

    # --- Database Setup (Only if enabled and imported) ---
    if config.SUPABASE_ENABLED and setup_supabase_tables and create_supabase_client_for_setup:
        logger.info("Attempting Supabase database schema setup/verification...")
        try:
            # Use service role key for setup
            # Note: The create_client in database_setup.py already uses config
            app.state.supabase_client_for_setup = create_supabase_client_for_setup()
            if app.state.supabase_client_for_setup:
                await setup_supabase_tables(app.state.supabase_client_for_setup)
                logger.info("Supabase database schema setup/verification potentially complete (check logs from database_setup).")
            else:
                # This case should be handled by create_client logging an error
                logger.error("Failed to create Supabase client for database setup (check config.py logs).")
        except Exception as db_setup_e:
            logger.critical(f"Error during Supabase database setup call: {db_setup_e}", exc_info=True)
    elif config.SUPABASE_ENABLED:
         logger.warning("Supabase is enabled in config, but setup functions failed to import. Skipping DB setup.")
    else:
        logger.warning("Supabase is not enabled. Skipping database table setup.")

    # --- Initialize Shared Dependencies ---
    # Using try-except blocks for each critical component initialization
    try:
        app.state.llm_client = LLMClient() if LLMClient else None
        if not app.state.llm_client: raise RuntimeError("LLMClient failed to initialize or import.")
    except Exception as e: logger.critical(f"Failed to initialize LLMClient: {e}", exc_info=True); app.state.llm_client = None

    try:
        app.state.fingerprint_generator = FingerprintGenerator(llm_client=app.state.llm_client) if FingerprintGenerator else None
        if not app.state.fingerprint_generator: raise RuntimeError("FingerprintGenerator failed to initialize or import.")
    except Exception as e: logger.critical(f"Failed to initialize FingerprintGenerator: {e}", exc_info=True); app.state.fingerprint_generator = None

    # --- Instantiate Browser Automator ---
    app.state.browser_automator = None # Initialize as None
    if AutomatorImplementation and app.state.llm_client: # Check if implementation class is available and LLM client ready
         try:
             app.state.browser_automator = AutomatorImplementation(
                 llm_client=app.state.llm_client,
                 headless=not config.PLAYWRIGHT_HEADFUL_MODE # Use config var
             )
             logger.info(f"Lifespan Startup: Initialized Browser Automator: {AutomatorImplementation.__name__}")
         except Exception as auto_err:
              logger.critical(f"Error initializing {AutomatorImplementation.__name__}: {auto_err}. Browser automation disabled.", exc_info=True)
              # No fallback here, just leave it as None
    elif not AutomatorImplementation:
         logger.warning("No Browser Automator implementation available (import failed or disabled).")
    elif not app.state.llm_client:
         logger.warning("LLMClient not available, cannot initialize Browser Automator requiring it.")


    # --- Initialize Services that depend on others ---
    # Ensure required components are available before initializing dependent ones
    try:
        if app.state.llm_client and app.state.fingerprint_generator and app.state.browser_automator and ResourceManager:
            app.state.resource_manager = ResourceManager(
                llm_client=app.state.llm_client,
                fingerprint_generator=app.state.fingerprint_generator,
                browser_automator=app.state.browser_automator
            )
        elif ResourceManager:
            # Log which specific dependency is missing
            missing_deps = [name for name, obj in [
                ("LLMClient", app.state.llm_client),
                ("FingerprintGenerator", app.state.fingerprint_generator),
                ("BrowserAutomator", app.state.browser_automator)
            ] if not obj]
            logger.critical(f"Cannot initialize ResourceManager: Missing dependencies: {', '.join(missing_deps)}")
            app.state.resource_manager = None
        else: app.state.resource_manager = None # ResourceManager itself failed import
    except Exception as e: logger.critical(f"Failed to initialize ResourceManager: {e}", exc_info=True); app.state.resource_manager = None

    try: app.state.data_wrapper = DataWrapper() if DataWrapper else None
    except Exception as e: logger.critical(f"Failed to initialize DataWrapper: {e}", exc_info=True); app.state.data_wrapper = None

    try: app.state.crm_wrapper = CRMWrapper() if CRMWrapper else None
    except Exception as e: logger.critical(f"Failed to initialize CRMWrapper: {e}", exc_info=True); app.state.crm_wrapper = None

    try: app.state.telephony_wrapper = TelephonyWrapper() if TelephonyWrapper else None
    except Exception as e: logger.critical(f"Failed to initialize TelephonyWrapper: {e}", exc_info=True); app.state.telephony_wrapper = None


    # --- Initialize and Start AcquisitionAgent ---
    # Check dependencies required by AcquisitionAgent
    if AcquisitionAgent and app.state.resource_manager and app.state.data_wrapper and app.state.llm_client and app.state.crm_wrapper:
        try:
            # Construct criteria from config
            acq_criteria = {
                "niche": config.AGENT_TARGET_NICHE_DEFAULT,
                "clay_enrichment_table_webhook_url": config.CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY,
                "clay_enrichment_table_name": config.get_env_var("CLAY_ENRICHMENT_TABLE_NAME", required=False), # Fetch potentially missing var
                "lead_source_type": config.ACQ_LEAD_SOURCE_TYPE,
                "lead_source_path": config.ACQ_LEAD_SOURCE_PATH,
                "lead_source_csv_field_mapping": config.ACQ_LEAD_SOURCE_CSV_MAPPING,
                "supabase_pending_lead_status": config.ACQ_SUPABASE_PENDING_LEAD_STATUS,
                "qualification_llm_score_threshold": config.ACQ_QUALIFICATION_THRESHOLD,
                "max_leads_to_process_per_cycle": config.ACQUISITION_AGENT_BATCH_SIZE,
                "run_interval_seconds": config.ACQUISITION_AGENT_RUN_INTERVAL_SECONDS,
                "batch_size": config.ACQUISITION_AGENT_BATCH_SIZE,
                "clay_service_automation_config": DEFAULT_CLAY_SERVICE_AUTOMATION_CONFIG
            }
            # Check required webhook URL (can be discovered later, but log if missing now)
            if not acq_criteria["clay_enrichment_table_webhook_url"]:
                logger.warning("CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY not set. AcquisitionAgent will attempt discovery if table name is set.")

            app.state.acquisition_agent = AcquisitionAgent(
                agent_id="GlobalAcquisitionAgent_001",
                resource_manager=app.state.resource_manager,
                data_wrapper=app.state.data_wrapper,
                llm_client=app.state.llm_client,
                crm_wrapper=app.state.crm_wrapper,
                target_criteria=acq_criteria
            )
            await app.state.acquisition_agent.start()
            logger.info("AcquisitionAgent started successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize or start AcquisitionAgent: {e}", exc_info=True)
            app.state.acquisition_agent = None
    else:
        missing_deps = [name for name, obj in [
                ("AcquisitionAgent Class", AcquisitionAgent),
                ("ResourceManager", app.state.resource_manager),
                ("DataWrapper", app.state.data_wrapper),
                ("LLMClient", app.state.llm_client),
                ("CRMWrapper", app.state.crm_wrapper)
            ] if not obj]
        logger.critical(f"Cannot initialize AcquisitionAgent: Missing dependencies: {', '.join(missing_deps)}")
        app.state.acquisition_agent = None

    # --- Log final status of initializations ---
    initialized_ok = all([
        app.state.llm_client, app.state.fingerprint_generator,
        # browser_automator is optional depending on config
        app.state.resource_manager, app.state.data_wrapper, app.state.crm_wrapper,
        app.state.telephony_wrapper,
        # AcquisitionAgent initialization logged above
    ])

    if initialized_ok:
        logger.info("Boutique AI Server initialization complete. Ready for requests.")
    else:
        logger.critical("Boutique AI Server initialization INCOMPLETE. Some core components failed. Check critical logs above.")
        # Consider raising an exception here to prevent FastAPI from fully starting if critical components failed
        # raise RuntimeError("Core component initialization failed.")

    yield # Application runs here

    # --- Shutdown Logic ---
    logger.info("Boutique AI Server shutting down...")
    app.state.is_shutting_down = True

    # Gracefully stop background agents/tasks
    if hasattr(app.state, 'acquisition_agent') and app.state.acquisition_agent and app.state.acquisition_agent._is_running:
        logger.info("Stopping AcquisitionAgent...")
        await app.state.acquisition_agent.stop()

    # Handle active calls
    active_calls = list(app.state.active_sales_agents.values())
    if active_calls:
        logger.info(f"Signaling end for {len(active_calls)} active sales calls...")
        stop_tasks = [agent.stop_call("Server Shutdown") for agent in active_calls if hasattr(agent, 'stop_call')]
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
            await asyncio.sleep(2) # Brief wait for cleanup signals

    # Close shared resources only if they were initialized
    logger.info("Closing shared resources...")
    if hasattr(app.state, 'data_wrapper') and app.state.data_wrapper and hasattr(DataWrapper, 'close_session'):
        await DataWrapper.close_session() # Assumes class method close_session exists
    if hasattr(app.state, 'browser_automator') and app.state.browser_automator and hasattr(app.state.browser_automator, 'close_session'):
        await app.state.browser_automator.close_session()

    logger.info("Boutique AI Server shutdown complete.")

# --- FastAPI App Instance ---
# Pass the lifespan context manager to the FastAPI app
app = FastAPI(
    title="Boutique AI Server & Orchestrator",
    version="1.5.0", # Incremented version
    lifespan=lifespan
)

# --- Dependency Injection Helper ---
# Use Depends(lambda: app.state) to inject state into route functions
async def get_app_state() -> AppState:
    """Dependency injector to get the application state from the FastAPI app instance."""
    if not hasattr(app, "state") or not isinstance(app.state, AppState):
         # This might happen if a request comes in before lifespan setup is complete or after shutdown
         logger.critical("App state not available or not initialized correctly!")
         raise HTTPException(status_code=503, detail="Server is starting up or shutting down")
    return app.state

# --- Twilio Call Webhook ---
# Ensure core components like TelephonyWrapper are checked
@app.post("/call_webhook", tags=["Twilio"], response_class=PlainTextResponse)
async def handle_twilio_call_webhook(request: Request, state: AppState = Depends(get_app_state)):
    """Handles Twilio webhook for incoming/answered calls. Responds with TwiML <Stream>."""
    # Check required dependencies at the start of the request
    if not state.telephony_wrapper:
        logger.error("/call_webhook: Telephony service unavailable.")
        raise HTTPException(status_code=503, detail="Telephony service unavailable")
    if not config.BASE_WEBHOOK_URL:
         logger.error("/call_webhook: BASE_WEBHOOK_URL is not configured.")
         raise HTTPException(status_code=500, detail="Server configuration error (webhook base URL)")

    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    call_status = form_data.get("CallStatus")
    to_number = form_data.get("To")
    from_number = form_data.get("From")
    logger.info(f"[{call_sid}] Received /call_webhook. Status: {call_status}, From: {from_number}, To: {to_number}")

    if not call_sid:
        logger.error("/call_webhook: CallSid missing from Twilio request.")
        raise HTTPException(status_code=400, detail="CallSid missing")

    # Determine prospect number (non-Twilio number)
    prospect_phone = from_number if to_number == config.TWILIO_PHONE_NUMBER else to_number
    if not prospect_phone:
        logger.error(f"[{call_sid}] Could not determine prospect phone. From={from_number}, To={to_number}, TwilioNum={config.TWILIO_PHONE_NUMBER}")
        # Return TwiML Hangup if we can't identify prospect? Or raise error?
        response = VoiceResponse(); response.hangup()
        return PlainTextResponse(str(response), media_type="application/xml")
        # raise HTTPException(status_code=400, detail="Could not determine prospect phone")

    # Construct WebSocket URL carefully
    # Use request.url components for host/scheme detection if behind proxy
    ws_scheme = "wss" if request.url.scheme == "https" or config.BASE_WEBHOOK_URL.startswith("https://") else "ws"
    # Use request headers for host if available, otherwise fallback to base URL host
    host = request.headers.get("host")
    if not host:
        try: host = config.BASE_WEBHOOK_URL.split("://")[1]
        except IndexError: host = f"localhost:{config.LOCAL_SERVER_PORT}" # Final fallback

    # URL encode parameters, especially '+' in phone numbers
    encoded_prospect_phone = requests.utils.quote(prospect_phone) # Requires 'requests' library
    encoded_call_sid = requests.utils.quote(call_sid)
    ws_path = f"/call_ws?call_sid={encoded_call_sid}&prospect_phone={encoded_prospect_phone}"
    ws_url_absolute = f"{ws_scheme}://{host}{ws_path}"

    # Generate TwiML response
    response = VoiceResponse()
    start_verb = Start()
    start_verb.stream(url=ws_url_absolute)
    response.append(start_verb)
    # Removed pause - Deepgram connects quickly, pausing might delay agent start
    # response.pause(length=1) # Consider removing unless needed for specific timing
    logger.info(f"[{call_sid}] Responding to Twilio with TwiML <Stream> to: {ws_url_absolute}")
    return PlainTextResponse(str(response), media_type="application/xml")

# --- Twilio Media Stream WebSocket ---
@app.websocket("/call_ws")
async def websocket_call_endpoint(
    websocket: WebSocket,
    call_sid: str = Query(...),
    prospect_phone: str = Query(...) # Assumes it's already URL-decoded by FastAPI
):
    """Handles bidirectional audio stream and manages SalesAgent lifecycle for the call."""
    # Get state directly from websocket app instance
    state: AppState = websocket.app.state
    await websocket.accept()
    logger.info(f"[{call_sid}] WebSocket accepted for prospect: {prospect_phone}. Initializing SalesAgent.")

    # --- Dependency Checks ---
    # Check all components needed for SalesAgent operation RIGHT before creating it
    if not all([state.llm_client, state.telephony_wrapper, state.crm_wrapper, VoiceHandler]):
        missing = [name for name, obj in [
            ("LLMClient", state.llm_client), ("TelephonyWrapper", state.telephony_wrapper),
            ("CRMWrapper", state.crm_wrapper), ("VoiceHandler Class", VoiceHandler)
        ] if not obj]
        reason = f"Server dependencies not ready: {', '.join(missing)}"
        logger.error(f"[{call_sid}] Cannot start SalesAgent. {reason}. Closing WebSocket.")
        await websocket.close(code=1011, reason=reason); return

    # Check if SalesAgent class itself imported correctly
    if not SalesAgent:
        reason = "SalesAgent class failed to import."
        logger.error(f"[{call_sid}] Cannot start SalesAgent. {reason}. Closing WebSocket.")
        await websocket.close(code=1011, reason=reason); return

    current_stream_sid: Optional[str] = None
    sales_agent_instance: Optional[SalesAgent] = None # Keep track of the instance

    # --- Nested Async Callbacks for SalesAgent ---
    async def send_audio_to_twilio_ws(csid: str, audio_chunk: bytes):
        # Check websocket state before sending
        if csid == call_sid and current_stream_sid and websocket.client_state == WebSocketState.CONNECTED:
            try:
                media_payload = base64.b64encode(audio_chunk).decode('utf-8')
                await websocket.send_json({"event": "media", "streamSid": current_stream_sid, "media": {"payload": media_payload}})
            except WebSocketDisconnect: logger.warning(f"[{call_sid}] WS disconnected while trying to send audio.")
            except Exception as e: logger.warning(f"[{call_sid}] WS Error sending audio: {type(e).__name__}")

    async def send_mark_to_twilio_ws(csid: str, mark_name: str):
         if csid == call_sid and current_stream_sid and websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json({"event": "mark", "streamSid": current_stream_sid, "mark": {"name": mark_name}})
                logger.info(f"[{call_sid}] Sent mark '{mark_name}' to Twilio.")
            except WebSocketDisconnect: logger.warning(f"[{call_sid}] WS disconnected while trying to send mark.")
            except Exception as e: logger.warning(f"[{call_sid}] WS Error sending mark: {type(e).__name__}")

    # --- Voice Handler and Sales Agent Initialization ---
    try:
        # Create voice handler instance per call
        voice_handler = VoiceHandler(transcript_callback=None, error_callback=None) # Callbacks will be set by SalesAgent

        # Fetch contact info (ensure CRM wrapper is available)
        pre_call_contact_info = None
        if state.crm_wrapper:
            pre_call_contact_info = await state.crm_wrapper.get_contact_info(identifier=prospect_phone, identifier_column="phone_number")
            if not pre_call_contact_info:
                logger.info(f"[{call_sid}] No existing contact for {prospect_phone}. Creating shell.")
                shell_data = {"phone_number": prospect_phone, "status": "IncomingCall_NewContact", "source_info": f"Call_{call_sid}"}
                # Need to make upsert return the created data or fetch again
                upserted_data = await state.crm_wrapper.upsert_contact(shell_data, "phone_number")
                pre_call_contact_info = upserted_data if upserted_data else shell_data # Use upserted if possible
        else:
             logger.warning(f"[{call_sid}] CRMWrapper not available. Cannot fetch/create contact info.")
             pre_call_contact_info = {"phone_number": prospect_phone, "status": "Unknown_NoCRM"}

        # Create SalesAgent instance
        sales_agent_instance = SalesAgent(
            agent_id=f"SalesAgent_{call_sid}", target_phone_number=prospect_phone,
            voice_handler=voice_handler, llm_client=state.llm_client,
            telephony_wrapper=state.telephony_wrapper, crm_wrapper=state.crm_wrapper,
            initial_prospect_data=pre_call_contact_info,
            send_audio_callback=send_audio_to_twilio_ws,
            send_mark_callback=send_mark_to_twilio_ws,
            # Simplified callbacks for logging within the server context
            on_call_complete_callback=lambda aid, status, hist, summary: logger.info(f"SERVER WS: Agent {aid} call complete. Status: {status}. Outcome: {summary.get('final_call_outcome_category','N/A')}"),
            on_call_error_callback=lambda aid, err: logger.error(f"SERVER WS: Agent {aid} call error: {err}")
        )
        state.active_sales_agents[call_sid] = sales_agent_instance # Track active agent

    except Exception as agent_init_e:
         logger.critical(f"[{call_sid}] Failed to initialize SalesAgent or dependencies: {agent_init_e}", exc_info=True)
         await websocket.close(code=1011, reason=f"Agent initialization failed: {agent_init_e}")
         return # Exit the handler

    # --- WebSocket Message Loop ---
    try:
        while True:
            message = await websocket.receive_json()
            event = message.get("event")

            if event == "start":
                stream_sid_from_msg = message.get("start", {}).get("streamSid")
                if not stream_sid_from_msg:
                    logger.error(f"[{call_sid}] WS 'start' event missing streamSid: {message}")
                    if sales_agent_instance: await sales_agent_instance._handle_fatal_error("Missing streamSid in start event")
                    break
                current_stream_sid = stream_sid_from_msg
                logger.info(f"[{call_sid}] WS 'start' event received. Stream SID: {current_stream_sid}. Starting SalesAgent call logic.")
                # Start agent processing in background task
                if sales_agent_instance:
                    asyncio.create_task(sales_agent_instance.start_sales_call(call_sid, current_stream_sid))
                else: logger.error(f"[{call_sid}] Received start event, but SalesAgent instance is None!")

            elif event == "media" and current_stream_sid:
                payload = message.get("media", {}).get("payload")
                if payload and sales_agent_instance:
                    # Pass audio chunk to the specific agent instance
                    await sales_agent_instance.handle_incoming_audio(base64.b64decode(payload))

            elif event == "stop":
                logger.info(f"[{call_sid}] WS 'stop' event received. Signaling agent call end.")
                if sales_agent_instance:
                    # Use the agent's method to handle external stop signals
                    sales_agent_instance.signal_call_ended_externally("WebSocket Stop Event")
                break # Exit the loop

            elif event == "mark": logger.debug(f"[{call_sid}] WS 'mark': {message.get('mark', {}).get('name')}")
            elif event == "connected": logger.info(f"[{call_sid}] WS 'connected' event (should follow accept).")
            else: logger.warning(f"[{call_sid}] Received unknown WS event: {event}. Message: {message}")

    except WebSocketDisconnect as e:
        logger.warning(f"[{call_sid}] WebSocket disconnected unexpectedly. Code: {e.code}, Reason: {e.reason}")
        # Ensure agent is signaled if disconnection happens while call was active
        if sales_agent_instance and sales_agent_instance.is_call_active:
            sales_agent_instance.signal_call_ended_externally(f"WS Disconnected (Code: {e.code})")
    except Exception as e:
        logger.error(f"[{call_sid}] Error in WebSocket handler loop: {type(e).__name__} - {e}", exc_info=True)
        # Ensure agent is signaled on unexpected errors
        if sales_agent_instance and sales_agent_instance.is_call_active:
            sales_agent_instance.signal_call_ended_externally(f"WS Handler Exception: {type(e).__name__}")
    finally:
        logger.info(f"[{call_sid}] Cleaning up WebSocket handler for {prospect_phone}...")
        # Agent cleanup (like DB logging, resource release) is handled internally by the agent's stop/cleanup logic
        # Remove agent from tracking dict
        if call_sid in state.active_sales_agents:
            del state.active_sales_agents[call_sid]
            logger.info(f"[{call_sid}] Removed SalesAgent instance from active tracking.")
        # Ensure websocket is closed if not already
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close(code=1000, reason="Handler cleanup")
        logger.info(f"[{call_sid}] WebSocket handler cleanup finished.")


# --- Webhook for Clay.com Enrichment Results ---
@app.post("/webhooks/clay/enrichment_results", tags=["Webhooks"])
async def handle_clay_enrichment_webhook(
    request: Request,
    state: AppState = Depends(get_app_state),
    # Use Header dependency for auth token
    x_callback_auth_token: Optional[str] = Header(None, alias=config.CLAY_CALLBACK_AUTH_HEADER_NAME)
):
    """Receives enriched data from Clay, validates, and passes to AcquisitionAgent."""
    # --- Authentication ---
    configured_secret = config.CLAY_RESULTS_CALLBACK_SECRET_TOKEN
    if configured_secret:
        if not x_callback_auth_token:
            logger.warning("Clay webhook rejected: Auth token header missing.")
            raise HTTPException(status_code=401, detail="Unauthorized: Missing authentication token")
        if x_callback_auth_token != configured_secret:
            logger.warning(f"Clay webhook rejected: Invalid auth token received.")
            raise HTTPException(status_code=403, detail="Forbidden: Invalid authentication token")
        logger.debug("Clay webhook authentication successful.")
    else:
        # Security Warning if no secret is configured
        logger.warning("Clay results webhook running WITHOUT authentication (CLAY_RESULTS_CALLBACK_SECRET_TOKEN not set). Endpoint is INSECURE.")

    logger.info("Received POST on /webhooks/clay/enrichment_results")
    # Check if AcquisitionAgent is initialized
    if not state.acquisition_agent:
        logger.error("Clay webhook error: AcquisitionAgent not available!")
        raise HTTPException(status_code=503, detail="Acquisition service unavailable")

    # --- Payload Processing ---
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        logger.error("Clay webhook received invalid JSON payload.")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    logger.debug(f"Clay webhook payload received (first 500 chars): {str(payload)[:500]}")

    # Check for Correlation ID (essential for matching)
    correlation_id = payload.get("_correlation_id")
    if not correlation_id:
        logger.warning("Clay webhook payload missing '_correlation_id'. Cannot process. Payload: %s", payload)
        # Return 400 Bad Request
        return FastAPIResponse(content={"status": "error", "message": "Payload missing _correlation_id"}, status_code=400)

    # --- Hand off to Agent (Background Task) ---
    # Use asyncio.create_task for non-blocking processing
    asyncio.create_task(state.acquisition_agent.handle_clay_enrichment_result(payload))

    logger.info(f"Clay enrichment result for correlation_id '{correlation_id}' queued for background processing.")
    # Respond immediately with 202 Accepted
    return FastAPIResponse(content={"status": "received", "message": "Result queued for processing."}, status_code=202)

# --- Admin/Test Endpoint to Trigger Outbound Call ---
@app.post("/admin/actions/initiate_call", tags=["Admin Actions"])
async def trigger_outbound_call_admin(
    target_number: str = Query(..., description="E.164 formatted number to call."), # Make target_number required
    crm_contact_id: Optional[str] = Query(None, description="Optional CRM ID to associate for context."),
    state: AppState = Depends(get_app_state)
):
    """Initiates an outbound call via Twilio for testing/admin purposes."""
    if not state.telephony_wrapper:
        logger.error("Admin Call Trigger: Telephony service unavailable.")
        raise HTTPException(status_code=503, detail="Telephony service unavailable.")
    # Basic validation for target_number format (optional but recommended)
    if not target_number.startswith('+') or not target_number[1:].isdigit() or len(target_number) < 10:
         logger.warning(f"Admin Call Trigger: Invalid target_number format: {target_number}")
         raise HTTPException(status_code=400, detail="Invalid 'target_number' format. Must be E.164 (e.g., +15551234567).")

    logger.info(f"ADMIN: API request to initiate call to: {target_number}. CRM Contact ID: {crm_contact_id}")

    # Prepare custom parameters for Twilio if needed (optional)
    custom_params_for_twilio = {}
    if crm_contact_id:
        # Example: Pass CRM ID to webhook (Twilio prefixes custom params with 'Custom_')
        # custom_params_for_twilio["crm_id"] = crm_contact_id
        # Note: Current /call_webhook doesn't explicitly handle these, but they'd be in request form data
        logger.debug(f"Passing CRM ID {crm_contact_id} conceptually (not explicitly handled by current webhook).")

    try:
        call_sid = await state.telephony_wrapper.initiate_call(
            target_number=target_number,
            # custom_parameters=custom_params_for_twilio if custom_params_for_twilio else None # Uncomment if needed
        )
        if call_sid:
            logger.info(f"Admin Call Trigger: Outbound call initiated successfully. Call SID: {call_sid}")
            return {"message": "Outbound call initiated.", "call_sid": call_sid}
        else:
            # Initiate_call should log specifics, return generic server error
            logger.error(f"Admin Call Trigger: Failed to initiate call to {target_number} via telephony provider (see TelephonyWrapper logs).")
            raise HTTPException(status_code=500, detail="Failed to initiate call via telephony provider.")
    except Exception as e:
         logger.error(f"Admin Call Trigger: Unexpected error initiating call: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Unexpected server error initiating call: {type(e).__name__}")


# --- Health Check Endpoint ---
@app.get("/health", tags=["System"])
async def health_check(state: AppState = Depends(get_app_state)):
    """Basic health check endpoint, includes status of key components."""
    status_report = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "llm_client": "ok" if state.llm_client else "error",
            "telephony": "ok" if state.telephony_wrapper else "error",
            "crm": "ok" if state.crm_wrapper else "unavailable/error",
            "data_wrapper": "ok" if state.data_wrapper else "error",
            "resource_manager": "ok" if state.resource_manager else "error",
            "browser_automator": "ok" if state.browser_automator else "unavailable/error",
            "acquisition_agent": "running" if state.acquisition_agent and state.acquisition_agent._is_running else "stopped/error",
        }
    }
    # Determine overall status based on critical components
    critical_components_ok = all([state.llm_client, state.telephony_wrapper, state.crm_wrapper])
    if not critical_components_ok:
        status_report["status"] = "degraded"
        return FastAPIResponse(content=status_report, status_code=503) # Service Unavailable

    return status_report

# --- Main Execution Block (for local testing) ---
# This block is typically NOT run when deployed via Docker CMD/ENTRYPOINT
if __name__ == "__main__":
    import uvicorn
    # Log essential info needed for setup/debugging
    logger.info(f"--- Starting Boutique AI Server (Local Development Mode) ---")
    logger.info(f"Log Level: {config.LOG_LEVEL}")
    logger.info(f"Uvicorn Reload: {config.UVICORN_RELOAD}")
    logger.info(f"Listening on: 0.0.0.0:{config.LOCAL_SERVER_PORT}")
    if config.BASE_WEBHOOK_URL:
        logger.info(f"Expected Base URL for Webhooks: {config.BASE_WEBHOOK_URL}")
        logger.info(f"-> Twilio Call Webhook URL: {config.BASE_WEBHOOK_URL.rstrip('/')}/call_webhook")
        logger.info(f"-> Clay Results Webhook URL: {config.BASE_WEBHOOK_URL.rstrip('/')}/webhooks/clay/enrichment_results")
    else:
        logger.critical("CRITICAL: BASE_WEBHOOK_URL is not set in .env. Webhooks will fail.")
    logger.info("---")

    uvicorn.run(
        "server:app", # Correct format: "module_name:fastapi_instance_name"
        host="0.0.0.0",
        port=config.LOCAL_SERVER_PORT,
        log_level=config.LOG_LEVEL.lower(),
        reload=config.UVICORN_RELOAD, # Use reload flag from config
        # lifespan="on" # lifespan is automatically handled by FastAPI now
    )
# --------------------------------------------------------------------------------
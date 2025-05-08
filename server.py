# boutique_ai_project/server.py

import logging
import asyncio
import base64
import json
import os
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Response as FastAPIResponse, Header, Query, Depends
from fastapi.responses import PlainTextResponse
from fastapi.routing import APIRoute
from starlette.websockets import WebSocketState # Import for checking state

from twilio.twiml.voice_response import VoiceResponse, Start

import config # Root config
from core.services.llm_client import LLMClient
from core.services.fingerprint_generator import FingerprintGenerator
from core.services.telephony_wrapper import TelephonyWrapper
from core.services.crm_wrapper import CRMWrapper
from core.services.data_wrapper import DataWrapper, ClayWebhookError
from core.communication.voice_handler import VoiceHandler
from core.agents.resource_manager import ResourceManager
from core.agents.acquisition_agent import AcquisitionAgent, DEFAULT_CLAY_SERVICE_AUTOMATION_CONFIG
from core.agents.sales_agent import SalesAgent
from core.automation.browser_automator_interface import BrowserAutomatorInterface
# --- Import the REAL Automator Implementation ---
# Ensure this file exists and Playwright is installed if using this automator
from core.automation.multimodal_playwright_automator import MultiModalPlaywrightAutomator
# --- OR Import a Mock for testing if Playwright isn't ready ---
# from core.automation.mock_browser_automator import MockBrowserAutomator # Assuming you create a mock file
from core.database_setup import setup_supabase_tables, create_client as create_supabase_client_for_setup

# --- Mock Automator (Defined here for simplicity if not using separate file) ---
# Replace this with importing your real implementation for production.
class MockBrowserAutomatorForServer(BrowserAutomatorInterface):
    def __init__(self, llm_client: Optional[LLMClient] = None): self.llm_client = llm_client; logger.info("MockBrowserAutomatorForServer initialized.")
    async def setup_session(self, *args, **kwargs): logger.info("MockAutomator(Server): setup_session"); return True
    async def close_session(self, *args, **kwargs): logger.info("MockAutomator(Server): close_session")
    async def navigate_to_page(self, url: str, *args, **kwargs): logger.info(f"MockAutomator(Server): navigate_to_page {url}"); return True
    async def take_screenshot(self, *args, **kwargs): logger.info("MockAutomator(Server): take_screenshot"); return b"dummy_screenshot_bytes_content"
    async def fill_form_and_submit(self, form_interaction_plan: List[Dict[str, Any]], *args, **kwargs): logger.info(f"MockAutomator(Server): fill_form_and_submit. Plan: {form_interaction_plan}"); return True
    async def check_success_condition(self, indicator: Dict[str, Any], *args, **kwargs): logger.info(f"MockAutomator(Server): check_success_condition with indicator: {indicator}"); return True
    async def extract_resources_from_page(self, rules: List[Dict[str, Any]], *args, **kwargs): logger.info(f"MockAutomator(Server): extract_resources_from_page with rules: {rules}"); return {"api_key": "mock_server_extracted_key_123"}
    async def solve_captcha_if_present(self, *args, **kwargs): logger.info("MockAutomator(Server): solve_captcha_if_present"); return True
    async def get_cookies(self) -> Optional[List[Dict[str, Any]]]: logger.info("MockAutomator(Server): get_cookies"); return [{"name": "mock_cookie", "value": "mock_value"}]
    async def full_signup_and_extract(self, service_name: str, signup_url: str, form_interaction_plan: List[Dict[str, Any]], signup_details_generated: Dict[str, Any], success_indicator: Dict[str, Any], resource_extraction_rules: List[Dict[str, Any]], captcha_config: Optional[Dict[str, Any]] = None, max_retries: int = 0):
        logger.info(f"MockAutomator(Server): full_signup_and_extract for {service_name} at {signup_url}")
        if service_name == "clay.com": return {"status": "success", "extracted_resources": {"api_key": "mock_clay_trial_key_from_server_automator"}}
        return {"status": "failed", "reason": "Mocked failure by MockAutomator(Server)"}

logger = logging.getLogger("boutique_ai_server_orchestrator")

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
    active_sales_agents: Dict[str, SalesAgent] = {}
    supabase_client_for_setup: Optional[Any] = None
    is_shutting_down: bool = False

# --- FastAPI Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    logger.info("Boutique AI Server (Level 45+ Orchestrator) initializing...")
    app.state = AppState() # Attach state object to the app instance

    # --- Database Setup ---
    if config.SUPABASE_ENABLED:
        logger.info("Attempting Supabase database schema setup/verification...")
        try:
            app.state.supabase_client_for_setup = create_supabase_client_for_setup()
            if app.state.supabase_client_for_setup:
                await setup_supabase_tables(app.state.supabase_client_for_setup)
                logger.info("Supabase database schema setup/verification complete.")
            else:
                logger.error("Failed to create Supabase client for database setup. Tables may not be created.")
        except Exception as db_setup_e:
            logger.error(f"Error during Supabase database setup: {db_setup_e}", exc_info=True)
            logger.critical("CRITICAL: Database tables might not be correctly set up. Check logs and Supabase dashboard.")
    else:
        logger.warning("Supabase is not enabled. Skipping database table setup.")

    # --- Initialize Shared Dependencies ---
    try:
        app.state.llm_client = LLMClient()
        app.state.fingerprint_generator = FingerprintGenerator(llm_client=app.state.llm_client)

        # --- Instantiate Browser Automator ---
        # DECISION POINT: Choose which automator to use.
        use_real_automator = config.get_bool_env_var("USE_REAL_BROWSER_AUTOMATOR", default=False)
        if use_real_automator:
            try:
                # Ensure Playwright is installed if using this
                app.state.browser_automator = MultiModalPlaywrightAutomator(
                    llm_client=app.state.llm_client,
                    headless=not config.get_bool_env_var("PLAYWRIGHT_HEADFUL_MODE", default=True) # Default headless for server
                )
                logger.info("Lifespan Startup: Using REAL MultiModalPlaywrightAutomator.")
            except ImportError:
                 logger.error("Failed to import MultiModalPlaywrightAutomator. Ensure 'playwright' is installed. Falling back to Mock.")
                 app.state.browser_automator = MockBrowserAutomatorForServer(llm_client=app.state.llm_client)
            except Exception as auto_err:
                 logger.error(f"Error initializing MultiModalPlaywrightAutomator: {auto_err}. Falling back to Mock.", exc_info=True)
                 app.state.browser_automator = MockBrowserAutomatorForServer(llm_client=app.state.llm_client)
        else:
            app.state.browser_automator = MockBrowserAutomatorForServer(llm_client=app.state.llm_client)
            logger.warning("Lifespan Startup: Using MOCK BrowserAutomator (set USE_REAL_BROWSER_AUTOMATOR=true in .env to enable real one).")

        app.state.resource_manager = ResourceManager(
            llm_client=app.state.llm_client,
            fingerprint_generator=app.state.fingerprint_generator,
            browser_automator=app.state.browser_automator # Pass the chosen instance
        )
        app.state.data_wrapper = DataWrapper()
        app.state.crm_wrapper = CRMWrapper()
        app.state.telephony_wrapper = TelephonyWrapper()

        # --- Initialize and Start AcquisitionAgent ---
        acq_criteria = {
            "niche": config.AGENT_TARGET_NICHE_DEFAULT,
            "clay_enrichment_table_webhook_url": config.CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY,
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
        if not acq_criteria["clay_enrichment_table_webhook_url"]:
            logger.error("CRITICAL: CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY not set. AcquisitionAgent Clay enrichment will fail.")

        app.state.acquisition_agent = AcquisitionAgent(
            agent_id="GlobalAcquisitionAgent_001",
            resource_manager=app.state.resource_manager, data_wrapper=app.state.data_wrapper,
            llm_client=app.state.llm_client, crm_wrapper=app.state.crm_wrapper,
            target_criteria=acq_criteria
        )
        await app.state.acquisition_agent.start()
        logger.info("AcquisitionAgent started in background.")

    except Exception as init_err:
        logger.critical(f"CRITICAL ERROR during application initialization: {init_err}", exc_info=True)
        raise RuntimeError("Failed to initialize core application components.") from init_err

    # --- Application is Ready ---
    app.state.is_shutting_down = False
    logger.info("Boutique AI Server initialization complete. Ready for requests.")
    yield # Application runs here

    # --- Shutdown Logic ---
    logger.info("Boutique AI Server shutting down...")
    app.state.is_shutting_down = True

    if hasattr(app.state, 'acquisition_agent') and app.state.acquisition_agent:
        logger.info("Stopping AcquisitionAgent...")
        await app.state.acquisition_agent.stop()

    active_calls = list(app.state.active_sales_agents.values())
    if active_calls:
        logger.info(f"Signaling end for {len(active_calls)} active sales calls...")
        stop_tasks = [agent.stop_call("Server Shutdown") for agent in active_calls]
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        await asyncio.sleep(2) # Brief wait for cleanup signals

    logger.info("Closing shared resources...")
    await DataWrapper.close_session()
    if hasattr(app.state, 'browser_automator') and app.state.browser_automator and hasattr(app.state.browser_automator, 'close_session'):
        await app.state.browser_automator.close_session()

    logger.info("Boutique AI Server shutdown complete.")

# --- FastAPI App Instance ---
app = FastAPI(
    title="Boutique AI Server & Orchestrator",
    version="1.4.1", # Incremented version
    lifespan=lifespan
)

# --- Dependency Injection Helper ---
async def get_app_state() -> AppState:
    """Dependency injector to get the application state."""
    if not hasattr(app, "state") or not isinstance(app.state, AppState):
         logger.critical("App state not initialized correctly!")
         raise HTTPException(status_code=503, detail="Server state not ready")
    return app.state

# --- Twilio Call Webhook ---
@app.post("/call_webhook", tags=["Twilio"], response_class=PlainTextResponse)
async def handle_twilio_call_webhook(request: Request, state: AppState = Depends(get_app_state)):
    """Handles Twilio webhook for incoming/answered calls. Responds with TwiML <Stream>."""
    if not state.telephony_wrapper:
        raise HTTPException(status_code=503, detail="Telephony service unavailable")

    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    call_status = form_data.get("CallStatus")
    to_number = form_data.get("To")
    from_number = form_data.get("From")
    logger.info(f"[{call_sid}] Received /call_webhook. Status: {call_status}, From: {from_number}, To: {to_number}")

    if not call_sid: raise HTTPException(status_code=400, detail="CallSid missing")

    prospect_phone = from_number if to_number == config.TWILIO_PHONE_NUMBER else to_number
    if not prospect_phone:
        logger.error(f"[{call_sid}] Could not determine prospect phone. From={from_number}, To={to_number}")
        raise HTTPException(status_code=400, detail="Could not determine prospect phone")

    ws_scheme = "wss" if request.url.scheme == "https" or config.BASE_WEBHOOK_URL.startswith("https://") else "ws"
    host = request.headers.get("host", f"localhost:{config.LOCAL_SERVER_PORT}")
    encoded_prospect_phone = prospect_phone.strip().replace('+', '%2B')
    ws_path = f"/call_ws?call_sid={call_sid}&prospect_phone={encoded_prospect_phone}"
    ws_url_absolute = f"{ws_scheme}://{host}{ws_path}"

    response = VoiceResponse()
    start_verb = Start()
    start_verb.stream(url=ws_url_absolute)
    response.append(start_verb)
    response.pause(length=1)
    logger.info(f"[{call_sid}] Responding to Twilio with TwiML <Stream> to: {ws_url_absolute}")
    return PlainTextResponse(str(response), media_type="application/xml")

# --- Twilio Media Stream WebSocket ---
@app.websocket("/call_ws")
async def websocket_call_endpoint(
    websocket: WebSocket,
    call_sid: str = Query(...),
    prospect_phone: str = Query(...)
):
    """Handles bidirectional audio stream and manages SalesAgent lifecycle for the call."""
    state: AppState = websocket.app.state
    await websocket.accept()
    logger.info(f"[{call_sid}] WebSocket accepted for prospect: {prospect_phone}. Initializing SalesAgent.")

    if not all([state.llm_client, state.telephony_wrapper, state.crm_wrapper, state.data_wrapper]):
        logger.error(f"[{call_sid}] Core services not ready. Closing WebSocket.")
        await websocket.close(code=1011, reason="Server services not ready"); return

    current_stream_sid: Optional[str] = None

    async def send_audio_to_twilio_ws(csid: str, audio_chunk: bytes):
        if csid == call_sid and current_stream_sid and websocket.client_state == WebSocketState.CONNECTED:
            try:
                media_payload = base64.b64encode(audio_chunk).decode('utf-8')
                await websocket.send_json({"event": "media", "streamSid": current_stream_sid, "media": {"payload": media_payload}})
            except Exception as e: logger.warning(f"[{call_sid}] WS Error sending audio: {type(e).__name__}")

    async def send_mark_to_twilio_ws(csid: str, mark_name: str):
        if csid == call_sid and current_stream_sid and websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json({"event": "mark", "streamSid": current_stream_sid, "mark": {"name": mark_name}})
                logger.info(f"[{call_sid}] Sent mark '{mark_name}' to Twilio.")
            except Exception as e: logger.warning(f"[{call_sid}] WS Error sending mark: {type(e).__name__}")

    voice_handler = VoiceHandler(transcript_callback=None, error_callback=None)

    pre_call_contact_info = await state.crm_wrapper.get_contact_info(identifier=prospect_phone, identifier_column="phone_number")
    if not pre_call_contact_info:
        logger.info(f"[{call_sid}] No existing contact for {prospect_phone}. Creating shell.")
        shell_data = {"phone_number": prospect_phone, "status": "IncomingCall_NewContact", "source_info": f"Call_{call_sid}"}
        pre_call_contact_info = await state.crm_wrapper.upsert_contact(shell_data, "phone_number") or shell_data

    sales_agent = SalesAgent(
        agent_id=f"SalesAgent_{call_sid}", target_phone_number=prospect_phone,
        voice_handler=voice_handler, llm_client=state.llm_client,
        telephony_wrapper=state.telephony_wrapper, crm_wrapper=state.crm_wrapper,
        initial_prospect_data=pre_call_contact_info,
        send_audio_callback=send_audio_to_twilio_ws, send_mark_callback=send_mark_to_twilio_ws,
        on_call_complete_callback=lambda aid, status, hist, summary: logger.info(f"SERVER: Agent {aid} call complete. Status: {status}. Outcome: {summary.get('final_call_outcome_category','N/A')}"),
        on_call_error_callback=lambda aid, err: logger.error(f"SERVER: Agent {aid} call error: {err}")
    )
    state.active_sales_agents[call_sid] = sales_agent

    try:
        while True:
            message = await websocket.receive_json()
            event = message.get("event")

            if event == "start":
                current_stream_sid = message.get("start", {}).get("streamSid")
                if not current_stream_sid:
                    logger.error(f"[{call_sid}] WS 'start' event missing streamSid: {message}")
                    await sales_agent._handle_fatal_error("Missing streamSid in start event"); break
                logger.info(f"[{call_sid}] WS 'start' event. Stream SID: {current_stream_sid}. Starting SalesAgent.")
                asyncio.create_task(sales_agent.start_sales_call(call_sid, current_stream_sid))
            elif event == "media" and current_stream_sid:
                payload = message.get("media", {}).get("payload")
                if payload: await sales_agent.handle_incoming_audio(base64.b64decode(payload))
            elif event == "stop":
                logger.info(f"[{call_sid}] WS 'stop' event received. Signaling agent.")
                sales_agent.signal_call_ended_externally("WebSocket Stop Event"); break
            elif event == "mark": logger.debug(f"[{call_sid}] WS 'mark': {message.get('mark', {}).get('name')}")
            elif event == "connected": logger.info(f"[{call_sid}] WS 'connected' event.")
            else: logger.debug(f"[{call_sid}] Unknown WS event: {event}")
    except WebSocketDisconnect as e:
        logger.warning(f"[{call_sid}] WebSocket disconnected. Code: {e.code}")
        if sales_agent.is_call_active: sales_agent.signal_call_ended_externally(f"WS Disconnected (Code: {e.code})")
    except Exception as e:
        logger.error(f"[{call_sid}] Error in WebSocket handler: {e}", exc_info=True)
        if sales_agent.is_call_active: sales_agent.signal_call_ended_externally(f"WS Handler Exception: {type(e).__name__}")
    finally:
        logger.info(f"[{call_sid}] Cleaning up WS & SalesAgent resources for {prospect_phone}...")
        if call_sid in state.active_sales_agents: del state.active_sales_agents[call_sid]
        logger.info(f"[{call_sid}] WS resources cleaned up from server perspective.")


# --- Webhook for Clay.com Enrichment Results ---
@app.post("/webhooks/clay/enrichment_results", tags=["Webhooks"])
async def handle_clay_enrichment_webhook(
    request: Request,
    state: AppState = Depends(get_app_state),
    x_callback_auth_token: Optional[str] = Header(None, alias=config.CLAY_CALLBACK_AUTH_HEADER_NAME)
):
    """Receives enriched data from Clay, validates, and passes to AcquisitionAgent."""
    configured_secret = config.CLAY_RESULTS_CALLBACK_SECRET_TOKEN
    if configured_secret:
        if not x_callback_auth_token:
            logger.warning("Clay webhook: Auth token header missing.")
            raise HTTPException(status_code=401, detail="Unauthorized: Missing authentication token")
        if x_callback_auth_token != configured_secret:
            logger.warning(f"Clay webhook: Invalid auth token.")
            raise HTTPException(status_code=403, detail="Forbidden: Invalid authentication token")
    elif not configured_secret:
        logger.warning("Clay webhook: CLAY_RESULTS_CALLBACK_SECRET_TOKEN not configured. Endpoint is INSECURE.")

    logger.info("Received POST on /webhooks/clay/enrichment_results")
    if not state.acquisition_agent:
        logger.error("Clay webhook: AcquisitionAgent not available!")
        raise HTTPException(status_code=503, detail="Acquisition service unavailable")

    try: payload = await request.json()
    except json.JSONDecodeError:
        logger.error("Clay webhook: Invalid JSON payload.")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    logger.debug(f"Clay webhook payload (first 500 chars): {json.dumps(payload)[:500]}")
    correlation_id = payload.get("_correlation_id")

    if not correlation_id:
        logger.warning("Clay webhook: Payload missing '_correlation_id'. Cannot process. Payload: %s", payload)
        return FastAPIResponse(content={"status": "error", "message": "Payload missing _correlation_id"}, status_code=400)

    asyncio.create_task(state.acquisition_agent.handle_clay_enrichment_result(payload))
    logger.info(f"Clay enrichment result for correlation_id '{correlation_id}' queued for processing.")
    return FastAPIResponse(content={"status": "received", "message": "Result queued for processing."}, status_code=202)

# --- Admin/Test Endpoint to Trigger Outbound Call ---
@app.post("/admin/actions/initiate_call", tags=["Admin Actions"])
async def trigger_outbound_call_admin(
    target_number: str,
    crm_contact_id: Optional[str] = Query(None, description="Optional CRM ID to associate"),
    state: AppState = Depends(get_app_state)
):
    """Initiates an outbound call via Twilio for testing/admin purposes."""
    if not state.telephony_wrapper: raise HTTPException(status_code=503, detail="Telephony service unavailable.")
    if not target_number: raise HTTPException(status_code=400, detail="Missing 'target_number'.")
    logger.info(f"ADMIN: API request to initiate call to: {target_number}. CRM Contact ID: {crm_contact_id}")
    
    custom_params_for_twilio = {}
    if crm_contact_id: custom_params_for_twilio["crm_id"] = crm_contact_id
    
    call_sid = await state.telephony_wrapper.initiate_call(
        target_number=target_number,
        custom_parameters=custom_params_for_twilio if custom_params_for_twilio else None
    )
    if call_sid: return {"message": "Outbound call initiated.", "call_sid": call_sid}
    raise HTTPException(status_code=500, detail="Failed to initiate call via telephony provider.")

# --- Health Check Endpoint ---
@app.get("/health", tags=["System"])
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

# --- Main Execution Block ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Boutique AI Server Orchestrator on host 0.0.0.0 port {config.LOCAL_SERVER_PORT}")
    logger.info(f"Ensure BASE_WEBHOOK_URL ({config.BASE_WEBHOOK_URL}) is publicly accessible.")
    logger.info(f"Twilio Call Webhook URL: {config.BASE_WEBHOOK_URL.rstrip('/')}/call_webhook")
    logger.info(f"Clay Enrichment Results Webhook URL: {config.BASE_WEBHOOK_URL.rstrip('/')}/webhooks/clay/enrichment_results")
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=config.LOCAL_SERVER_PORT,
        log_level=config.LOG_LEVEL.lower(),
        reload=config.UVICORN_RELOAD # Use reload flag from config
    )
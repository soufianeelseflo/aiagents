# boutique_ai_project/server.py
import logging # Moved up
import asyncio
import base64
import json
import os # Make sure os is imported
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

# Initialize logger at the very start of the module, AFTER config might have set up basicConfig
import config # Ensures config.py runs and sets up basic logging first
logger = logging.getLogger("boutique_ai_server_orchestrator") # DEFINED EARLY

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Response as FastAPIResponse, Header, Query, Depends
from fastapi.responses import HTMLResponse, PlainTextResponse # Ensure HTMLResponse is imported
from fastapi.staticfiles import StaticFiles # <-- NEW IMPORT
from fastapi.templating import Jinja2Templates # <-- NEW IMPORT
from pydantic import BaseModel, EmailStr # <-- NEW IMPORT for contact form
from starlette.websockets import WebSocketState


from twilio.twiml.voice_response import VoiceResponse, Start

# Attempt to import LLMClient and handle potential failure gracefully
LLMClientImported = False
try:
    from core.services.llm_client import LLMClient
    LLMClientImported = True
except ImportError as e:
    logger.error(f"Failed to import LLMClient from core.services.llm_client: {e}. LLM functionalities will be impaired. This is often due to a circular dependency or an error within LLMClient itself.")
    LLMClient = None # Define as None so type hints don't break later if it's optional

from core.services.fingerprint_generator import FingerprintGenerator
from core.services.telephony_wrapper import TelephonyWrapper
from core.services.crm_wrapper import CRMWrapper
from core.services.data_wrapper import DataWrapper, ClayWebhookError
from core.communication.voice_handler import VoiceHandler
from core.agents.resource_manager import ResourceManager
from core.agents.acquisition_agent import AcquisitionAgent, DEFAULT_CLAY_SERVICE_AUTOMATION_CONFIG
from core.agents.sales_agent import SalesAgent
from core.automation.browser_automator_interface import BrowserAutomatorInterface
from core.database_setup import setup_supabase_tables, create_client as create_supabase_client_for_setup
from core.services.deployment_manager import DeploymentManagerInterface, DockerDeploymentManager # Assuming DockerDeploymentManager

# --- Choose Automator Implementation ---
if config.get_bool_env_var("USE_REAL_BROWSER_AUTOMATOR", default=False):
    try:
        from core.automation.multimodal_playwright_automator import MultiModalPlaywrightAutomator
        SelectedBrowserAutomator = MultiModalPlaywrightAutomator
        logger.info("Server Startup: Using REAL MultiModalPlaywrightAutomator.")
    except ImportError as e:
        logger.error(f"Server Startup: Failed to import MultiModalPlaywrightAutomator (Playwright installed?): {e}. Falling back to Mock.")
        from core.automation.mock_browser_automator import MockBrowserAutomator
        SelectedBrowserAutomator = MockBrowserAutomator
else:
    from core.automation.mock_browser_automator import MockBrowserAutomator
    SelectedBrowserAutomator = MockBrowserAutomator
    logger.warning("Server Startup: Using MOCK BrowserAutomator. Set USE_REAL_BROWSER_AUTOMATOR=true to enable Playwright.")

# --- Application State ---
class AppState:
    llm_client: Optional[LLMClient] # Now Optional
    fingerprint_generator: FingerprintGenerator
    browser_automator: BrowserAutomatorInterface
    resource_manager: ResourceManager
    data_wrapper: DataWrapper
    crm_wrapper: CRMWrapper
    telephony_wrapper: TelephonyWrapper
    acquisition_agent: Optional[AcquisitionAgent] = None # Initialize as None
    provisioning_agent: Optional[Any] = None # Placeholder for ProvisioningAgent
    deployment_manager: Optional[DeploymentManagerInterface] = None # Placeholder
    active_sales_agents: Dict[str, SalesAgent] = {}
    supabase_client_for_setup: Optional[Any] = None
    is_shutting_down: bool = False

# --- FastAPI Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app_fastapi: FastAPI):
    logger.info("Boutique AI Server initializing (lifespan manager)...")
    app_fastapi.state = AppState()

    try:
        if LLMClientImported and LLMClient is not None:
            app_fastapi.state.llm_client = LLMClient()
        else:
            app_fastapi.state.llm_client = None
            logger.error("LLMClient could not be initialized during lifespan setup. LLM-dependent features will be UNAVAILABLE.")

        if config.SUPABASE_ENABLED:
            logger.info("Lifespan: Attempting Supabase database schema setup/verification...")
            app_fastapi.state.supabase_client_for_setup = create_supabase_client_for_setup()
            if app_fastapi.state.supabase_client_for_setup:
                await setup_supabase_tables(app_fastapi.state.supabase_client_for_setup)
            else: logger.error("Lifespan: Failed to create Supabase client for DB setup.")
        else: logger.warning("Lifespan: Supabase not enabled.")

        app_fastapi.state.fingerprint_generator = FingerprintGenerator(llm_client=app_fastapi.state.llm_client)
        app_fastapi.state.browser_automator = SelectedBrowserAutomator(
            llm_client=app_fastapi.state.llm_client,
            headless=not config.get_bool_env_var("PLAYWRIGHT_HEADFUL_MODE", default=True)
        )
        app_fastapi.state.resource_manager = ResourceManager(
            llm_client=app_fastapi.state.llm_client,
            fingerprint_generator=app_fastapi.state.fingerprint_generator,
            browser_automator=app_fastapi.state.browser_automator
        )
        app_fastapi.state.data_wrapper = DataWrapper()
        app_fastapi.state.crm_wrapper = CRMWrapper()
        app_fastapi.state.telephony_wrapper = TelephonyWrapper()
        
        app_fastapi.state.deployment_manager = DockerDeploymentManager() 
        logger.info("Lifespan: DockerDeploymentManager initialized.")


        acq_criteria = {
            "niche": config.AGENT_TARGET_NICHE_DEFAULT,
            "clay_enrichment_table_webhook_url": config.CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY,
            "clay_enrichment_table_name": config.get_env_var("CLAY_ENRICHMENT_TABLE_NAME", "Primary Lead Enrichment", required=False),
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
        if not app_fastapi.state.llm_client:
            logger.warning("Lifespan: AcquisitionAgent will be initialized without a functional LLMClient. Qualification and analysis features will be impaired.")
        
        app_fastapi.state.acquisition_agent = AcquisitionAgent(
            agent_id="GlobalAcquisitionAgent_001",
            resource_manager=app_fastapi.state.resource_manager,
            data_wrapper=app_fastapi.state.data_wrapper,
            llm_client=app_fastapi.state.llm_client,
            crm_wrapper=app_fastapi.state.crm_wrapper,
            target_criteria=acq_criteria
        )
        await app_fastapi.state.acquisition_agent.start()
        logger.info("Lifespan: AcquisitionAgent started.")

        if app_fastapi.state.deployment_manager and app_fastapi.state.llm_client: 
            from core.agents.provisioning_agent import ProvisioningAgent 
            app_fastapi.state.provisioning_agent = ProvisioningAgent(
                agent_id="GlobalProvisioningAgent_001",
                resource_manager=app_fastapi.state.resource_manager,
                crm_wrapper=app_fastapi.state.crm_wrapper,
                llm_client=app_fastapi.state.llm_client,
                deployment_manager=app_fastapi.state.deployment_manager
            )
            await app_fastapi.state.provisioning_agent.start()
            logger.info("Lifespan: ProvisioningAgent started.")
        elif not app_fastapi.state.deployment_manager:
            logger.warning("Lifespan: DeploymentManager not initialized. ProvisioningAgent will not start.")
        elif not app_fastapi.state.llm_client:
             logger.warning("Lifespan: LLMClient not initialized. ProvisioningAgent will not start as it might need LLM for config interpretation.")


    except Exception as init_err:
        logger.critical(f"CRITICAL ERROR during application lifespan startup: {init_err}", exc_info=True)
        if app_fastapi.state.acquisition_agent and app_fastapi.state.acquisition_agent._is_running: # type: ignore
            await app_fastapi.state.acquisition_agent.stop() # type: ignore
        if app_fastapi.state.provisioning_agent and app_fastapi.state.provisioning_agent._is_running: # type: ignore
            await app_fastapi.state.provisioning_agent.stop() # type: ignore
        raise RuntimeError("Failed to initialize core application components during lifespan startup.") from init_err

    app_fastapi.state.is_shutting_down = False
    logger.info("Boutique AI Server initialization complete via lifespan manager. Ready for requests.")
    yield 

    logger.info("Boutique AI Server shutting down (lifespan manager)...")
    app_fastapi.state.is_shutting_down = True

    if app_fastapi.state.acquisition_agent and app_fastapi.state.acquisition_agent._is_running: # type: ignore
        logger.info("Lifespan: Stopping AcquisitionAgent...")
        await app_fastapi.state.acquisition_agent.stop() # type: ignore
    
    if app_fastapi.state.provisioning_agent and app_fastapi.state.provisioning_agent._is_running: # type: ignore
        logger.info("Lifespan: Stopping ProvisioningAgent...")
        await app_fastapi.state.provisioning_agent.stop() # type: ignore

    active_calls = list(app_fastapi.state.active_sales_agents.values())
    if active_calls:
        logger.info(f"Lifespan: Signaling end for {len(active_calls)} active sales calls...")
        stop_tasks = [agent.stop_call("Server Shutdown") for agent in active_calls]
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        await asyncio.sleep(2)

    logger.info("Lifespan: Closing shared resources...")
    await DataWrapper.close_session()
    if hasattr(app_fastapi.state, 'browser_automator') and app_fastapi.state.browser_automator and hasattr(app_fastapi.state.browser_automator, 'close_session'):
        await app_fastapi.state.browser_automator.close_session()

    logger.info("Boutique AI Server shutdown complete via lifespan manager.")

# --- FastAPI App Instance ---
app = FastAPI(
    title="Boutique AI Server & Orchestrator",
    version="1.6.0", 
    lifespan=lifespan 
)

# --- Dependency Injection Helper ---
async def get_app_state() -> AppState:
    if not hasattr(app, "state") or not isinstance(app.state, AppState): # type: ignore
         logger.critical("App state not initialized correctly! This should not happen if lifespan manager ran.")
         raise HTTPException(status_code=503, detail="Server application state not ready")
    return app.state # type: ignore

# *****************************************************************************
# * NEW WEBSITE SECTION START                        *
# *****************************************************************************

# --- Determine base directory for static/templates ---
# Assumes server.py is at the project root. Adjust if your structure is different.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Mount static files (CSS, JS, images) ---
# This makes files in the "static" directory accessible via "/static" URL path
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# --- Configure Jinja2 templates ---
# This tells FastAPI where to find your HTML files
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# --- Pydantic model for contact form data validation ---
class ContactFormSubmission(BaseModel):
    name: str
    email: EmailStr # Pydantic's EmailStr validates email format
    company: Optional[str] = None
    phone: Optional[str] = None
    interest: str
    message: str

# --- API endpoint for handling contact form submissions ---
@app.post("/api/contact_submission", tags=["Website Interactions"])
async def handle_contact_submission(
    submission: ContactFormSubmission,
    state: AppState = Depends(get_app_state) # Reuse your existing AppState dependency
):
    logger.info(f"Received new contact form submission from: {submission.email}, Interest: {submission.interest}")
    
    if not state.crm_wrapper:
        logger.error("CRM_WRAPPER NOT AVAILABLE. Cannot process contact submission.")
        raise HTTPException(status_code=503, detail="Service not available to process contact form.")

    # Prepare data for CRM, splitting name and setting appropriate status
    first_name = submission.name.split(" ")[0] if " " in submission.name else submission.name
    last_name_parts = submission.name.split(" ", 1)
    last_name = last_name_parts[1] if len(last_name_parts) > 1 else None

    crm_payload = {
        "email": submission.email,
        "first_name": first_name,
        "last_name": last_name,
        "company_name": submission.company,
        "phone_number": submission.phone,
        "status": "Website_Inquiry_New", # You can customize this status
        "source_info": f"Website Contact Form - Interest: {submission.interest}",
        "notes": submission.message, # This is the message from the textarea
        "last_activity_timestamp": datetime.now(timezone.utc).isoformat(),
        "client_id": None # Assuming these leads are for Boutique AI itself. Adjust if needed.
    }
    
    # Clean payload of any keys with None values if your CRM upsert requires it
    crm_payload_cleaned = {k: v for k, v in crm_payload.items() if v is not None}


    upserted_contact = await state.crm_wrapper.upsert_contact(
        contact_data=crm_payload_cleaned,
        unique_key_column="email" # Assuming email is the primary unique identifier for website leads
    )

    if upserted_contact:
        contact_id = upserted_contact.get('id', 'N/A')
        logger.info(f"Contact submission from {submission.email} successfully saved/updated in CRM. Contact ID: {contact_id}")
        # You could add notification logic here (e.g., email your sales team)
        # For example: await send_sales_notification(submission)
        return {"message": "Thank you for your submission! We will be in touch shortly."}
    else:
        logger.error(f"Failed to save contact submission from {submission.email} to CRM.")
        raise HTTPException(status_code=500, detail="There was an error processing your request. Please try again later.")

# --- Routes to serve your HTML pages ---
# These make your HTML files accessible via browser navigation
@app.get("/", response_class=HTMLResponse, tags=["Website Pages"])
async def serve_homepage(request: Request):
    # "request": request is necessary for Jinja2 templates
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/features", response_class=HTMLResponse, tags=["Website Pages"]) # Using path without .html for cleaner URLs
async def serve_features_page(request: Request):
    return templates.TemplateResponse("features.html", {"request": request})

@app.get("/solutions", response_class=HTMLResponse, tags=["Website Pages"])
async def serve_solutions_page(request: Request):
    return templates.TemplateResponse("solutions.html", {"request": request})

@app.get("/how-it-works", response_class=HTMLResponse, tags=["Website Pages"])
async def serve_how_it_works_page(request: Request):
    return templates.TemplateResponse("how-it-works.html", {"request": request})

@app.get("/contact", response_class=HTMLResponse, tags=["Website Pages"])
async def serve_contact_page(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})

# *****************************************************************************
# * NEW WEBSITE SECTION END                         *
# *****************************************************************************


# --- Twilio Call Webhook ---
@app.post("/call_webhook", tags=["Twilio"], response_class=PlainTextResponse)
async def handle_twilio_call_webhook(request: Request, state: AppState = Depends(get_app_state)):
    if not state.telephony_wrapper:
        logger.error("/call_webhook: Telephony service unavailable.")
        raise HTTPException(status_code=503, detail="Telephony service unavailable")

    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    call_status = form_data.get("CallStatus")
    to_number = form_data.get("To")
    from_number = form_data.get("From")
    logger.info(f"[{call_sid}] Received /call_webhook. Status: {call_status}, From: {from_number}, To: {to_number}")

    if not call_sid:
        logger.error("/call_webhook: CallSid missing from Twilio request.")
        raise HTTPException(status_code=400, detail="CallSid missing")

    prospect_phone = from_number if to_number == config.TWILIO_PHONE_NUMBER else to_number
    if not prospect_phone:
        logger.error(f"[{call_sid}] /call_webhook: Could not determine prospect phone. From={from_number}, To={to_number}")
        raise HTTPException(status_code=400, detail="Could not determine prospect phone")

    host = request.headers.get("x-forwarded-host", request.headers.get("host"))
    ws_scheme = request.headers.get("x-forwarded-proto", request.url.scheme)

    if not host: 
        host = f"localhost:{config.LOCAL_SERVER_PORT}"
        logger.warning(f"[{call_sid}] No x-forwarded-host or host header, defaulting to localhost for WS URL: {host}")
    
    if ws_scheme not in ["ws", "wss"]:
        logger.warning(f"[{call_sid}] Invalid scheme '{ws_scheme}' from x-forwarded-proto/request.url.scheme. Defaulting to 'ws' for localhost, 'wss' otherwise.")
        ws_scheme = "ws" if "localhost" in host else "wss"


    encoded_prospect_phone = base64.urlsafe_b64encode(prospect_phone.encode()).decode() 
    ws_path_with_query = f"/call_ws?call_sid={call_sid}&prospect_phone_b64={encoded_prospect_phone}"
    ws_url_absolute = f"{ws_scheme}://{host}{ws_path_with_query}"

    response = VoiceResponse()
    start_verb = Start()
    start_verb.stream(url=ws_url_absolute)
    response.append(start_verb)
    logger.info(f"[{call_sid}] /call_webhook: Responding to Twilio with TwiML <Stream> to: {ws_url_absolute}")
    return PlainTextResponse(str(response), media_type="application/xml")

# --- Twilio Media Stream WebSocket ---
@app.websocket("/call_ws")
async def websocket_call_endpoint(
    websocket: WebSocket,
    call_sid: str = Query(...),
    prospect_phone_b64: str = Query(...) 
):
    state: AppState = websocket.app.state # type: ignore
    await websocket.accept()
    
    try:
        prospect_phone = base64.urlsafe_b64decode(prospect_phone_b64.encode()).decode()
    except Exception as e:
        logger.error(f"[{call_sid}] WebSocket: Invalid prospect_phone_b64 encoding: {prospect_phone_b64}. Error: {e}")
        await websocket.close(code=1003, reason="Invalid prospect phone encoding"); return
        
    logger.info(f"[{call_sid}] WebSocket accepted for prospect: {prospect_phone}. Initializing SalesAgent.")

    if not LLMClientImported or not state.llm_client: 
        logger.error(f"[{call_sid}] WebSocket: LLMClient not available. Cannot proceed with SalesAgent. Closing WebSocket.")
        await websocket.close(code=1011, reason="Server LLM service not ready"); return
    if not all([state.telephony_wrapper, state.crm_wrapper, state.data_wrapper]):
        logger.error(f"[{call_sid}] WebSocket: One or more core services (Telephony, CRM, Data) not ready. Closing WebSocket.")
        await websocket.close(code=1011, reason="Server core services not ready"); return

    current_stream_sid: Optional[str] = None

    async def send_audio_to_twilio_ws(csid: str, audio_chunk: bytes):
        if csid == call_sid and current_stream_sid and websocket.application_state == WebSocketState.CONNECTED:
            try:
                media_payload = base64.b64encode(audio_chunk).decode('utf-8')
                await websocket.send_json({"event": "media", "streamSid": current_stream_sid, "media": {"payload": media_payload}})
            except WebSocketDisconnect: logger.warning(f"[{call_sid}] WS disconnected while trying to send audio.")
            except Exception as e: logger.warning(f"[{call_sid}] WS Error sending audio: {type(e).__name__} - {e}")
        elif websocket.application_state != WebSocketState.CONNECTED:
             logger.warning(f"[{call_sid}] WS send audio: Not connected (State: {websocket.application_state}).")

    async def send_mark_to_twilio_ws(csid: str, mark_name: str):
        if csid == call_sid and current_stream_sid and websocket.application_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json({"event": "mark", "streamSid": current_stream_sid, "mark": {"name": mark_name}})
                logger.info(f"[{call_sid}] Sent mark '{mark_name}' to Twilio.")
            except WebSocketDisconnect: logger.warning(f"[{call_sid}] WS disconnected while trying to send mark.")
            except Exception as e: logger.warning(f"[{call_sid}] WS Error sending mark: {type(e).__name__} - {e}")
        elif websocket.application_state != WebSocketState.CONNECTED:
             logger.warning(f"[{call_sid}] WS send mark: Not connected (State: {websocket.application_state}).")

    voice_handler = VoiceHandler(transcript_callback=None, error_callback=None) # Will be re-assigned by SalesAgent
    pre_call_contact_info = await state.crm_wrapper.get_contact_info(identifier=prospect_phone, identifier_column="phone_number") # type: ignore
    if not pre_call_contact_info:
        shell_data = {"phone_number": prospect_phone, "status": "IncomingCall_NewContact", "source_info": f"Call_{call_sid}"}
        pre_call_contact_info = await state.crm_wrapper.upsert_contact(shell_data, "phone_number") or shell_data # type: ignore

    sales_agent = SalesAgent(
        agent_id=f"SalesAgent_{call_sid}", target_phone_number=prospect_phone,
        voice_handler=voice_handler, llm_client=state.llm_client, # type: ignore
        telephony_wrapper=state.telephony_wrapper, crm_wrapper=state.crm_wrapper, # type: ignore
        initial_prospect_data=pre_call_contact_info, # type: ignore
        send_audio_callback=send_audio_to_twilio_ws, send_mark_callback=send_mark_to_twilio_ws,
        on_call_complete_callback=lambda aid, status, hist, summary: logger.info(f"SERVER: Agent {aid} call complete. Status: {status}. Outcome: {summary.get('final_call_outcome_category','N/A')}"),
        on_call_error_callback=lambda aid, err: logger.error(f"SERVER: Agent {aid} call error: {err}")
    )
    state.active_sales_agents[call_sid] = sales_agent

    try:
        while websocket.application_state == WebSocketState.CONNECTED:
            message = await websocket.receive_json()
            event = message.get("event")

            if event == "start":
                current_stream_sid = message.get("start", {}).get("streamSid")
                if not current_stream_sid:
                    logger.error(f"[{call_sid}] WS 'start' event missing streamSid: {message}")
                    await sales_agent._handle_fatal_error("Missing streamSid in start event"); break # type: ignore
                logger.info(f"[{call_sid}] WS 'start' event. Stream SID: {current_stream_sid}. Starting SalesAgent.")
                asyncio.create_task(sales_agent.start_sales_call(call_sid, current_stream_sid))
            elif event == "media" and current_stream_sid:
                payload = message.get("media", {}).get("payload")
                if payload: await sales_agent.handle_incoming_audio(base64.b64decode(payload))
            elif event == "stop":
                logger.info(f"[{call_sid}] WS 'stop' event received. Signaling agent and breaking loop.")
                sales_agent.signal_call_ended_externally("WebSocket Stop Event"); break
            elif event == "mark": logger.debug(f"[{call_sid}] WS 'mark': {message.get('mark', {}).get('name')}")
            elif event == "connected": logger.info(f"[{call_sid}] WS 'connected' event (already handled).")
            else: logger.debug(f"[{call_sid}] Unknown WS event: {event}, Data: {json.dumps(message)[:200]}...")
    except WebSocketDisconnect as e:
        logger.warning(f"[{call_sid}] WebSocket disconnected. Code: {e.code}, Reason: {e.reason}")
        if sales_agent.is_call_active: sales_agent.signal_call_ended_externally(f"WS Disconnected (Code: {e.code})")
    except Exception as e:
        logger.error(f"[{call_sid}] Error in WebSocket handler: {e}", exc_info=True)
        if sales_agent.is_call_active: sales_agent.signal_call_ended_externally(f"WS Handler Exception: {type(e).__name__}")
    finally:
        logger.info(f"[{call_sid}] WebSocket cleaning up for {prospect_phone}...")
        if call_sid in state.active_sales_agents:
            agent_to_cleanup = state.active_sales_agents.pop(call_sid)
            if agent_to_cleanup and agent_to_cleanup.is_call_active:
                logger.info(f"[{call_sid}] Forcing SalesAgent cleanup as it might still be active.")
                asyncio.create_task(agent_to_cleanup._cleanup_call("WebSocket Final Cleanup", None)) # type: ignore
        logger.info(f"[{call_sid}] WebSocket resources cleanup from server perspective complete for {prospect_phone}.")


# --- Webhook for Clay.com Enrichment Results ---
@app.post("/webhooks/clay/enrichment_results", tags=["Webhooks"])
async def handle_clay_enrichment_webhook(
    request: Request,
    state: AppState = Depends(get_app_state),
    x_callback_auth_token: Optional[str] = Header(None, alias=config.CLAY_CALLBACK_AUTH_HEADER_NAME)
):
    configured_secret = config.CLAY_RESULTS_CALLBACK_SECRET_TOKEN
    if configured_secret:
        if not x_callback_auth_token:
            logger.warning("Clay webhook: Auth token header missing.")
            raise HTTPException(status_code=401, detail="Unauthorized: Missing authentication token")
        if x_callback_auth_token != configured_secret:
            logger.warning(f"Clay webhook: Invalid auth token received.")
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

    correlation_id = payload.get("_correlation_id")
    if not correlation_id:
        logger.warning("Clay webhook: Payload missing '_correlation_id'. Cannot process. Full Payload: %s", str(payload)[:500])
        return FastAPIResponse(content={"status": "error", "message": "Payload missing _correlation_id"}, status_code=400) # type: ignore

    logger.info(f"Clay enrichment result for correlation_id '{correlation_id}' queued for processing.")
    asyncio.create_task(state.acquisition_agent.handle_clay_enrichment_result(payload)) # type: ignore
    return FastAPIResponse(content={"status": "received", "message": "Result queued for processing."}, status_code=202) # type: ignore

# --- Admin/Test Endpoint to Trigger Outbound Call ---
@app.post("/admin/actions/initiate_call", tags=["Admin Actions"])
async def trigger_outbound_call_admin(
    target_number: str = Query(..., description="E.164 formatted phone number to call."), 
    crm_contact_id: Optional[str] = Query(None, description="Optional CRM ID to associate with the call log."),
    state: AppState = Depends(get_app_state)
):
    if not state.telephony_wrapper:
        logger.error("Admin initiate_call: Telephony service unavailable.")
        raise HTTPException(status_code=503, detail="Telephony service unavailable.")
    if not target_number: 
        logger.error("Admin initiate_call: Missing 'target_number'.")
        raise HTTPException(status_code=400, detail="Missing 'target_number'.")

    logger.info(f"ADMIN: API request to initiate call to: {target_number}. CRM Contact ID: {crm_contact_id}")
    custom_params_for_twilio: Dict[str, str] = {}
    if crm_contact_id: custom_params_for_twilio["crm_id"] = crm_contact_id

    call_sid = await state.telephony_wrapper.initiate_call( # type: ignore
        target_number=target_number,
        custom_parameters=custom_params_for_twilio if custom_params_for_twilio else None
    )
    if call_sid:
        logger.info(f"ADMIN: Outbound call initiated successfully. Call SID: {call_sid}")
        return {"message": "Outbound call initiated.", "call_sid": call_sid}

    logger.error(f"ADMIN: Failed to initiate call to {target_number} via telephony provider.")
    raise HTTPException(status_code=500, detail="Failed to initiate call via telephony provider.")

# --- Health Check Endpoint ---
@app.get("/health", tags=["System"])
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat(), "version": app.version} # type: ignore

# --- Main Execution Block ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Attempting to start Boutique AI Server Orchestrator on host 0.0.0.0 port {config.LOCAL_SERVER_PORT}")
    if not config.BASE_WEBHOOK_URL:
        logger.critical("CRITICAL: BASE_WEBHOOK_URL is not set in environment variables. Webhooks (Twilio, Clay callback) WILL FAIL.")
    else:
        logger.info(f"Expected Twilio Call Webhook URL (configure in Twilio): {config.BASE_WEBHOOK_URL.rstrip('/')}/call_webhook")
        logger.info(f"Expected Clay Enrichment Results Webhook URL (configure in Clay): {config.BASE_WEBHOOK_URL.rstrip('/')}/webhooks/clay/enrichment_results")
        logger.info(f"Website will be available at your server's base URL, e.g., {config.BASE_WEBHOOK_URL} or http://localhost:{config.LOCAL_SERVER_PORT}")


    if not LLMClientImported or LLMClient is None:
        logger.critical("CRITICAL: LLMClient failed to import or initialize. Core AI functionalities will be severely impaired or non-functional. Review startup logs for import errors related to 'core.services.llm_client'.")

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=config.LOCAL_SERVER_PORT,
        log_level=config.LOG_LEVEL.lower(),
        reload=config.UVICORN_RELOAD,
        lifespan="on" 
    )
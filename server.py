# /server.py: (Holistically Reviewed, Twilio Validation Added)
# --------------------------------------------------------------------------------
# boutique_ai_project/server.py

import logging
import asyncio
import base64
import json
import os
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List, Tuple # Added Tuple
from datetime import datetime, timezone

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Response as FastAPIResponse, Header, Query, Depends
from fastapi.responses import PlainTextResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from twilio.twiml.voice_response import VoiceResponse, Start
from twilio.request_validator import RequestValidator # For webhook security

# --- Configuration (Must be imported first) ---
import config

# --- Initialize Logger for this module ---
logger = logging.getLogger(__name__) # Use standard __name__

# --- Graceful Service & Agent Imports with Aliases ---
# This helps manage optional components or allows for mocking during testing.
try:
    from core.services.llm_client import LLMClient, LLMClientSetupError
except ImportError as e: logger.error(f"Failed to import LLMClient: {e}", exc_info=True); LLMClient = None; LLMClientSetupError = type('LLMClientSetupError', (Exception,), {})
try:
    from core.services.fingerprint_generator import FingerprintGenerator
except ImportError as e: logger.error(f"Failed to import FingerprintGenerator: {e}", exc_info=True); FingerprintGenerator = None
try:
    from core.services.telephony_wrapper import TelephonyWrapper, TelephonyError
except ImportError as e: logger.error(f"Failed to import TelephonyWrapper: {e}", exc_info=True); TelephonyWrapper = None; TelephonyError = type('TelephonyError', (Exception,), {})
try:
    from core.services.crm_wrapper import CRMWrapper, CRMError
except ImportError as e: logger.error(f"Failed to import CRMWrapper: {e}", exc_info=True); CRMWrapper = None; CRMError = type('CRMError', (Exception,), {})
try:
    from core.services.data_wrapper import DataWrapper, ClayWebhookError
except ImportError as e: logger.error(f"Failed to import DataWrapper: {e}", exc_info=True); DataWrapper = None; ClayWebhookError = type('ClayWebhookError', (Exception,), {})
try:
    from core.communication.voice_handler import VoiceHandler, VoiceHandlerError, STTProvider, TTSProvider
except ImportError as e: logger.error(f"Failed to import VoiceHandler: {e}", exc_info=True); VoiceHandler = None; VoiceHandlerError = type('VoiceHandlerError', (Exception,), {})
try:
    from core.agents.resource_manager import ResourceManager
except ImportError as e: logger.error(f"Failed to import ResourceManager: {e}", exc_info=True); ResourceManager = None
try:
    from core.agents.acquisition_agent import AcquisitionAgent, DEFAULT_CLAY_SERVICE_AUTOMATION_CONFIG
except ImportError as e: logger.error(f"Failed to import AcquisitionAgent: {e}", exc_info=True); AcquisitionAgent = None; DEFAULT_CLAY_SERVICE_AUTOMATION_CONFIG = {}
try:
    from core.agents.sales_agent import SalesAgent, SalesCallPhase, ProspectSentiment, CallOutcomeCategory
except ImportError as e: logger.error(f"Failed to import SalesAgent: {e}", exc_info=True); SalesAgent = None
try:
    from core.automation.browser_automator_interface import BrowserAutomatorInterface
except ImportError as e: logger.error(f"Failed to import BrowserAutomatorInterface: {e}", exc_info=True); BrowserAutomatorInterface = None

# --- Browser Automator Implementation Choice ---
AutomatorImplementation = None
if config.USE_REAL_BROWSER_AUTOMATOR and BrowserAutomatorInterface:
    try:
        from core.automation.multimodal_playwright_automator import MultiModalPlaywrightAutomator, PlaywrightAutomatorError
        AutomatorImplementation = MultiModalPlaywrightAutomator
        logger.info("Using REAL MultiModalPlaywrightAutomator.")
    except ImportError as e:
        logger.error(f"Could not import MultiModalPlaywrightAutomator: {e}. Browser automation may be limited.", exc_info=True)
        PlaywrightAutomatorError = type('PlaywrightAutomatorError', (Exception,), {}) # Define dummy error
elif BrowserAutomatorInterface: # If real not requested or interface missing
    logger.warning("USE_REAL_BROWSER_AUTOMATOR is False or BrowserAutomatorInterface failed import. Real browser automation disabled.")
    # Consider having a MockBrowserAutomator here or ensure None is handled gracefully

# --- Database Setup (Conditional Import) ---
if config.SUPABASE_ENABLED:
    try:
        from core.database_setup import setup_supabase_tables
        # Supabase client (async version)
        from supabase_py_async import create_client as create_supabase_async_client, AsyncClient as SupabaseAsyncClient
    except ImportError as e:
        logger.error(f"Failed to import database_setup or supabase_py_async: {e}. Supabase functionality will be impaired.", exc_info=True)
        setup_supabase_tables = None
        create_supabase_async_client = None
        SupabaseAsyncClient = None
else:
    logger.warning("Supabase is DISABLED via config. No database operations will occur.")
    setup_supabase_tables = None
    create_supabase_async_client = None
    SupabaseAsyncClient = None

# --- Application State ---
# This class holds shared service instances.
class AppState:
    llm_client: Optional[LLMClient] = None
    fingerprint_generator: Optional[FingerprintGenerator] = None
    browser_automator: Optional[BrowserAutomatorInterface] = None
    resource_manager: Optional[ResourceManager] = None
    data_wrapper: Optional[DataWrapper] = None
    crm_wrapper: Optional[CRMWrapper] = None
    telephony_wrapper: Optional[TelephonyWrapper] = None
    acquisition_agent: Optional[AcquisitionAgent] = None
    active_sales_agents: Dict[str, SalesAgent] = {} # call_sid -> SalesAgent instance
    supabase_client: Optional[SupabaseAsyncClient] = None # For shared async client
    twilio_validator: Optional[RequestValidator] = None
    is_shutting_down: bool = False

# --- FastAPI Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    logger.info("Application Lifespan: Startup sequence initiated...")
    app.state = AppState() # Initialize state object on the app

    # 1. Initialize Twilio Request Validator (critical for webhook security)
    if config.TWILIO_AUTH_TOKEN:
        app.state.twilio_validator = RequestValidator(config.TWILIO_AUTH_TOKEN)
        logger.info("Twilio RequestValidator initialized.")
    else:
        logger.warning("TWILIO_AUTH_TOKEN not found. Twilio webhooks will be INSECURE (validation skipped).")

    # 2. Initialize Supabase Client (if enabled)
    if config.SUPABASE_ENABLED and create_supabase_async_client and config.SUPABASE_URL and config.SUPABASE_KEY:
        try:
            app.state.supabase_client = await create_supabase_async_client(config.SUPABASE_URL, config.SUPABASE_KEY)
            logger.info(f"Supabase async client initialized for URL: {config.SUPABASE_URL[:20]}...")
            # Attempt to run database schema setup
            if setup_supabase_tables:
                try:
                    await setup_supabase_tables(app.state.supabase_client)
                    logger.info("Supabase database schema setup/verification process completed (check specific logs from database_setup).")
                except Exception as db_setup_e:
                    logger.error(f"Error during Supabase database setup execution: {db_setup_e}", exc_info=True)
            else:
                logger.warning("setup_supabase_tables function not available, skipping schema setup.")
        except Exception as e:
            logger.critical(f"Failed to initialize Supabase client: {e}", exc_info=True)
            app.state.supabase_client = None # Ensure it's None on failure
    else:
        logger.warning("Supabase not enabled or client/config missing. Skipping Supabase client initialization and DB setup.")

    # 3. Initialize LLMClient
    if LLMClient:
        try:
            app.state.llm_client = LLMClient()
            logger.info("LLMClient initialized.")
        except LLMClientSetupError as e: # Catch specific setup error from LLMClient
            logger.critical(f"LLMClient setup failed: {e}. LLM-dependent features will be unavailable.")
            app.state.llm_client = None
        except Exception as e:
            logger.critical(f"Unexpected error initializing LLMClient: {e}", exc_info=True)
            app.state.llm_client = None
    else: logger.warning("LLMClient class not imported. LLM functionalities disabled.")

    # 4. Initialize FingerprintGenerator
    if FingerprintGenerator and app.state.llm_client: # LLMClient is optional for FingerprintGenerator
        app.state.fingerprint_generator = FingerprintGenerator(llm_client=app.state.llm_client)
        logger.info("FingerprintGenerator initialized (with LLM support).")
    elif FingerprintGenerator:
        app.state.fingerprint_generator = FingerprintGenerator(llm_client=None)
        logger.info("FingerprintGenerator initialized (LLM support disabled).")
    else: logger.warning("FingerprintGenerator class not imported.")

    # 5. Initialize BrowserAutomator
    if AutomatorImplementation and app.state.llm_client:
        try:
            app.state.browser_automator = AutomatorImplementation(
                llm_client=app.state.llm_client,
                headless=not config.PLAYWRIGHT_HEADFUL_MODE
            )
            # Perform initial setup for the browser instance (e.g., launch browser)
            # This makes it ready for agents to create pages/contexts.
            # Some automators might not need an explicit global setup like this.
            # For Playwright, launching the browser instance here can be beneficial.
            # await app.state.browser_automator.setup_session() # Removed: Agents should manage their own full sessions
            logger.info(f"BrowserAutomator ({AutomatorImplementation.__name__}) initialized.")
        except Exception as e:
            logger.critical(f"Failed to initialize BrowserAutomator ({AutomatorImplementation.__name__}): {e}", exc_info=True)
            app.state.browser_automator = None
    elif config.USE_REAL_BROWSER_AUTOMATOR:
        logger.warning("Real BrowserAutomator requested but could not be initialized (LLM client missing or import failed).")

    # 6. Initialize other services
    if CRMWrapper and app.state.supabase_client: # CRMWrapper depends on supabase_client
        app.state.crm_wrapper = CRMWrapper(db_client=app.state.supabase_client)
        logger.info("CRMWrapper initialized.")
    elif CRMWrapper and not app.state.supabase_client:
        logger.warning("CRMWrapper not initialized: Supabase client unavailable.")
    else: logger.warning("CRMWrapper class not imported.")

    if DataWrapper: # DataWrapper might use aiohttp internally
        app.state.data_wrapper = DataWrapper() # Assuming it manages its own HTTP client session if needed
        logger.info("DataWrapper initialized.")
    else: logger.warning("DataWrapper class not imported.")

    if TelephonyWrapper:
        try:
            app.state.telephony_wrapper = TelephonyWrapper()
            logger.info("TelephonyWrapper initialized.")
        except TelephonyError as e:
            logger.critical(f"TelephonyWrapper initialization failed: {e}. Telephony functions will be unavailable.")
            app.state.telephony_wrapper = None
        except Exception as e:
            logger.critical(f"Unexpected error initializing TelephonyWrapper: {e}", exc_info=True)
            app.state.telephony_wrapper = None
    else: logger.warning("TelephonyWrapper class not imported. Telephony functions disabled.")

    # 7. Initialize ResourceManager
    if ResourceManager and app.state.llm_client and app.state.fingerprint_generator and app.state.browser_automator:
        app.state.resource_manager = ResourceManager(
            llm_client=app.state.llm_client,
            fingerprint_generator=app.state.fingerprint_generator,
            browser_automator=app.state.browser_automator,
            crm_wrapper=app.state.crm_wrapper # Pass CRM if available
        )
        logger.info("ResourceManager initialized.")
    elif ResourceManager:
        logger.warning("ResourceManager not initialized due to missing dependencies (LLM, FingerprintGenerator, or BrowserAutomator).")

    # 8. Initialize and Start AcquisitionAgent (as a background task)
    if AcquisitionAgent and app.state.llm_client and app.state.resource_manager and app.state.data_wrapper and app.state.crm_wrapper:
        try:
            acq_criteria = {
                "niche": config.AGENT_TARGET_NICHE_DEFAULT,
                "clay_enrichment_table_webhook_url": config.CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY,
                "clay_enrichment_table_name": config.get_env_var("CLAY_ENRICHMENT_TABLE_NAME", required=False, default="leads_for_enrichment"),
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
            app.state.acquisition_agent = AcquisitionAgent(
                agent_id="GlobalAcquisitionAgent_001",
                resource_manager=app.state.resource_manager,
                data_wrapper=app.state.data_wrapper,
                llm_client=app.state.llm_client,
                crm_wrapper=app.state.crm_wrapper,
                target_criteria=acq_criteria
            )
            # Start the agent's main loop as a background task
            asyncio.create_task(app.state.acquisition_agent.start()) # Call its main run/start method
            logger.info("AcquisitionAgent background task created and started.")
        except Exception as e:
            logger.critical(f"Failed to initialize or start AcquisitionAgent: {e}", exc_info=True)
            app.state.acquisition_agent = None
    elif AcquisitionAgent:
        logger.warning("AcquisitionAgent not initialized due to missing dependencies.")


    logger.info("Application startup sequence complete. Server is ready to accept requests.")
    yield # Application runs here

    # --- Shutdown Logic ---
    logger.info("Application Lifespan: Shutdown sequence initiated...")
    app.state.is_shutting_down = True

    if app.state.acquisition_agent and hasattr(app.state.acquisition_agent, 'stop') and asyncio.iscoroutinefunction(app.state.acquisition_agent.stop):
        logger.info("Attempting to stop AcquisitionAgent...")
        try:
            await asyncio.wait_for(app.state.acquisition_agent.stop(), timeout=10.0)
            logger.info("AcquisitionAgent stopped.")
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for AcquisitionAgent to stop.")
        except Exception as e:
            logger.error(f"Error stopping AcquisitionAgent: {e}", exc_info=True)

    active_call_sids = list(app.state.active_sales_agents.keys())
    if active_call_sids:
        logger.info(f"Stopping {len(active_call_sids)} active sales calls...")
        stop_tasks = [
            agent.stop_call("Server Shutdown")
            for agent in app.state.active_sales_agents.values() if hasattr(agent, 'stop_call')
        ]
        results = await asyncio.gather(*stop_tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error stopping sales agent for call SID {active_call_sids[i]}: {result}", exc_info=result)
    app.state.active_sales_agents.clear()


    if app.state.browser_automator and hasattr(app.state.browser_automator, 'close_session'):
        logger.info("Closing BrowserAutomator session...")
        try:
            await app.state.browser_automator.close_session()
            logger.info("BrowserAutomator session closed.")
        except Exception as e:
            logger.error(f"Error closing BrowserAutomator session: {e}", exc_info=True)
    
    if DataWrapper and hasattr(DataWrapper, 'close_global_session'): # If DataWrapper uses a global aiohttp session
        logger.info("Closing DataWrapper global HTTP session (if any)...")
        await DataWrapper.close_global_session()


    # Supabase client (supabase-py-async) does not require explicit closing of the client itself typically.
    # The underlying httpx client connections are managed.
    if app.state.supabase_client:
        logger.info("Supabase client does not require explicit async close in this version of the library.")
        # If using a version that wraps httpx.AsyncClient directly and needs closing:
        # if hasattr(app.state.supabase_client, 'aclose'): # Check if an aclose method exists
        #    await app.state.supabase_client.aclose()
        #    logger.info("Supabase async client (underlying httpx) closed.")


    logger.info("Application shutdown sequence complete.")

# --- FastAPI App Instance ---
app = FastAPI(
    title=config.OPENROUTER_APP_NAME or "Boutique AI Server",
    version="2.0.0", # Signifies major refactor
    lifespan=lifespan,
    # Add other FastAPI configurations like middleware, exception handlers if needed
)

# --- Middleware for Twilio Webhook Validation ---
class TwilioRequestValidationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if "/call_webhook" in request.url.path or "/status_callback" in request.url.path: # Adjust paths as needed
            if app.state.twilio_validator:
                try:
                    # Reconstruct full URL
                    # scheme = request.headers.get("x-forwarded-proto", request.url.scheme) # Prefer x-forwarded-proto if behind proxy
                    # host = request.headers.get("x-forwarded-host", request.url.hostname)
                    # port = request.url.port
                    # if (scheme == "http" and port != 80) or (scheme == "https" and port != 443):
                    #     url = f"{scheme}://{host}:{port}{request.url.path}"
                    # else:
                    #     url = f"{scheme}://{host}{request.url.path}"
                    # More reliable: use BASE_WEBHOOK_URL if path matches
                    url = str(request.url) # This might be internal URL if behind proxy
                    if config.BASE_WEBHOOK_URL: # Use configured public URL
                        url = config.BASE_WEBHOOK_URL.rstrip('/') + request.url.path

                    # Get form data without consuming the body for FastAPI
                    form_data = {}
                    async with request.form() as form:
                         form_data = {key: value for key, value in form.items()}

                    twilio_signature = request.headers.get("X-Twilio-Signature")

                    if not twilio_signature:
                        logger.warning(f"Twilio webhook missing X-Twilio-Signature header for URL: {url}")
                        return JSONResponse(content={"error": "Missing X-Twilio-Signature header"}, status_code=403)

                    logger.debug(f"Validating Twilio request for URL: {url}, Form: {form_data}, Signature: {twilio_signature}")
                    if not app.state.twilio_validator.validate(url, form_data, twilio_signature):
                        logger.error(f"Twilio webhook validation FAILED for URL: {url}. Signature: {twilio_signature}. Form Data: {form_data}")
                        return JSONResponse(content={"error": "Twilio signature validation failed"}, status_code=403)
                    logger.info(f"Twilio webhook validation SUCCESS for URL: {url}")
                except Exception as e:
                    logger.error(f"Error during Twilio validation: {e}", exc_info=True)
                    # Fallback to deny if validation process itself fails.
                    return JSONResponse(content={"error": "Twilio validation process error"}, status_code=500)
            else:
                logger.warning(f"Twilio validator not initialized. Skipping validation for {request.url.path} (INSECURE).")
        
        response = await call_next(request)
        return response

if config.TWILIO_AUTH_TOKEN: # Only add middleware if validator can be initialized
    app.add_middleware(TwilioRequestValidationMiddleware)
else:
    logger.warning("TwilioRequestValidationMiddleware NOT ADDED due to missing TWILIO_AUTH_TOKEN.")


# --- Dependency Injection Helper for AppState ---
async def get_app_state() -> AppState:
    if not hasattr(app, "state") or not app.state: # Check if app.state exists and is initialized
        # This can happen if a request comes in before lifespan has fully run,
        # or if lifespan failed critically.
        logger.error("Application state not available in dependency. Lifespan might have failed.")
        raise HTTPException(status_code=503, detail="Server components not ready or initialization failed.")
    return app.state

# --- API Endpoints ---

@app.get("/health", tags=["System"])
async def health_check(state: AppState = Depends(get_app_state)):
    """Basic health check, also reports status of some core components."""
    # Check critical components that should have been initialized in lifespan
    llm_status = "ok" if state.llm_client else "error: not_initialized"
    telephony_status = "ok" if state.telephony_wrapper else "error: not_initialized"
    supabase_status = "ok" if state.supabase_client else ("disabled" if not config.SUPABASE_ENABLED else "error: not_initialized")
    
    overall_status = "ok"
    if not all([state.llm_client, state.telephony_wrapper, (state.supabase_client or not config.SUPABASE_ENABLED)]):
        overall_status = "degraded"
        logger.warning(f"Health check degraded: LLM: {llm_status}, Telephony: {telephony_status}, Supabase: {supabase_status}")
        return JSONResponse(
            content={
                "status": overall_status, "timestamp": datetime.now(timezone.utc).isoformat(),
                "services": {"llm_client": llm_status, "telephony_wrapper": telephony_status, "supabase_client": supabase_status}
            }, status_code=503
        )

    return {
        "status": overall_status, "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": "Boutique AI Server is operational.",
        "services": {
            "llm_client": llm_status,
            "telephony_wrapper": telephony_status,
            "supabase_client": supabase_status,
            "acquisition_agent_running": bool(state.acquisition_agent and hasattr(state.acquisition_agent, '_is_running') and state.acquisition_agent._is_running)
        }
    }

# Twilio Call Webhook (Handles Incoming Calls or Start of Outbound Calls)
@app.post("/call_webhook", tags=["Twilio"], response_class=PlainTextResponse)
async def handle_twilio_call_webhook(
    request: Request, 
    state: AppState = Depends(get_app_state),
    # FastAPI automatically parses form data if request.form() is awaited
    CallSid: Optional[str] = Query(None, alias="CallSid"), # For some reason, Twilio might send as query param sometimes
    From: Optional[str] = Query(None),
    To: Optional[str] = Query(None),
    CallStatus: Optional[str] = Query(None)
):
    # Form data will be available via await request.form()
    # Twilio validation is now handled by middleware if token is present
    # If validation fails, middleware returns 403 before this function is reached.

    form_data = await request.form()
    # Prioritize form data, but fallback to query params if needed (though Twilio usually POSTs form data)
    call_sid = form_data.get("CallSid", CallSid)
    from_number = form_data.get("From", From)
    to_number = form_data.get("To", To)
    call_status = form_data.get("CallStatus", CallStatus)

    if not state.telephony_wrapper:
        logger.error(f"[{call_sid}] /call_webhook: TelephonyWrapper not available. Cannot process call.")
        # Return a TwiML response that hangs up or plays an error.
        response = VoiceResponse(); response.say("We are currently unable to process your call. Please try again later."); response.hangup()
        return PlainTextResponse(str(response), media_type="application/xml", status_code=503)
    if not config.BASE_WEBHOOK_URL:
        logger.error(f"[{call_sid}] /call_webhook: BASE_WEBHOOK_URL not configured. Cannot construct WebSocket URL.")
        response = VoiceResponse(); response.say("Server configuration error. Call cannot proceed."); response.hangup()
        return PlainTextResponse(str(response), media_type="application/xml", status_code=500)


    logger.info(f"[{call_sid}] Received /call_webhook. Status: {call_status}, From: {from_number}, To: {to_number}")

    if not call_sid:
        logger.error("/call_webhook: CallSid missing from Twilio request (form and query).")
        # Cannot construct a meaningful TwiML response without CallSid context for logging,
        # but must still respond to Twilio.
        response = VoiceResponse(); response.reject(reason="busy") # Or hangup
        return PlainTextResponse(str(response), media_type="application/xml", status_code=400)

    prospect_phone = from_number if to_number == config.TWILIO_PHONE_NUMBER else to_number
    if not prospect_phone:
        logger.error(f"[{call_sid}] Could not determine prospect phone. From={from_number}, To={to_number}, TwilioNum={config.TWILIO_PHONE_NUMBER}")
        response = VoiceResponse(); response.hangup()
        return PlainTextResponse(str(response), media_type="application/xml")

    # Construct WebSocket URL for Twilio Media Stream
    # Determine scheme based on BASE_WEBHOOK_URL
    ws_scheme = "wss" if config.BASE_WEBHOOK_URL.lower().startswith("https://") else "ws"
    # Extract host and port from BASE_WEBHOOK_URL if possible, otherwise use request's host.
    try:
        base_url_parts = config.BASE_WEBHOOK_URL.split("://")[1].split(":")
        host = base_url_parts[0]
        # port_str = base_url_parts[1] if len(base_url_parts) > 1 else None # Port might not be in BASE_WEBHOOK_URL if std http/s
    except Exception: # Fallback to request headers if BASE_WEBHOOK_URL parsing fails
        logger.warning(f"[{call_sid}] Could not parse host from BASE_WEBHOOK_URL ('{config.BASE_WEBHOOK_URL}'). Falling back to request headers for WS URL construction.")
        host = request.headers.get("host", f"localhost:{config.LOCAL_SERVER_PORT}") # Fallback further

    # Ensure prospect_phone and call_sid are URL-encoded for query parameters
    encoded_prospect_phone = requests.utils.quote(prospect_phone) if requests else prospect_phone # If requests lib available
    encoded_call_sid = requests.utils.quote(call_sid) if requests else call_sid

    ws_path = f"/call_ws?call_sid={encoded_call_sid}&prospect_phone={encoded_prospect_phone}"
    # Construct with host derived from BASE_WEBHOOK_URL to ensure it's the public-facing one
    ws_url_absolute = f"{ws_scheme}://{host.split(':')[0]}{ws_path}" # Remove port from host if already there, as scheme implies it

    twiml_response = VoiceResponse()
    start_verb = Start()
    start_verb.stream(url=ws_url_absolute)
    twiml_response.append(start_verb)
    # No pause needed typically, Deepgram connects quickly.
    logger.info(f"[{call_sid}] Responding to Twilio with TwiML <Stream> to: {ws_url_absolute}")
    return PlainTextResponse(str(twiml_response), media_type="application/xml")

# Twilio Media Stream WebSocket
@app.websocket("/call_ws")
async def websocket_call_endpoint(
    websocket: WebSocket,
    call_sid: str = Query(...), # Ensure these are always provided
    prospect_phone: str = Query(...),
    state: AppState = Depends(get_app_state) # Get AppState via dependency injection
):
    await websocket.accept()
    logger.info(f"[{call_sid}] WebSocket accepted for prospect: {prospect_phone}. Initializing SalesAgent.")

    if state.is_shutting_down:
        logger.warning(f"[{call_sid}] Server is shutting down. Rejecting new WebSocket connection.")
        await websocket.close(code=1011, reason="Server shutting down")
        return

    # Critical dependency check for SalesAgent operation
    if not all([SalesAgent, state.llm_client, state.telephony_wrapper, VoiceHandler]):
        missing_deps = [
            name for name, obj in [
                ("SalesAgentClass", SalesAgent), ("LLMClient", state.llm_client),
                ("TelephonyWrapper", state.telephony_wrapper), ("VoiceHandlerClass", VoiceHandler)
            ] if not obj
        ]
        reason = f"Server dependencies not ready for SalesAgent: {', '.join(missing_deps)}"
        logger.error(f"[{call_sid}] Cannot start SalesAgent. {reason}. Closing WebSocket.")
        await websocket.close(code=1011, reason=reason)
        return

    current_stream_sid: Optional[str] = None # Will be set by 'start' event from Twilio
    sales_agent_instance: Optional[SalesAgent] = None

    async def send_audio_to_twilio_ws(csid: str, audio_chunk: bytes):
        if csid == call_sid and current_stream_sid and websocket.client_state == WebSocketState.CONNECTED:
            try:
                media_payload = base64.b64encode(audio_chunk).decode('utf-8')
                await websocket.send_json({"event": "media", "streamSid": current_stream_sid, "media": {"payload": media_payload}})
            except WebSocketDisconnect: logger.warning(f"[{call_sid}] WS disconnected (send_audio).")
            except Exception as e: logger.error(f"[{call_sid}] WS Error sending audio: {e}", exc_info=True)

    async def send_mark_to_twilio_ws(csid: str, mark_name: str):
         if csid == call_sid and current_stream_sid and websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json({"event": "mark", "streamSid": current_stream_sid, "mark": {"name": mark_name}})
                logger.info(f"[{call_sid}] Sent mark '{mark_name}' to Twilio.")
            except WebSocketDisconnect: logger.warning(f"[{call_sid}] WS disconnected (send_mark).")
            except Exception as e: logger.error(f"[{call_sid}] WS Error sending mark: {e}", exc_info=True)

    # Callback for when the SalesAgent completes its call (either success or failure)
    async def on_agent_call_complete(agent_id: str, final_phase: SalesCallPhase, history: List[Dict], summary: Dict):
        logger.info(f"[{agent_id}/{call_sid}] SalesAgent reported call complete. Final Phase: {final_phase.name}. Summary: {summary.get('final_call_outcome_category', 'N/A')}")
        # WebSocket is typically closed by Twilio after call ends or by stop_call if agent initiates hangup
        if websocket.client_state == WebSocketState.CONNECTED:
             await websocket.close(code=1000, reason=f"Call ended: {final_phase.name}")


    try:
        voice_handler_instance = VoiceHandler(
            call_identifier=call_sid, # For tagging TTS audio
            stt_provider_config={"api_key": config.DEEPGRAM_API_KEY, "model": config.DEEPGRAM_STT_MODEL},
            tts_provider_config={"api_key": config.DEEPGRAM_API_KEY, "model": config.DEEPGRAM_TTS_MODEL},
            send_audio_upstream_callback=send_audio_to_twilio_ws,
            mark_callback=send_mark_to_twilio_ws
            # Transcript and error callbacks will be set by SalesAgent
        )

        # Fetch/create contact info (ensure CRM wrapper is available and configured)
        prospect_data = {"phone_number": prospect_phone, "status": "IncomingCall_NewContact"} # Default
        if state.crm_wrapper and config.SUPABASE_ENABLED:
            try:
                contact_info = await state.crm_wrapper.get_contact_info(identifier=prospect_phone, identifier_column="phone_number")
                if contact_info:
                    prospect_data.update(contact_info)
                    prospect_data["status"] = contact_info.get("status", "IncomingCall_ExistingContact")
                else: # Create a shell contact
                    logger.info(f"[{call_sid}] No existing contact for {prospect_phone}. Creating shell entry.")
                    # Ensure phone_number is a string
                    shell_data = {"phone_number": str(prospect_phone), "status": "IncomingCall_NewContact", "source_info": f"Call_{call_sid}"}
                    created_contact = await state.crm_wrapper.upsert_contact(shell_data, "phone_number")
                    if created_contact: prospect_data.update(created_contact)
            except CRMError as e:
                logger.error(f"[{call_sid}] CRM error fetching/creating contact: {e}", exc_info=True)
                # Proceed with minimal prospect_data
            except Exception as e: # Catch any other unexpected error
                logger.error(f"[{call_sid}] Unexpected error during CRM interaction: {e}", exc_info=True)
        else:
            logger.warning(f"[{call_sid}] CRMWrapper or Supabase not available. Using minimal prospect data.")


        sales_agent_instance = SalesAgent(
            agent_id=f"SalesAgent_{call_sid}", target_phone_number=prospect_phone,
            voice_handler=voice_handler_instance, llm_client=state.llm_client,
            telephony_wrapper=state.telephony_wrapper, crm_wrapper=state.crm_wrapper,
            initial_prospect_data=prospect_data,
            send_audio_callback=send_audio_to_twilio_ws, # Already defined above
            send_mark_callback=send_mark_to_twilio_ws,   # Already defined above
            on_call_complete_callback=on_agent_call_complete,
            on_call_error_callback=lambda aid, err_msg: logger.error(f"SERVER WS: Agent {aid} reported error: {err_msg}")
        )
        state.active_sales_agents[call_sid] = sales_agent_instance
        logger.info(f"[{sales_agent_instance.agent_id}] SalesAgent instance created and tracked.")

        # Main WebSocket message loop
        while True:
            message = await websocket.receive_json()
            event = message.get("event")

            if event == "connected":
                logger.info(f"[{call_sid}] WebSocket 'connected' event received from Twilio.")
                # This is more of an ack from Twilio, main logic starts with "start"

            elif event == "start":
                stream_sid_from_msg = message.get("start", {}).get("streamSid")
                if not stream_sid_from_msg:
                    logger.error(f"[{call_sid}] WS 'start' event missing streamSid: {message}")
                    await sales_agent_instance._handle_fatal_error("Missing streamSid in Twilio start event")
                    break # Fatal error for this call
                current_stream_sid = stream_sid_from_msg
                logger.info(f"[{call_sid}] WS 'start' event. Stream SID: {current_stream_sid}. Starting SalesAgent call logic.")
                asyncio.create_task(sales_agent_instance.start_sales_call(call_sid, current_stream_sid))

            elif event == "media" and current_stream_sid:
                payload = message.get("media", {}).get("payload")
                if payload:
                    await sales_agent_instance.handle_incoming_audio(base64.b64decode(payload))
                else:
                    logger.warning(f"[{call_sid}] WS 'media' event received with no payload.")

            elif event == "mark":
                mark_name = message.get("mark", {}).get("name")
                logger.info(f"[{call_sid}] WS 'mark' event received for mark: {mark_name}")
                if sales_agent_instance and hasattr(sales_agent_instance, 'handle_mark_event'):
                     await sales_agent_instance.handle_mark_event(mark_name)


            elif event == "stop":
                logger.info(f"[{call_sid}] WS 'stop' event received from Twilio. Call ending.")
                await sales_agent_instance.signal_call_ended_externally("Twilio Stop Event")
                break # Exit the message loop

            else:
                logger.warning(f"[{call_sid}] Received unknown WS event: {event}. Message: {json.dumps(message, indent=2)}")

    except WebSocketDisconnect as e:
        logger.warning(f"[{call_sid}] WebSocket disconnected. Code: {e.code}, Reason: '{e.reason}'")
        if sales_agent_instance and sales_agent_instance.is_call_active:
            await sales_agent_instance.signal_call_ended_externally(f"WebSocket Disconnected (Code: {e.code})")
    except Exception as e:
        logger.error(f"[{call_sid}] Error in WebSocket handler: {type(e).__name__} - {e}", exc_info=True)
        if sales_agent_instance and sales_agent_instance.is_call_active:
             await sales_agent_instance.signal_call_ended_externally(f"WebSocket Handler Exception: {type(e).__name__}")
    finally:
        logger.info(f"[{call_sid}] Cleaning up WebSocket handler for prospect {prospect_phone}...")
        if call_sid in state.active_sales_agents:
            # Call stop_call on the agent if it hasn't been called already to ensure final logging etc.
            # but check if call is still marked active by agent to avoid duplicate cleanup.
            if sales_agent_instance and sales_agent_instance.is_call_active:
                logger.info(f"[{call_sid}] Ensuring agent's stop_call is invoked during final cleanup.")
                await sales_agent_instance.stop_call("WebSocket handler final cleanup")
            del state.active_sales_agents[call_sid]
            logger.info(f"[{call_sid}] SalesAgent instance removed from active tracking.")
        
        # Ensure voice_handler is cleaned up if it was initialized for this call
        if 'voice_handler_instance' in locals() and voice_handler_instance.is_connected():
            logger.info(f"[{call_sid}] Ensuring voice_handler is disconnected during final cleanup.")
            await voice_handler_instance.disconnect()

        if websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close(code=1000)
            except RuntimeError as e: # Can happen if already closing
                logger.debug(f"[{call_sid}] WebSocket already closing during final cleanup: {e}")
        logger.info(f"[{call_sid}] WebSocket handler cleanup for {prospect_phone} finished.")


# Webhook for Clay.com Enrichment Results
@app.post("/webhooks/clay/enrichment_results", tags=["Webhooks"])
async def handle_clay_enrichment_webhook(
    request: Request,
    state: AppState = Depends(get_app_state),
    x_callback_auth_token: Optional[str] = Header(None, alias=config.CLAY_CALLBACK_AUTH_HEADER_NAME)
):
    if config.CLAY_RESULTS_CALLBACK_SECRET_TOKEN:
        if not x_callback_auth_token or x_callback_auth_token != config.CLAY_RESULTS_CALLBACK_SECRET_TOKEN:
            logger.warning("Clay webhook rejected: Invalid or missing auth token.")
            raise HTTPException(status_code=403, detail="Forbidden: Invalid authentication token")
        logger.debug("Clay webhook authentication successful.")
    else:
        logger.warning("Clay results webhook running WITHOUT authentication (CLAY_RESULTS_CALLBACK_SECRET_TOKEN not set). Endpoint is INSECURE.")

    if not state.acquisition_agent:
        logger.error("Clay webhook error: AcquisitionAgent not available!")
        return JSONResponse(content={"status": "error", "message": "Acquisition service unavailable"}, status_code=503)

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        logger.error("Clay webhook received invalid JSON payload.")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    logger.info(f"Clay enrichment result received. Payload keys: {list(payload.keys())}")
    correlation_id = payload.get("_correlation_id") # Standard Clay field for matching requests
    if not correlation_id:
        logger.warning("Clay webhook payload missing '_correlation_id'. Cannot process effectively.")
        # Still attempt to process if possible, or return error
        # return JSONResponse(content={"status": "error", "message": "Payload missing _correlation_id"}, status_code=400)


    # Process in background to respond quickly to webhook
    asyncio.create_task(state.acquisition_agent.handle_clay_enrichment_result(payload))
    return JSONResponse(content={"status": "received", "message": "Result queued for processing."}, status_code=202)


# Admin/Test Endpoint to Trigger Outbound Call
@app.post("/admin/actions/initiate_call", tags=["Admin Actions"], response_model=Dict[str, Any])
async def trigger_outbound_call_admin(
    target_number: str = Query(..., description="E.164 formatted number to call, e.g., +15551234567"),
    crm_contact_id: Optional[str] = Query(None, description="Optional CRM ID (UUID) to associate for context."),
    state: AppState = Depends(get_app_state)
):
    if not state.telephony_wrapper:
        logger.error("Admin Call Trigger: Telephony service unavailable.")
        raise HTTPException(status_code=503, detail="Telephony service unavailable.")
    if not target_number.startswith('+') or not target_number[1:].isdigit() or len(target_number) < 10:
         raise HTTPException(status_code=400, detail="Invalid 'target_number' format. Must be E.164 (e.g., +15551234567).")

    logger.info(f"ADMIN: API request to initiate call to: {target_number}. CRM Contact ID: {crm_contact_id}")
    try:
        call_sid = await state.telephony_wrapper.initiate_call(target_number=target_number)
        if call_sid:
            logger.info(f"Admin Call Trigger: Outbound call initiated successfully. Call SID: {call_sid}")
            # Store a placeholder for this admin-initiated call if needed for tracking
            # Or, rely on Twilio status callbacks to eventually log it.
            # For now, just return success.
            return {"message": "Outbound call initiated.", "call_sid": call_sid, "target_number": target_number}
        else:
            logger.error(f"Admin Call Trigger: Failed to initiate call to {target_number} (TelephonyWrapper returned no SID).")
            raise HTTPException(status_code=500, detail="Failed to initiate call via telephony provider.")
    except TelephonyError as e:
        logger.error(f"Admin Call Trigger: TelephonyError initiating call to {target_number}: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Telephony provider error: {e}")
    except Exception as e:
         logger.error(f"Admin Call Trigger: Unexpected error initiating call to {target_number}: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail=f"Unexpected server error: {type(e).__name__}")


# --- Main Execution Block (for local testing) ---
if __name__ == "__main__":
    # This block is for running the server directly with `python server.py`
    # It's not used when Docker runs the CMD instruction.
    # Ensure Uvicorn is installed: pip install uvicorn
    import uvicorn
    logger.info(f"--- Starting Boutique AI Server (Local Development Mode via __main__) ---")
    logger.info(f"Log Level from config: {config.LOG_LEVEL}")
    logger.info(f"Uvicorn Reload enabled: {config.UVICORN_RELOAD}")
    logger.info(f"Server will listen on: 0.0.0.0:{config.LOCAL_SERVER_PORT}")
    if config.BASE_WEBHOOK_URL:
        logger.info(f"Expected Public Base URL for Webhooks: {config.BASE_WEBHOOK_URL}")
        logger.info(f"-> Twilio Call Webhook should be POST to: {config.BASE_WEBHOOK_URL.rstrip('/')}/call_webhook")
        logger.info(f"-> Clay Results Webhook should be POST to: {config.BASE_WEBHOOK_URL.rstrip('/')}/webhooks/clay/enrichment_results")
    else:
        logger.critical("CRITICAL LOCAL DEV WARNING: BASE_WEBHOOK_URL is not set in .env. Webhooks from external services like Twilio WILL fail to reach this local server unless ngrok or similar is used and BASE_WEBHOOK_URL is set accordingly.")
    logger.info("---")

    uvicorn.run(
        "server:app", # "module_name:fastapi_instance_name"
        host="0.0.0.0",
        port=config.LOCAL_SERVER_PORT,
        log_level=config.LOG_LEVEL.lower(), # Uvicorn uses lowercase log levels
        reload=config.UVICORN_RELOAD,
        # lifespan="on" # This is automatically handled by FastAPI via the app's lifespan parameter
    )
# --------------------------------------------------------------------------------
# boutique_ai_project/server.py

import logging
import asyncio
import base64
import json
import os
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Response as FastAPIResponse, Header, Query
from fastapi.responses import PlainTextResponse
from twilio.twiml.voice_response import VoiceResponse, Start

import config # Root config
from core.services.llm_client import LLMClient
from core.services.fingerprint_generator import FingerprintGenerator
from core.services.telephony_wrapper import TelephonyWrapper
from core.services.crm_wrapper import CRMWrapper
from core.services.data_wrapper import DataWrapper # Ensure this is the webhook-based one
from core.communication.voice_handler import VoiceHandler
from core.agents.resource_manager import ResourceManager
from core.agents.acquisition_agent import AcquisitionAgent, DEFAULT_CLAY_SERVICE_AUTOMATION_CONFIG # For RM
from core.agents.sales_agent import SalesAgent
from core.automation.browser_automator_interface import BrowserAutomatorInterface
# from core.automation.multimodal_playwright_automator import MultiModalPlaywrightAutomator # Your production implementation
from core.database_setup import setup_supabase_tables, create_client as create_supabase_client_for_setup

# --- Mock Automator (Replace with your real one in production) ---
# This is provided so the server can run. For actual trial acquisition,
# you need to implement a robust BrowserAutomator (e.g., using Playwright).
class MockBrowserAutomatorForServer(BrowserAutomatorInterface):
    def __init__(self, llm_client: Optional[LLMClient] = None): # Added llm_client for consistency
        self.llm_client = llm_client # Unused in mock, but good for interface consistency
        logger.info("MockBrowserAutomatorForServer initialized.")

    async def setup_session(self, *args, **kwargs): logger.info("MockAutomator(Server): setup_session"); return True
    async def close_session(self, *args, **kwargs): logger.info("MockAutomator(Server): close_session")
    async def navigate_to_page(self, url: str, *args, **kwargs): logger.info(f"MockAutomator(Server): navigate_to_page {url}"); return True
    async def take_screenshot(self, *args, **kwargs): logger.info("MockAutomator(Server): take_screenshot"); return b"dummy_screenshot_bytes_content"
    async def fill_form_and_submit(self, form_selectors_and_values: Dict[str, str], submit_button_selector: str, *args, **kwargs):
        logger.info(f"MockAutomator(Server): fill_form_and_submit. Data: {form_selectors_and_values}, Submit: {submit_button_selector}"); return True
    async def check_success_condition(self, indicator: Dict[str, Any], *args, **kwargs):
        logger.info(f"MockAutomator(Server): check_success_condition with indicator: {indicator}"); return True # Assume success
    async def extract_resources_from_page(self, rules: List[Dict[str, Any]], *args, **kwargs):
        logger.info(f"MockAutomator(Server): extract_resources_from_page with rules: {rules}"); return {"api_key": "mock_server_extracted_key_123"}
    async def solve_captcha_if_present(self, *args, **kwargs): logger.info("MockAutomator(Server): solve_captcha_if_present"); return True
    async def full_signup_and_extract(self, service_name: str, signup_url: str, form_interaction_plan: List[Dict[str, Any]], signup_details_generated: Dict[str, Any], success_indicator: Dict[str, Any], resource_extraction_rules: List[Dict[str, Any]], captcha_config: Optional[Dict[str, Any]] = None, max_retries: int = 0):
        logger.info(f"MockAutomator(Server): full_signup_and_extract for {service_name} at {signup_url}")
        if service_name == "clay.com":
            return {"status": "success", "extracted_resources": {"api_key": "mock_clay_trial_key_from_server_automator"}}
        return {"status": "failed", "reason": "Mocked failure for other services by MockAutomator(Server)"}

logger = logging.getLogger("boutique_ai_server_orchestrator")

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Boutique AI Server (Level 45+ Orchestrator) initializing...")
    app.state = AppState()

    if config.SUPABASE_ENABLED:
        logger.info("Attempting Supabase database schema setup/verification...")
        try:
            app.state.supabase_client_for_setup = create_supabase_client_for_setup() # Uses config internally
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

    app.state.llm_client = LLMClient()
    app.state.fingerprint_generator = FingerprintGenerator(llm_client=app.state.llm_client)
    
    # PRODUCTION: Replace MockBrowserAutomatorForServer with your actual, robust implementation
    # from core.automation.multimodal_playwright_automator import MultiModalPlaywrightAutomator
    # app.state.browser_automator = MultiModalPlaywrightAutomator(llm_client=app.state.llm_client, headless=True)
    app.state.browser_automator = MockBrowserAutomatorForServer(llm_client=app.state.llm_client)
    if isinstance(app.state.browser_automator, MockBrowserAutomatorForServer):
        logger.warning("Lifespan: Using MOCK BrowserAutomator. Trial acquisition will be SIMULATED.")

    app.state.resource_manager = ResourceManager(
        llm_client=app.state.llm_client,
        fingerprint_generator=app.state.fingerprint_generator,
        browser_automator=app.state.browser_automator
    )
    app.state.data_wrapper = DataWrapper()
    app.state.crm_wrapper = CRMWrapper()
    app.state.telephony_wrapper = TelephonyWrapper()

    acq_criteria = {
        "niche": config.AGENT_TARGET_NICHE_DEFAULT,
        "clay_enrichment_table_webhook_url": config.get_env_var("CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY", required=False),
        "lead_source_type": config.get_env_var("ACQ_LEAD_SOURCE_TYPE", default="supabase_query", required=False),
        "lead_source_path": config.get_env_var("ACQ_LEAD_SOURCE_PATH", default="data/initial_leads.csv", required=False),
        "lead_source_csv_field_mapping": json.loads(config.get_env_var("ACQ_LEAD_SOURCE_CSV_MAPPING", default='{"company_name": "company_name", "domain": "domain", "email": "primary_contact_email_guess"}', required=False)),
        "supabase_pending_lead_status": config.get_env_var("ACQ_SUPABASE_PENDING_STATUS", default="New_Raw_Lead", required=False),
        "qualification_llm_score_threshold": config.get_int_env_var("ACQ_QUALIFICATION_THRESHOLD", default=7, required=False),
        "max_leads_to_process_per_cycle": config.ACQUISITION_AGENT_BATCH_SIZE,
        "run_interval_seconds": config.ACQUISITION_AGENT_RUN_INTERVAL_SECONDS,
        "batch_size": config.ACQUISITION_AGENT_BATCH_SIZE,
        "clay_service_automation_config": DEFAULT_CLAY_SERVICE_AUTOMATION_CONFIG # Example from AcqAgent
    }
    if not acq_criteria["clay_enrichment_table_webhook_url"]:
        logger.error("CRITICAL: CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY (or equivalent env var for acq_criteria) not set. Clay enrichment will fail.")
    
    app.state.acquisition_agent = AcquisitionAgent(
        agent_id="GlobalAcquisitionAgent_001",
        resource_manager=app.state.resource_manager, data_wrapper=app.state.data_wrapper,
        llm_client=app.state.llm_client, crm_wrapper=app.state.crm_wrapper,
        target_criteria=acq_criteria
    )
    await app.state.acquisition_agent.start()
    logger.info("AcquisitionAgent started in background.")

    yield

    logger.info("Boutique AI Server shutting down...")
    if app.state.acquisition_agent: await app.state.acquisition_agent.stop()
    
    active_calls = list(app.state.active_sales_agents.values())
    if active_calls:
        logger.info(f"Ending {len(active_calls)} active sales calls...")
        await asyncio.gather(*(agent.stop_call("Server Shutdown") for agent in active_calls), return_exceptions=True)

    await DataWrapper.close_session()
    if app.state.browser_automator and hasattr(app.state.browser_automator, 'close_session'):
        await app.state.browser_automator.close_session()
    logger.info("Boutique AI Server shutdown complete.")

app = FastAPI(title="Boutique AI Server & Orchestrator", version="1.2.1", lifespan=lifespan)

@app.post("/call_webhook", response_class=PlainTextResponse)
async def handle_twilio_call_webhook(request: Request):
    """
    Handles Twilio's webhook when an outbound call is answered or an inbound call is received.
    Responds with TwiML to start a media stream to our WebSocket.
    """
    if not app.state.telephony_wrapper:
        logger.error("/call_webhook: TelephonyWrapper not initialized!")
        raise HTTPException(status_code=503, detail="Telephony service unavailable")

    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    call_status = form_data.get("CallStatus")
    to_number = form_data.get("To")
    from_number = form_data.get("From")
    # Twilio also sends CustomParameters if you set them in client.calls.create()
    # custom_params = {k: v for k, v in form_data.items() if k.startswith("Custom_")}

    logger.info(f"[{call_sid}] Received /call_webhook. Status: {call_status}, From: {from_number}, To: {to_number}")

    if not call_sid:
        logger.error("Missing CallSid in Twilio webhook request.")
        raise HTTPException(status_code=400, detail="CallSid missing")

    # Determine prospect's phone number
    prospect_phone = from_number if to_number == config.TWILIO_PHONE_NUMBER else to_number
    if not prospect_phone:
        logger.error(f"[{call_sid}] Could not determine prospect phone number. From: {from_number}, To: {to_number}, Configured Twilio Num: {config.TWILIO_PHONE_NUMBER}")
        raise HTTPException(status_code=400, detail="Could not determine prospect phone")

    # Construct WebSocket URL using the host header from the incoming request for robustness
    ws_scheme = "wss" if request.url.scheme == "https" or config.BASE_WEBHOOK_URL.startswith("https://") else "ws"
    host = request.headers.get("host", f"localhost:{config.LOCAL_SERVER_PORT}")
    # URL Encode prospect_phone as it might contain '+'
    encoded_prospect_phone = prospect_phone.strip().replace('+', '%2B')
    ws_url = f"{ws_scheme}://{host}/call_ws?call_sid={call_sid}&prospect_phone={encoded_prospect_phone}"

    response = VoiceResponse()
    start_verb = Start()
    start_verb.stream(url=ws_url)
    response.append(start_verb)
    response.pause(length=1) # Brief pause for WebSocket connection setup
    logger.info(f"[{call_sid}] Responding to Twilio with TwiML <Stream> to: {ws_url}")
    return PlainTextResponse(str(response), media_type="application/xml")

@app.websocket("/call_ws")
async def websocket_call_endpoint(
    websocket: WebSocket,
    call_sid: str = Query(...),
    prospect_phone: str = Query(...) # FastAPI URL decodes query parameters
):
    await websocket.accept()
    logger.info(f"[{call_sid}] WebSocket accepted for prospect: {prospect_phone}. Initializing SalesAgent.")

    if not all([app.state.llm_client, app.state.telephony_wrapper, app.state.crm_wrapper, app.state.data_wrapper]):
        logger.error(f"[{call_sid}] Core services not initialized in app.state. Closing WebSocket.")
        await websocket.close(code=1011, reason="Server services not ready")
        return

    current_stream_sid: Optional[str] = None # Set by Twilio's 'start' media event

    async def send_audio_to_twilio_ws(csid: str, audio_chunk: bytes):
        if csid == call_sid and current_stream_sid and websocket.client_state == WebSocketState.CONNECTED:
            try:
                media_payload = base64.b64encode(audio_chunk).decode('utf-8')
                await websocket.send_json({"event": "media", "streamSid": current_stream_sid, "media": {"payload": media_payload}})
            except WebSocketDisconnect:
                logger.warning(f"[{call_sid}] WS disconnected while sending audio.")
                if call_sid in app.state.active_sales_agents:
                    app.state.active_sales_agents[call_sid].signal_call_ended_externally("WS Disconnected during TTS send")
            except RuntimeError as re: # Catches "send is called after close"
                 logger.warning(f"[{call_sid}] WS runtime error sending audio (likely already closing): {re}")
            except Exception as e:
                logger.error(f"[{call_sid}] Error sending audio via WS: {e}", exc_info=True)

    async def send_mark_to_twilio_ws(csid: str, mark_name: str):
        if csid == call_sid and current_stream_sid and websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json({"event": "mark", "streamSid": current_stream_sid, "mark": {"name": mark_name}})
                logger.info(f"[{call_sid}] Sent mark '{mark_name}' to Twilio.")
            except WebSocketDisconnect: logger.warning(f"[{call_sid}] WS disconnected sending mark.")
            except RuntimeError as re: logger.warning(f"[{call_sid}] WS runtime error sending mark: {re}")
            except Exception as e: logger.error(f"[{call_sid}] Error sending mark via WS: {e}", exc_info=True)

    voice_handler = VoiceHandler(transcript_callback=None, error_callback=None) # SalesAgent sets these

    pre_call_contact_info = await app.state.crm_wrapper.get_contact_info(identifier=prospect_phone, identifier_column="phone_number")
    if not pre_call_contact_info:
        logger.info(f"[{call_sid}] No existing contact for {prospect_phone}. Creating shell for call tracking.")
        shell_data = {"phone_number": prospect_phone, "status": "IncomingCall_NewContact", "source_info": f"Call_{call_sid}"}
        pre_call_contact_info = await app.state.crm_wrapper.upsert_contact(shell_data, "phone_number") or shell_data

    sales_agent = SalesAgent(
        agent_id=f"SalesAgent_{call_sid}", target_phone_number=prospect_phone,
        voice_handler=voice_handler, llm_client=app.state.llm_client,
        telephony_wrapper=app.state.telephony_wrapper, crm_wrapper=app.state.crm_wrapper,
        initial_prospect_data=pre_call_contact_info,
        send_audio_callback=send_audio_to_twilio_ws, send_mark_callback=send_mark_to_twilio_ws,
        on_call_complete_callback=lambda aid, status, hist, summary: logger.info(f"SERVER: Agent {aid} call complete. Status: {status}. Summary: {summary.get('final_call_outcome_category','N/A')}"),
        on_call_error_callback=lambda aid, err: logger.error(f"SERVER: Agent {aid} call error: {err}")
    )
    app.state.active_sales_agents[call_sid] = sales_agent

    try:
        while True:
            message = await websocket.receive_json()
            event = message.get("event")

            if event == "start":
                current_stream_sid = message.get("start", {}).get("streamSid")
                if not current_stream_sid:
                    logger.error(f"[{call_sid}] WS 'start' event missing streamSid. Message: {message}")
                    await sales_agent._handle_fatal_error("Missing streamSid in start event")
                    break
                logger.info(f"[{call_sid}] WS 'start' event. Stream SID: {current_stream_sid}. Starting SalesAgent call logic.")
                asyncio.create_task(sales_agent.start_sales_call(call_sid, current_stream_sid))
            elif event == "media":
                if not current_stream_sid:
                    logger.warning(f"[{call_sid}] Received 'media' before 'start' or after 'stop'. Ignoring.")
                    continue
                payload = message.get("media", {}).get("payload")
                if payload:
                    await sales_agent.handle_incoming_audio(base64.b64decode(payload))
            elif event == "stop":
                logger.info(f"[{call_sid}] WS 'stop' event received. Signaling agent call ended.")
                sales_agent.signal_call_ended_externally("WebSocket Stop Event")
                break # Exit WebSocket loop
            elif event == "mark":
                mark_name = message.get("mark", {}).get("name")
                logger.debug(f"[{call_sid}] Received WS 'mark': {mark_name}")
            elif event == "connected":
                logger.info(f"[{call_sid}] WS 'connected' event from Twilio (media stream established).")
            else:
                logger.debug(f"[{call_sid}] Unknown WS event: {event}, Data: {message}")
    except WebSocketDisconnect as e:
        logger.warning(f"[{call_sid}] WebSocket disconnected by client/Twilio. Code: {e.code}")
        if sales_agent.is_call_active:
            sales_agent.signal_call_ended_externally(f"WebSocket Disconnected (Code: {e.code})")
    except Exception as e:
        logger.error(f"[{call_sid}] Error in WebSocket handler: {e}", exc_info=True)
        if sales_agent.is_call_active:
            sales_agent.signal_call_ended_externally(f"WebSocket Handler Exception: {type(e).__name__}")
    finally:
        logger.info(f"[{call_sid}] Cleaning up WebSocket & SalesAgent resources for {prospect_phone}...")
        # Agent's own _cleanup_call should be triggered by signal_call_ended or lifecycle task.
        # We just remove it from the active list here.
        if call_sid in app.state.active_sales_agents:
            del app.state.active_sales_agents[call_sid]
        logger.info(f"[{call_sid}] WebSocket resources for {prospect_phone} cleaned up from server's perspective.")


@app.post("/webhooks/clay/enrichment_results")
async def handle_clay_enrichment_webhook(
    request: Request,
    x_callback_auth_token: Optional[str] = Header(None, alias=config.get_env_var("CLAY_CALLBACK_AUTH_HEADER_NAME", default="X-Callback-Auth-Token", required=False))
):
    configured_secret = config.get_env_var("CLAY_RESULTS_CALLBACK_SECRET_TOKEN", required=False)
    if configured_secret: # Only enforce if a secret is configured
        if not x_callback_auth_token:
            logger.warning("Clay webhook: Auth token header missing, but secret is configured.")
            raise HTTPException(status_code=401, detail="Unauthorized: Missing authentication token")
        if x_callback_auth_token != configured_secret:
            logger.warning(f"Clay webhook: Unauthorized access attempt (invalid token).")
            raise HTTPException(status_code=403, detail="Forbidden: Invalid authentication token")
    elif not configured_secret:
        logger.warning("Clay webhook: CLAY_RESULTS_CALLBACK_SECRET_TOKEN not configured. Webhook is INSECURE.")

    logger.info("Received POST on /webhooks/clay/enrichment_results")
    if not app.state.acquisition_agent:
        logger.error("Clay webhook: AcquisitionAgent not available in app.state!")
        raise HTTPException(status_code=503, detail="Acquisition service temporarily unavailable")

    try: payload = await request.json()
    except json.JSONDecodeError:
        logger.error("Clay webhook: Invalid JSON payload.")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    logger.debug(f"Clay webhook received payload (first 500 chars): {json.dumps(payload)[:500]}")
    correlation_id = payload.get("_correlation_id")

    if not correlation_id:
        logger.warning("Clay webhook: Payload missing '_correlation_id'. Cannot process. Payload: %s", payload)
        return FastAPIResponse(content={"status": "error", "message": "Payload missing _correlation_id"}, status_code=400)

    asyncio.create_task(app.state.acquisition_agent.handle_clay_enrichment_result(payload))
    logger.info(f"Clay enrichment result for correlation_id '{correlation_id}' queued for processing.")
    return FastAPIResponse(content={"status": "received", "message": "Result queued for processing."}, status_code=202)

@app.post("/admin/actions/initiate_call", tags=["Admin Actions"])
async def trigger_outbound_call_admin(target_number: str, crm_contact_id: Optional[str] = None):
    if not app.state.telephony_wrapper: raise HTTPException(status_code=503, detail="Telephony service unavailable.")
    if not target_number: raise HTTPException(status_code=400, detail="Missing 'target_number'.")
    logger.info(f"ADMIN: API request to initiate call to: {target_number}. CRM Contact ID: {crm_contact_id}")
    
    custom_params_for_twilio = {}
    if crm_contact_id: custom_params_for_twilio["crm_id"] = crm_contact_id
    
    call_sid = await app.state.telephony_wrapper.initiate_call(
        target_number=target_number,
        custom_parameters=custom_params_for_twilio if custom_params_for_twilio else None
    )
    if call_sid: return {"message": "Outbound call initiated.", "call_sid": call_sid}
    raise HTTPException(status_code=500, detail="Failed to initiate call.")

if __name__ == "__main__":
    import uvicorn
    # Ensure .env is loaded if running this script directly for development
    # This should be handled by config.py's load_dotenv() at import time.
    # from dotenv import load_dotenv
    # load_dotenv()

    logger.info(f"Starting Boutique AI Server Orchestrator on host 0.0.0.0 port {config.LOCAL_SERVER_PORT}")
    logger.info(f"Ensure your BASE_WEBHOOK_URL ({config.BASE_WEBHOOK_URL}) is publicly accessible.")
    logger.info(f"Twilio Call Webhook URL: {config.BASE_WEBHOOK_URL.rstrip('/')}/call_webhook")
    logger.info(f"Clay Enrichment Results Webhook URL: {config.BASE_WEBHOOK_URL.rstrip('/')}/webhooks/clay/enrichment_results")
    
    uvicorn.run(
        "server:app", # Points to the 'app' instance in this 'server.py' file
        host="0.0.0.0",
        port=config.LOCAL_SERVER_PORT,
        log_level=config.LOG_LEVEL.lower(),
        reload=config.get_bool_env_var("UVICORN_RELOAD", default=False) # Add UVICORN_RELOAD=true to .env for dev
    )
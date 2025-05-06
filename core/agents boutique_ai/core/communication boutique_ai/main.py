# main.py

import asyncio
import logging
import signal # For graceful shutdown
import json
import base64
import time # Added for unique agent ID generation
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

# --- Web Server Framework ---
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response, HTTPException
from fastapi.responses import PlainTextResponse

# --- Import core components ---
# Ensure config is imported first to set up logging and constants
import config
from core.agents.sales_agent import SalesAgent
from core.agents.resource_manager import ResourceManager # Supabase version
from core.communication.voice_handler import VoiceHandler
from core.services.llm_client import LLMClient # Includes caching
from core.services.telephony_wrapper import TelephonyWrapper
from core.services.crm_wrapper import CRMWrapper # Supabase version
from core.services.data_wrapper import DataWrapper # Functional Clay table interaction

# Configure logging (config.py sets the level and handlers)
logger = logging.getLogger(__name__)

# --- Orchestrator Class ---
class Orchestrator:
    """
    Main orchestrator for Boutique AI. Initializes services, manages agent lifecycles,
    handles webhooks/websockets via FastAPI, and controls graceful shutdown.
    Uses Supabase for CRM logging and resource state persistence.
    """

    def __init__(self):
        """ Initializes the orchestrator and its core components. """
        logger.info("Initializing Boutique AI Orchestrator...")
        try:
            # Initialize Core Services (Singletons)
            self.llm_client = LLMClient() # Includes caching
            self.telephony_wrapper = TelephonyWrapper()
            self.crm_wrapper = CRMWrapper() # Uses Supabase
            self.resource_manager = ResourceManager() # Uses Supabase
            self.data_wrapper = DataWrapper() # For Clay interaction
            self._configure_data_wrapper() # Set Clay API key

            # Agent & Connection Management
            self.active_agents: Dict[str, SalesAgent] = {} # Keyed by call_sid
            self.active_websockets: Dict[str, WebSocket] = {} # Keyed by call_sid

            self._shutdown_requested = asyncio.Event()
            logger.info("Orchestrator initialized successfully.")

        except Exception as e:
            logger.critical(f"CRITICAL ERROR during Orchestrator initialization: {e}", exc_info=True)
            raise RuntimeError("Orchestrator failed to initialize.") from e

    def _configure_data_wrapper(self):
        """ Retrieves Clay API key from ResourceManager and sets it on DataWrapper. """
        logger.info("Configuring DataWrapper with Clay API key...")
        # Note: get_clay_api_key is now async as it might interact with Supabase
        # We need to run this synchronously during init or handle it differently.
        # For simplicity during init, let's make ResourceManager's key retrieval sync
        # or call it from an async context later.
        # --- TEMPORARY SYNC CALL (Consider refactoring RM or init) ---
        try:
             # This is NOT ideal in an async app's main init, but simplest for now.
             # Better: Make __init__ async or configure data_wrapper in lifespan startup.
             clay_api_key = asyncio.run(self.resource_manager.get_clay_api_key())
             if clay_api_key:
                  self.data_wrapper.set_api_key(clay_api_key)
                  logger.info("DataWrapper configured with Clay API key.")
             else:
                  logger.warning("Could not retrieve Clay API key via ResourceManager. DataWrapper calls will fail.")
        except Exception as e:
             logger.error(f"Error configuring DataWrapper during init: {e}")
        # --- END TEMPORARY SYNC CALL ---


    # --- Agent Lifecycle Callbacks ---
    async def handle_agent_completion(self, agent_id: str, final_status: str, conversation_history: List[Dict[str, str]]):
        """ Callback from SalesAgent on successful call completion. """
        logger.info(f"Orchestrator: Agent {agent_id} completed call. Final Status: {final_status}")
        # Find call_sid associated with this agent_id for cleanup
        call_sid_to_clean = None
        for sid, agent in self.active_agents.items():
             if agent.agent_id == agent_id:
                  call_sid_to_clean = sid
                  break
        if call_sid_to_clean:
             self._cleanup_websocket_resources(call_sid_to_clean)
        else:
             logger.warning(f"Could not find active call_sid for completed agent {agent_id} during callback.")
        # TODO: Add logic for post-call analysis or follow-up triggers

    async def handle_agent_error(self, agent_id: str, error_message: str):
        """ Callback from SalesAgent on fatal error during call. """
        logger.error(f"Orchestrator: Agent {agent_id} reported fatal error: {error_message}")
        # Find call_sid for cleanup
        call_sid_to_clean = None
        for sid, agent in self.active_agents.items():
             if agent.agent_id == agent_id:
                  call_sid_to_clean = sid
                  break
        if call_sid_to_clean:
             self._cleanup_websocket_resources(call_sid_to_clean)
        else:
             logger.warning(f"Could not find active call_sid for errored agent {agent_id} during callback.")
        # TODO: Implement alerting or specific error handling logic

    def _cleanup_websocket_resources(self, call_sid: str):
         """ Removes agent and websocket references after a call ends. """
         logger.debug(f"[{call_sid}] Orchestrator cleaning up resources.")
         if call_sid in self.active_agents:
              del self.active_agents[call_sid]
              logger.info(f"[{call_sid}] Removed agent instance from active agents.")
         if call_sid in self.active_websockets:
              # We don't close the websocket here, FastAPI/Uvicorn handles that on disconnect/stop event
              del self.active_websockets[call_sid]
              logger.info(f"[{call_sid}] Removed websocket reference from active connections.")


    # --- WebSocket Helper Callbacks (Passed to Agent) ---
    async def send_audio_via_websocket(self, call_sid: str, audio_chunk: bytes):
        """ Safely sends TTS audio chunk back over the correct WebSocket. """
        websocket = self.active_websockets.get(call_sid)
        agent = self.active_agents.get(call_sid)
        if websocket and agent and agent.stream_sid:
            try:
                encoded_audio = base64.b64encode(audio_chunk).decode('utf-8')
                media_message = {"event": "media", "streamSid": agent.stream_sid, "media": {"payload": encoded_audio}}
                await websocket.send_json(media_message)
            except (WebSocketDisconnect, ConnectionResetError, RuntimeError) as e:
                 logger.warning(f"[{call_sid}] WebSocket error during send_audio: {type(e).__name__}")
                 if agent: agent.signal_call_ended_externally(f"WebSocket Error during TTS: {type(e).__name__}")
            except Exception as e: logger.error(f"[{call_sid}] Unexpected error sending audio via WebSocket: {e}", exc_info=True)
        elif not websocket: logger.warning(f"[{call_sid}] Cannot send audio: WebSocket not found.")
        elif not agent: logger.warning(f"[{call_sid}] Cannot send audio: Agent not found.")
        elif not agent.stream_sid: logger.warning(f"[{call_sid}] Cannot send audio: Agent missing stream_sid.")

    async def send_mark_via_websocket(self, call_sid: str, mark_name: str):
        """ Safely sends a Mark message over the correct WebSocket. """
        websocket = self.active_websockets.get(call_sid)
        agent = self.active_agents.get(call_sid)
        if websocket and agent and agent.stream_sid:
            try:
                mark_message = {"event": "mark", "streamSid": agent.stream_sid, "mark": {"name": mark_name}}
                await websocket.send_json(mark_message)
                logger.info(f"[{call_sid}] Sent mark '{mark_name}' via WebSocket.")
            except (WebSocketDisconnect, ConnectionResetError, RuntimeError) as e:
                 logger.warning(f"[{call_sid}] WebSocket error during send_mark: {type(e).__name__}")
                 if agent: agent.signal_call_ended_externally(f"WebSocket Error during Mark: {type(e).__name__}")
            except Exception as e: logger.error(f"[{call_sid}] Unexpected error sending mark via WebSocket: {e}", exc_info=True)
        elif not websocket: logger.warning(f"[{call_sid}] Cannot send mark: WebSocket not found.")
        elif not agent: logger.warning(f"[{call_sid}] Cannot send mark: Agent not found.")
        elif not agent.stream_sid: logger.warning(f"[{call_sid}] Cannot send mark: Agent missing stream_sid.")


    # --- Public Method to Trigger a Call ---
    async def initiate_new_call(self, target_phone: str, agent_id_prefix: str = "Agent"):
        """ Triggers the telephony provider to initiate an outbound call. """
        if self._shutdown_requested.is_set():
             logger.warning("Shutdown requested, not initiating new call.")
             return None
        intended_agent_id = f"{agent_id_prefix}_{int(time.time()*1000)}_{target_phone[-4:]}"
        logger.info(f"Orchestrator: Requesting new call via TelephonyWrapper for intended Agent {intended_agent_id} to {target_phone}.")
        call_sid = await self.telephony_wrapper.initiate_call(target_number=target_phone)
        if call_sid:
            logger.info(f"Orchestrator: Call initiated by TelephonyWrapper. SID: {call_sid}. Agent will be created upon WebSocket connection.")
            return call_sid
        else:
            logger.error(f"Orchestrator: Failed to initiate call for intended Agent {intended_agent_id} via TelephonyWrapper.")
            return None

    # --- Web Server Integration (FastAPI Endpoints & WebSocket) ---
    def setup_fastapi_routes(self, app: FastAPI):
        """ Adds webhook and WebSocket endpoints to the FastAPI app instance. """
        logger.info("Setting up FastAPI routes for Orchestrator...")

        @app.post("/call_webhook", response_class=PlainTextResponse)
        async def handle_call_webhook(request: Request):
            """ Handles Twilio webhook when call is answered. Returns TwiML to start media stream. """
            form_data = await request.form()
            call_sid = form_data.get("CallSid", "UnknownSID")
            call_status = form_data.get("CallStatus")
            logger.info(f"[{call_sid}] Received /call_webhook request. Status: {call_status}")

            # Proceed only if call is connecting
            if call_status not in ['in-progress', 'ringing', 'answered']:
                 logger.warning(f"[{call_sid}] Webhook received with non-connecting status '{call_status}'. Ignoring.")
                 return PlainTextResponse("<Response></Response>", media_type='text/xml')

            # Determine WebSocket scheme (ws vs wss)
            scheme = "wss" if config.BASE_WEBHOOK_URL.startswith("https://") else "ws"
            host = request.headers.get("host", f"localhost:{config.LOCAL_SERVER_PORT}")
            ws_url = f"{scheme}://{host}/call_ws"

            from twilio.twiml.voice_response import VoiceResponse, Start
            response = VoiceResponse(); start = Start(); start.stream(url=ws_url)
            response.append(start); response.pause(length=2)
            logger.info(f"[{call_sid}] Responding to webhook with TwiML <Stream> pointing to: {ws_url}")
            return PlainTextResponse(str(response), media_type='text/xml')

        @app.websocket("/call_ws")
        async def websocket_endpoint(websocket: WebSocket):
            """ Handles bidirectional audio stream via WebSocket with Twilio media streams. """
            await websocket.accept()
            logger.info("WebSocket connection accepted from telephony provider.")
            call_sid: Optional[str] = None; stream_sid: Optional[str] = None; agent: Optional[SalesAgent] = None
            try:
                while True: # Main loop processing messages from Twilio
                    message_str = await websocket.receive_text()
                    message = json.loads(message_str)
                    event = message.get("event")

                    if event == "start":
                        stream_sid = message["start"]["streamSid"]
                        call_sid = message["start"]["callSid"]
                        logger.info(f"[{call_sid}] Received WebSocket 'start' event. Stream SID: {stream_sid}. Creating/Associating Agent...")
                        if call_sid in self.active_agents:
                             logger.warning(f"[{call_sid}] 'start' event for already active call. Re-associating WS.")
                             agent = self.active_agents[call_sid]
                             self.active_websockets[call_sid] = websocket
                        else:
                             agent_id = f"Agent_{call_sid}"
                             self.active_websockets[call_sid] = websocket
                             voice_handler = VoiceHandler(transcript_callback=None, error_callback=None)
                             agent = SalesAgent(
                                 agent_id=agent_id,
                                 target_phone_number=message["start"].get("customParameters", {}).get("target_phone", "Unknown"),
                                 voice_handler=voice_handler, llm_client=self.llm_client,
                                 telephony_wrapper=self.telephony_wrapper, crm_wrapper=self.crm_wrapper,
                                 on_call_complete_callback=self.handle_agent_completion,
                                 on_call_error_callback=self.handle_agent_error,
                                 send_audio_callback=self.send_audio_via_websocket,
                                 send_mark_callback=self.send_mark_via_websocket
                             )
                             self.active_agents[call_sid] = agent
                             # Start agent logic in background task
                             asyncio.create_task(agent.start_sales_call(call_sid, stream_sid))

                    elif event == "media":
                        if call_sid and call_sid in self.active_agents:
                            audio_chunk = base64.b64decode(message["media"]["payload"])
                            await self.active_agents[call_sid].handle_incoming_audio(audio_chunk)
                        # else: logger.warning(f"Received 'media' event but no active agent found for SID '{call_sid}'.") # Can be noisy

                    elif event == "stop":
                        logger.info(f"[{call_sid}] Received WebSocket 'stop' event. Call ending.")
                        if call_sid and call_sid in self.active_agents:
                             self.active_agents[call_sid].signal_call_ended_externally("Telephony Stop Event")
                        break # Exit WebSocket loop

                    elif event == "mark": logger.debug(f"[{call_sid}] Received WebSocket 'mark': {message.get('mark', {}).get('name')}")
                    elif event == "connected": logger.info(f"[{call_sid}] Received WebSocket 'connected' event.")
                    else: logger.debug(f"[{call_sid}] Received unknown WebSocket event: {event}")

            except WebSocketDisconnect as e: logger.warning(f"[{call_sid}] WebSocket disconnected: {e.code}")
            except Exception as e: logger.error(f"[{call_sid}] Error in WebSocket handler: {e}", exc_info=True)
            finally:
                logger.info(f"[{call_sid}] Cleaning up WebSocket connection resources...")
                # Ensure agent cleanup is triggered if it didn't finish normally
                if call_sid and call_sid in self.active_agents:
                     agent_instance = self.active_agents.get(call_sid)
                     if agent_instance and agent_instance.is_call_active:
                          logger.warning(f"[{call_sid}] WebSocket closed but agent still active. Forcing cleanup.")
                          asyncio.create_task(agent_instance._cleanup_call("WebSocket Closed Abnormally", None))
                # Clean up orchestrator's references
                self._cleanup_websocket_resources(call_sid)


    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """ FastAPI lifespan context manager for startup and shutdown. """
        logger.info("Orchestrator lifespan startup...")
        yield # Server runs here
        logger.info("Orchestrator lifespan shutdown...")
        await self.shutdown() # Trigger graceful shutdown

    async def shutdown(self):
        """ Coordinates the graceful shutdown of the orchestrator and active agents. """
        logger.info("Initiating graceful shutdown sequence...")
        self._shutdown_requested.set()
        active_agent_instances = list(self.active_agents.values())
        if active_agent_instances:
            logger.info(f"Requesting stop for {len(active_agent_instances)} active agent(s)...")
            stop_tasks = [asyncio.create_task(agent.stop_call("Orchestrator Shutdown")) for agent in active_agent_instances]
            await asyncio.wait(stop_tasks, timeout=5.0)
            logger.info(f"Waiting up to 30s for agent tasks to complete...")
            lifecycle_tasks = [agent.call_lifecycle_task for agent in active_agent_instances if agent.call_lifecycle_task and not agent.call_lifecycle_task.done()]
            if lifecycle_tasks:
                done, pending = await asyncio.wait(lifecycle_tasks, timeout=30.0)
                logger.info(f"{len(done)} agent tasks completed gracefully.")
                if pending:
                    logger.warning(f"{len(pending)} agent tasks did not complete in time. Cancelling...")
                    for task in pending: task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
        logger.info("Orchestrator shutdown sequence complete.")

    def request_shutdown_signal(self):
        """ Signals the orchestrator to begin graceful shutdown (for OS signal handlers). """
        if not self._shutdown_requested.is_set():
            logger.info("Shutdown requested via signal.")
            self._shutdown_requested.set()

# --- Global Orchestrator Instance & FastAPI App ---
orchestrator: Optional[Orchestrator] = None
app: Optional[FastAPI] = None
try:
    orchestrator = Orchestrator()
    app = FastAPI(
        title="Boutique AI Orchestrator", version="0.1.0",
        lifespan=orchestrator.lifespan # Use orchestrator's lifespan manager
    )
    orchestrator.setup_fastapi_routes(app)
except Exception as init_e:
     logger.critical(f"Failed during global Orchestrator/FastAPI setup: {init_e}", exc_info=True)
     orchestrator = None
     app = None

# --- Manual Call Initiation Endpoint (Example) ---
if app: # Only add endpoint if app initialized
    @app.post("/call")
    async def make_call(target_number: str):
         """ Example API endpoint to trigger an outbound call. """
         if not orchestrator: raise HTTPException(status_code=503, detail="Orchestrator not available.")
         if not target_number: raise HTTPException(status_code=400, detail="Missing 'target_number' query parameter.")
         logger.info(f"Received API request to call: {target_number}")
         call_sid = await orchestrator.initiate_new_call(target_phone=target_number)
         if call_sid: return {"message": "Call initiated successfully", "call_sid": call_sid}
         else: raise HTTPException(status_code=500, detail="Failed to initiate call via telephony provider.")

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting Boutique AI Orchestrator Server (main.py)...")
    if not app or not orchestrator:
         print("\nCRITICAL ERROR: Orchestrator or FastAPI app failed to initialize.")
         exit(1)
    if not config.TWILIO_ACCOUNT_SID or not config.DEEPGRAM_API_KEY or not config.OPENROUTER_API_KEY or not config.BASE_WEBHOOK_URL:
         print("\nCRITICAL ERROR: Essential API keys or BASE_WEBHOOK_URL missing.")
         exit(1)

    # Setup signal handlers
    def signal_handler(sig, frame):
        if orchestrator:
             loop = asyncio.get_event_loop()
             if loop.is_running(): loop.call_soon_threadsafe(orchestrator.request_shutdown_signal)
             else: orchestrator.request_shutdown_signal() # If loop stopped
        else: exit(1)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the Uvicorn server
    import uvicorn
    print(f"Attempting to start server on 0.0.0.0:{config.LOCAL_SERVER_PORT}")
    print(f"Ensure BASE_WEBHOOK_URL in .env ({config.BASE_WEBHOOK_URL}) is accessible.")
    try:
        uvicorn.run(
            "main:app", # Point uvicorn to the app instance in this file
            host="0.0.0.0",
            port=config.LOCAL_SERVER_PORT,
            log_level=config.LOG_LEVEL.lower(),
            reload=False # Important: Set reload=False for production/stable testing
        )
    except Exception as e:
         logger.critical(f"Failed to start Uvicorn server: {e}", exc_info=True)
         exit(1)
    finally:
         print("Boutique AI Orchestrator Server stopped.")


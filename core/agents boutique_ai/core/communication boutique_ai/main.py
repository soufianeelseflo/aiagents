# main.py

import asyncio
import logging
import signal # For graceful shutdown
import json
import base64
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

# --- Web Server Framework ---
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response, HTTPException
from fastapi.responses import PlainTextResponse

# --- Import core components ---
import config # Load config early (ensures logging is set up)
from core.agents.sales_agent import SalesAgent
from core.agents.resource_manager import ResourceManager
from core.communication.voice_handler import VoiceHandler
from core.services.llm_client import LLMClient
from core.services.telephony_wrapper import TelephonyWrapper
from core.services.crm_wrapper import CRMWrapper # Using local logging version
from core.services.data_wrapper import DataWrapper # For Clay table interaction
# Proxy wrapper might be used directly by ResourceManager or AcquisitionAgent
# from core.services.proxy_manager_wrapper import ProxyManagerWrapper

# Configure logging (config.py already sets up basic logging)
logger = logging.getLogger(__name__)

# --- Global State (Managed within Orchestrator instance) ---
# Avoid true global variables; pass orchestrator instance where needed.

class Orchestrator:
    """
    Main orchestrator for Boutique AI. Initializes services, manages agent lifecycles,
    handles webhooks/websockets via FastAPI, and controls graceful shutdown.
    """

    def __init__(self):
        """ Initializes the orchestrator and its core components. """
        logger.info("Initializing Boutique AI Orchestrator...")
        try:
            # --- Initialize Core Services (Singletons) ---
            self.llm_client = LLMClient()
            self.telephony_wrapper = TelephonyWrapper()
            self.crm_wrapper = CRMWrapper() # Uses local file logging
            self.resource_manager = ResourceManager()
            self.data_wrapper = DataWrapper() # For Clay interaction

            # --- Agent & Connection Management ---
            # Stores active agents, keyed by call_sid for easy lookup from WebSocket
            self.active_agents: Dict[str, SalesAgent] = {}
            # Stores active WebSocket connections, keyed by call_sid
            self.active_websockets: Dict[str, WebSocket] = {}

            self._shutdown_requested = asyncio.Event() # Event to signal shutdown

            logger.info("Orchestrator initialized successfully.")

        except Exception as e:
            logger.critical(f"CRITICAL ERROR during Orchestrator initialization: {e}", exc_info=True)
            raise RuntimeError("Orchestrator failed to initialize.") from e

    # --- Agent Lifecycle Callbacks ---
    async def handle_agent_completion(self, agent_id: str, final_status: str, conversation_history: List[Dict[str, str]]):
        """ Callback when a SalesAgent completes its call successfully. """
        logger.info(f"Orchestrator: Agent {agent_id} completed call with status: {final_status}")
        # Find call_sid associated with this agent_id if needed (might require mapping)
        # For now, assume cleanup is handled based on call_sid elsewhere
        # Log conversation or trigger follow-up actions here
        # logger.debug(f"Agent {agent_id} Conv History:\n{json.dumps(conversation_history, indent=2)}")
        # No need to call cleanup here, agent's own cleanup triggers this

    async def handle_agent_error(self, agent_id: str, error_message: str):
        """ Callback when a SalesAgent encounters a fatal error. """
        logger.error(f"Orchestrator: Agent {agent_id} reported fatal error: {error_message}")
        # Implement error handling strategy (e.g., alert, retry logic?)
        # Cleanup is handled by the agent's error path

    # --- WebSocket Helper Callbacks (Passed to Agent) ---
    async def send_audio_via_websocket(self, call_sid: str, audio_chunk: bytes):
        """ Sends TTS audio chunk back over the correct WebSocket. """
        websocket = self.active_websockets.get(call_sid)
        if websocket:
            try:
                # Format according to Twilio <Stream> Media message format
                # https://www.twilio.com/docs/voice/twiml/stream#message-media
                encoded_audio = base64.b64encode(audio_chunk).decode('utf-8')
                media_message = {
                    "event": "media",
                    "streamSid": self.active_agents[call_sid].stream_sid, # Get streamSid from agent state
                    "media": {
                        "payload": encoded_audio
                    }
                }
                await websocket.send_json(media_message)
                # logger.debug(f"[{call_sid}] Sent TTS audio chunk ({len(audio_chunk)} bytes) via WebSocket.")
            except WebSocketDisconnect:
                 logger.warning(f"[{call_sid}] WebSocket disconnected while trying to send audio.")
                 # Agent needs to handle this, potentially signal call end
                 if call_sid in self.active_agents:
                      self.active_agents[call_sid].signal_call_ended_externally("WebSocket Disconnected during TTS")
            except Exception as e:
                logger.error(f"[{call_sid}] Error sending audio via WebSocket: {e}", exc_info=True)
        else:
            logger.warning(f"[{call_sid}] Cannot send audio: WebSocket not found for active call.")

    async def send_mark_via_websocket(self, call_sid: str, mark_name: str):
        """ Sends a Mark message over the correct WebSocket (signals end of TTS). """
        websocket = self.active_websockets.get(call_sid)
        if websocket:
            try:
                # Format according to Twilio <Stream> Mark message format
                # https://www.twilio.com/docs/voice/twiml/stream#message-mark
                mark_message = {
                    "event": "mark",
                    "streamSid": self.active_agents[call_sid].stream_sid,
                    "mark": {
                        "name": mark_name
                    }
                }
                await websocket.send_json(mark_message)
                logger.info(f"[{call_sid}] Sent mark '{mark_name}' via WebSocket.")
            except WebSocketDisconnect:
                 logger.warning(f"[{call_sid}] WebSocket disconnected while trying to send mark.")
                 if call_sid in self.active_agents:
                      self.active_agents[call_sid].signal_call_ended_externally("WebSocket Disconnected during Mark")
            except Exception as e:
                logger.error(f"[{call_sid}] Error sending mark via WebSocket: {e}", exc_info=True)
        else:
            logger.warning(f"[{call_sid}] Cannot send mark: WebSocket not found for active call.")

    # --- Public Method to Start a Call ---
    async def initiate_new_call(self, target_phone: str, agent_id_prefix: str = "Agent"):
        """ Initiates the process of creating and starting a new sales call. """
        if self._shutdown_requested.is_set():
             logger.warning("Shutdown requested, not initiating new call.")
             return None

        agent_id = f"{agent_id_prefix}_{int(time.time()*1000)}_{target_phone[-4:]}" # Create unique ID
        logger.info(f"Orchestrator: Requesting new call via TelephonyWrapper for Agent {agent_id} to {target_phone}.")

        # Call initiation is now just handled by TelephonyWrapper.
        # The actual agent creation and start happens when the /call_webhook is hit
        # and the /call_ws connection is established.
        # We just need to trigger the call.
        call_sid = await self.telephony_wrapper.initiate_call(target_number=target_phone)

        if call_sid:
            logger.info(f"Orchestrator: Call initiated by TelephonyWrapper. SID: {call_sid}. Agent {agent_id} will be created on connection.")
            # We don't create the agent instance here anymore, it happens on WebSocket connect.
            return agent_id # Return the intended agent ID for tracking if needed
        else:
            logger.error(f"Orchestrator: Failed to initiate call for Agent {agent_id} via TelephonyWrapper.")
            return None

    # --- Web Server Integration (FastAPI Endpoints) ---

    def setup_fastapi_routes(self, app: FastAPI):
        """ Adds webhook and WebSocket endpoints to the FastAPI app. """
        logger.info("Setting up FastAPI routes for Orchestrator...")

        @app.post("/call_webhook", response_class=PlainTextResponse)
        async def handle_call_webhook(request: Request):
            """ Handles incoming webhook from Twilio when the call is answered. Returns TwiML. """
            # This endpoint tells Twilio to connect the call audio to our WebSocket endpoint.
            # Extract call SID for logging/context if needed (comes in form data)
            form_data = await request.form()
            call_sid = form_data.get("CallSid", "UnknownSID")
            logger.info(f"[{call_sid}] Received /call_webhook request from Twilio.")

            # Construct the WebSocket URL dynamically using the request's base URL
            # This assumes the server is running behind a proxy (like ngrok) that sets X-Forwarded-Proto
            scheme = request.headers.get("x-forwarded-proto", "ws") # Default to ws, use wss if behind TLS proxy
            host = request.headers.get("host", f"localhost:{config.LOCAL_SERVER_PORT}")
            ws_url = f"{scheme}://{host}/call_ws"

            # Generate TwiML response using Twilio library (minimal SDK use is okay here)
            from twilio.twiml.voice_response import VoiceResponse, Start # Import locally
            response = VoiceResponse()
            start = Start()
            start.stream(url=ws_url)
            response.append(start)
            # Add a pause to allow WebSocket connection and potentially initial agent processing/greeting
            response.pause(length=2) # Pause for 2 seconds
            logger.info(f"[{call_sid}] Responding to Twilio webhook with TwiML <Stream> pointing to: {ws_url}")
            # Return TwiML as XML string
            return PlainTextResponse(str(response), media_type='text/xml')

        @app.websocket("/call_ws")
        async def websocket_endpoint(websocket: WebSocket):
            """ Handles the bidirectional audio stream via WebSocket with Twilio. """
            await websocket.accept()
            logger.info("WebSocket connection accepted.")
            call_sid = None
            stream_sid = None
            agent: Optional[SalesAgent] = None

            try:
                while True:
                    message_str = await websocket.receive_text()
                    message = json.loads(message_str)
                    event = message.get("event")

                    if event == "connected":
                        logger.info("Received 'connected' event from Twilio.")
                        # This is just an informational event from Twilio

                    elif event == "start":
                        stream_sid = message["start"]["streamSid"]
                        call_sid = message["start"]["callSid"]
                        # --- Agent Creation/Association ---
                        # This is where we create the agent instance associated with this call
                        logger.info(f"[{call_sid}] Received 'start' event. Stream SID: {stream_sid}. Creating/Associating Agent...")
                        # Create a unique agent ID based on call SID
                        agent_id = f"Agent_{call_sid}"
                        # Store WebSocket connection
                        self.active_websockets[call_sid] = websocket
                        # Create Voice Handler instance for this call
                        # Pass the necessary callbacks from the orchestrator
                        voice_handler = VoiceHandler(
                            transcript_callback=None, # Agent will set this
                            error_callback=None       # Agent will set this
                        )
                        # Create SalesAgent instance
                        agent = SalesAgent(
                            agent_id=agent_id,
                            target_phone_number=message["start"].get("customParameters", {}).get("target_phone", "Unknown"), # Get target from custom params if sent, else default
                            voice_handler=voice_handler,
                            llm_client=self.llm_client,
                            telephony_wrapper=self.telephony_wrapper,
                            crm_wrapper=self.crm_wrapper,
                            on_call_complete_callback=self.handle_agent_completion,
                            on_call_error_callback=self.handle_agent_error,
                            send_audio_callback=self.send_audio_via_websocket,
                            send_mark_callback=self.send_mark_via_websocket
                        )
                        self.active_agents[call_sid] = agent
                        # Start the agent's logic now that connection is established
                        asyncio.create_task(agent.start_sales_call(call_sid, stream_sid))
                        # ---------------------------------

                    elif event == "media":
                        if call_sid and call_sid in self.active_agents:
                            # Forward audio chunk to the correct agent instance
                            audio_chunk = base64.b64decode(message["media"]["payload"])
                            await self.active_agents[call_sid].handle_incoming_audio(audio_chunk)
                        else:
                             logger.warning(f"Received 'media' event but no active agent found for inferred SID '{call_sid}'.")

                    elif event == "stop":
                        logger.info(f"[{call_sid}] Received 'stop' event from Twilio. Call ending.")
                        if call_sid and call_sid in self.active_agents:
                             # Signal the agent that the call was stopped externally
                             self.active_agents[call_sid].signal_call_ended_externally("Twilio Stop Event")
                        break # Exit the WebSocket loop

                    elif event == "mark":
                        mark_name = message.get("mark", {}).get("name", "Unknown Mark")
                        logger.debug(f"[{call_sid}] Received 'mark' event: {mark_name}")
                        # Agent logic might use this, but orchestrator likely ignores it

                    else:
                        logger.debug(f"[{call_sid}] Received unknown WebSocket event type: {event}")
                        logger.debug(f"Full message: {message}")

            except WebSocketDisconnect:
                logger.info(f"[{call_sid}] WebSocket disconnected.")
                if call_sid and call_sid in self.active_agents:
                     self.active_agents[call_sid].signal_call_ended_externally("WebSocket Disconnected")
            except Exception as e:
                logger.error(f"[{call_sid}] Error in WebSocket handler: {e}", exc_info=True)
                if call_sid and call_sid in self.active_agents:
                     # Signal agent about the error if possible
                     await self.active_agents[call_sid]._handle_fatal_error(f"WebSocket Error: {e}")
            finally:
                logger.info(f"[{call_sid}] Cleaning up WebSocket connection resources...")
                if call_sid:
                    # Remove agent and websocket references
                    if call_sid in self.active_agents:
                         # Ensure agent cleanup runs if not already triggered
                         if self.active_agents[call_sid].is_call_active:
                              logger.warning(f"[{call_sid}] Forcing agent cleanup due to WebSocket closure.")
                              # Use create_task to avoid blocking cleanup
                              asyncio.create_task(self.active_agents[call_sid]._cleanup_call("WebSocket Closed", None))
                         del self.active_agents[call_sid]
                    if call_sid in self.active_websockets:
                         del self.active_websockets[call_sid]
                logger.info(f"[{call_sid}] WebSocket resources cleaned up.")


    async def run_server(self):
        """ Runs the FastAPI web server. """
        import uvicorn # Import uvicorn here

        # Create FastAPI app instance within the async context if needed,
        # or define it globally if simpler for this structure.
        # Defining it globally for simplicity here:
        app = FastAPI(
             title="Boutique AI Orchestrator",
             version="0.1.0",
             # Define lifespan context manager for startup/shutdown events
             lifespan=self.lifespan
        )
        self.setup_fastapi_routes(app)

        server_config = uvicorn.Config(
            app,
            host="0.0.0.0", # Listen on all interfaces
            port=config.LOCAL_SERVER_PORT,
            log_level=config.LOG_LEVEL.lower(), # Use log level from config
            # Disable uvicorn's default access logs if desired, use FastAPI middleware instead if needed
            # access_log=False,
        )
        server = uvicorn.Server(server_config)

        logger.info(f"Starting FastAPI server on port {config.LOCAL_SERVER_PORT}...")
        # Run the server within the current asyncio event loop
        # This replaces asyncio.run(main_async()) in the original __main__ block
        await server.serve()
        # Code here will run after the server is stopped (e.g., by shutdown signal)
        logger.info("FastAPI server stopped.")


    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """ FastAPI lifespan context manager for startup and shutdown events. """
        logger.info("Orchestrator lifespan startup...")
        # --- Add any startup tasks here ---
        # Example: Pre-load resources, connect to databases (if any)
        # --- End startup tasks ---
        yield # Server runs here
        logger.info("Orchestrator lifespan shutdown...")
        # --- Add graceful shutdown tasks here ---
        await self.shutdown()
        # --- End shutdown tasks ---

    async def shutdown(self):
        """ Coordinates the graceful shutdown of the orchestrator and agents. """
        logger.info("Initiating graceful shutdown sequence...")
        self._shutdown_requested.set() # Signal any long-running loops to stop

        # Stop launching new agents (already handled by checking _shutdown_requested)

        # Wait for active agent tasks to complete (with a timeout)
        active_agent_instances = list(self.active_agents.values()) # Get current agents
        if active_agent_instances:
            logger.info(f"Requesting stop for {len(active_agent_instances)} active agent(s)...")
            stop_tasks = [asyncio.create_task(agent.stop_call("Orchestrator Shutdown")) for agent in active_agent_instances]
            await asyncio.wait(stop_tasks, timeout=5.0) # Give agents a moment to react

            logger.info(f"Waiting up to 30s for agent tasks to complete...")
            lifecycle_tasks = [agent.call_lifecycle_task for agent in active_agent_instances if agent.call_lifecycle_task]
            if lifecycle_tasks:
                done, pending = await asyncio.wait(lifecycle_tasks, timeout=30.0)
                logger.info(f"{len(done)} agent tasks completed gracefully.")
                if pending:
                    logger.warning(f"{len(pending)} agent tasks did not complete in time. Cancelling...")
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True) # Wait for cancellations
            else:
                 logger.info("No active agent lifecycle tasks were being tracked.")
        else:
            logger.info("No active agents found during shutdown.")

        # --- Add cleanup for other resources if needed ---
        # e.g., close database connections

        logger.info("Orchestrator shutdown sequence complete.")

    def request_shutdown_signal(self):
        """ Signals the orchestrator to begin graceful shutdown (for signal handlers). """
        logger.info("Shutdown requested via signal.")
        self._shutdown_requested.set()


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Boutique AI Orchestrator (main.py)...")
    # Basic check for essential config before starting loop
    if not config.TWILIO_ACCOUNT_SID or not config.DEEPGRAM_API_KEY or not config.OPENROUTER_API_KEY or not config.BASE_WEBHOOK_URL:
         print("\nCRITICAL ERROR: Essential API keys or BASE_WEBHOOK_URL missing in config.")
         print("Please ensure .env file is populated correctly.")
         exit(1)

    orchestrator = None # Define orchestrator in outer scope for signal handler
    try:
        orchestrator = Orchestrator()

        # --- Setup Signal Handlers ---
        # Must be done *before* starting the event loop if using loop.add_signal_handler
        # Using signal.signal might be more cross-platform compatible from main thread
        def signal_handler(sig, frame):
             if orchestrator:
                  # Schedule shutdown in the running event loop
                  asyncio.get_event_loop().call_soon_threadsafe(orchestrator.request_shutdown_signal)
             else:
                  print("Orchestrator not initialized, exiting immediately.")
                  exit(1)

        signal.signal(signal.SIGINT, signal_handler) # Handle Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler) # Handle termination signal

        # --- Run the Server ---
        # The run_server method now blocks until shutdown
        asyncio.run(orchestrator.run_server())

    except RuntimeError as e:
         # Catch errors during orchestrator init
         print(f"Failed to start Orchestrator: {e}")
         exit(1)
    except KeyboardInterrupt:
        # This might not be reached if signal handler works, but good fallback
        logger.info("KeyboardInterrupt received, initiating shutdown...")
        if orchestrator:
             # Try graceful shutdown if instance exists
             asyncio.run(orchestrator.shutdown())
    except Exception as e:
         logger.critical(f"Unhandled exception in main execution: {e}", exc_info=True)
         exit(1)
    finally:
         print("Boutique AI Orchestrator stopped.")


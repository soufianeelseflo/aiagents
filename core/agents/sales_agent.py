# core/agents/sales_agent.py

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Coroutine

# Import dependencies from other modules
from core.communication.voice_handler import VoiceHandler # Handles STT/TTS streaming
from core.services.llm_client import LLMClient # Handles OpenRouter interaction
from core.services.telephony_wrapper import TelephonyWrapper # Handles Twilio calls
from core.services.crm_wrapper import CRMWrapper # Handles CRM interaction (local logging)
# Import configuration centrally
import config

# Configure logging (inherits config from config.py)
logger = logging.getLogger(__name__)

class SalesAgent:
    """
    Autonomous AI agent responsible for conducting hyper-realistic voice sales calls
    from initiation to potential close for Boutique AI. Integrates various services.
    Manages its own call lifecycle and interaction flow.
    """

    def __init__(
        self,
        agent_id: str,
        target_phone_number: str,
        # --- Dependency Injection ---
        voice_handler: VoiceHandler,
        llm_client: LLMClient,
        telephony_wrapper: TelephonyWrapper,
        crm_wrapper: CRMWrapper,
        # --- Configuration ---
        initial_prompt: Optional[str] = None, # Allows overriding default prompt
        target_niche: str = config.AGENT_TARGET_NICHE_DEFAULT,
        max_call_duration: int = config.AGENT_MAX_CALL_DURATION_DEFAULT,
        # --- Callbacks (For Orchestrator interaction) ---
        on_call_complete_callback: Optional[Callable[[str, str, List[Dict[str, str]]], Coroutine[Any, Any, None]]] = None,
        on_call_error_callback: Optional[Callable[[str, str], Coroutine[Any, Any, None]]] = None,
        # --- Webhook/WebSocket Coordination ---
        # This agent needs a way to send TTS audio back via the correct telephony WebSocket.
        # Option 1: Pass the WebSocket object (complex state management).
        # Option 2: Pass an async queue the agent puts audio into, and the server reads from.
        # Option 3: Pass an async function that directly sends data over the correct WebSocket.
        # Using Option 3 for this implementation:
        send_audio_callback: Optional[Callable[[str, bytes], Coroutine[Any, Any, None]]] = None, # Args: call_sid, audio_chunk
        send_mark_callback: Optional[Callable[[str, str], Coroutine[Any, Any, None]]] = None # Args: call_sid, mark_name
        ):
        """
        Initializes the Sales Agent instance with injected dependencies and callbacks.

        Args:
            agent_id: A unique identifier for this agent instance.
            target_phone_number: The phone number the agent needs to call.
            voice_handler: Instance of VoiceHandler for STT/TTS.
            llm_client: Instance of LLMClient for language model interaction.
            telephony_wrapper: Instance of TelephonyWrapper for call control.
            crm_wrapper: Instance of CRMWrapper for CRM interaction/logging.
            initial_prompt: Specific system prompt for this agent/call. If None, uses a default.
            target_niche: Information about the specific niche for tailored conversation.
            max_call_duration: Maximum duration for the call attempt in seconds.
            on_call_complete_callback: Async callback for orchestrator on call success.
            on_call_error_callback: Async callback for orchestrator on call failure.
            send_audio_callback: Async function to send TTS audio chunk back via telephony WebSocket.
            send_mark_callback: Async function to send a 'mark' message back via telephony WebSocket.
        """
        self.agent_id = agent_id
        self.target_phone_number = target_phone_number
        self.voice_handler = voice_handler
        self.llm_client = llm_client
        self.telephony_wrapper = telephony_wrapper
        self.crm_wrapper = crm_wrapper
        self.target_niche = target_niche
        self.max_call_duration = max_call_duration
        self.on_call_complete_callback = on_call_complete_callback
        self.on_call_error_callback = on_call_error_callback
        self.send_audio_callback = send_audio_callback
        self.send_mark_callback = send_mark_callback

        # Construct the initial prompt
        self.initial_prompt = initial_prompt or self._get_default_initial_prompt()

        # State variables
        self.call_sid: Optional[str] = None
        self.stream_sid: Optional[str] = None # Sid for the specific media stream (from Twilio start event)
        self.conversation_history: List[Dict[str, str]] = []
        self.is_call_active: bool = False
        self.call_lifecycle_task: Optional[asyncio.Task] = None
        self.tts_task: Optional[asyncio.Task] = None # Task for ongoing TTS generation
        self.call_end_event = asyncio.Event() # Signal for call completion/failure

        # --- Critical Check ---
        if not self.send_audio_callback or not self.send_mark_callback:
             logger.error(f"Agent {self.agent_id}: Missing required callbacks 'send_audio_callback' or 'send_mark_callback'. Cannot function.")
             raise ValueError("SalesAgent requires send_audio_callback and send_mark_callback.")

        # Link voice handler callbacks to agent methods AFTER initialization
        # This ensures the agent instance is fully formed before callbacks are set.
        # Important: Assumes the VoiceHandler instance is dedicated to this agent/call.
        self.voice_handler.transcript_callback = self._handle_incoming_transcript
        self.voice_handler.error_callback = self._handle_voice_handler_error

        logger.info(f"SalesAgent {self.agent_id} initialized for target {self.target_phone_number} in niche '{self.target_niche}'.")

    def _get_default_initial_prompt(self) -> str:
        """ Generates a default system prompt including niche focus. """
        return (f"You are a highly skilled, hyper-realistic AI sales agent for Boutique AI, specializing in {self.target_niche}. "
                "Your goal is to engage the prospect, understand their needs, present Boutique AI's value proposition persuasively, "
                "handle objections effectively, and move the conversation towards a clear next step (e.g., booking a demo, closing a deal). "
                "Maintain a natural, conversational, confident, and empathetic tone. Keep responses concise for voice interaction. "
                "You must sound indistinguishable from a top-tier human salesperson.")

    async def start_sales_call(self, call_sid: str, stream_sid: str):
        """
        Starts the agent's logic once the call is connected and media stream is established.
        Called by the component managing the WebSocket connection (e.g., the FastAPI endpoint).

        Args:
            call_sid: The SID of the established call.
            stream_sid: The SID of the media stream.
        """
        if self.is_call_active:
            logger.warning(f"Agent {self.agent_id}: start_sales_call invoked but call already active (SID: {self.call_sid}). Ignoring.")
            return

        logger.info(f"Agent {self.agent_id}: Call connected (SID: {call_sid}, Stream: {stream_sid}). Starting active phase.")
        self.is_call_active = True
        self.call_sid = call_sid
        self.stream_sid = stream_sid
        self.call_end_event.clear()
        self.conversation_history = [] # Reset history

        try:
            # 1. Connect Voice Handler to Deepgram
            connected = await self.voice_handler.connect()
            if not connected or not self.voice_handler.is_connected:
                 raise ConnectionError(f"Agent {self.agent_id} [{self.call_sid}]: Failed to connect VoiceHandler to Deepgram.")

            # 2. (Optional) Fetch CRM data and potentially send an initial greeting
            prospect_data = await self.crm_wrapper.get_contact_info(self.target_phone_number)
            if prospect_data:
                 logger.info(f"Agent {self.agent_id} [{self.call_sid}]: Found CRM data for prospect.")
                 # Example: Start with a personalized greeting
                 # greeting = f"Hi {prospect_data.get('firstName', '')}, this is [Your Agent Persona Name] from Boutique AI. Is now still a good time?"
                 # await self._initiate_agent_turn(greeting) # Start the conversation
            else:
                 # Example: Start with a generic greeting if no data
                 # greeting = "Hi there, this is [Your Agent Persona Name] from Boutique AI. Am I speaking with the right person regarding [topic]?"
                 # await self._initiate_agent_turn(greeting)
                 logger.info(f"Agent {self.agent_id} [{self.call_sid}]: No CRM data found. Agent will proceed with generic opening or wait for user.")
                 # Or simply wait for the first transcript

            # 3. Start background task to manage call lifecycle (timeout, completion signal)
            self.call_lifecycle_task = asyncio.create_task(self._manage_call_lifecycle())
            logger.info(f"Agent {self.agent_id} [{self.call_sid}]: Lifecycle management started.")

        except Exception as e:
            error_msg = f"Error during sales call startup phase: {e}"
            logger.error(f"Agent {self.agent_id} [{self.call_sid}]: {error_msg}", exc_info=True)
            await self._handle_fatal_error(error_msg) # Trigger cleanup and error callback

    async def _manage_call_lifecycle(self):
        """ Background task monitoring call duration and end signal. """
        final_status = "Unknown"
        error_message = None
        if not self.call_sid: # Should not happen if called correctly
             logger.error(f"Agent {self.agent_id}: Lifecycle management started without call_sid.")
             return
        try:
            logger.info(f"Agent {self.agent_id} [{self.call_sid}]: Call lifecycle monitoring started (Timeout: {self.max_call_duration}s).")
            await asyncio.wait_for(self.call_end_event.wait(), timeout=self.max_call_duration)
            logger.info(f"Agent {self.agent_id} [{self.call_sid}]: Call end signal received normally.")
            # If signaled normally, final_status is determined by the reason _signal_call_end was called
            # For now, assume "Completed" unless an error was explicitly signaled.
            # A more robust system might pass status via the event or check agent state.
            final_status = "Completed" # Default assumption on normal signal

        except asyncio.TimeoutError:
            logger.warning(f"Agent {self.agent_id} [{self.call_sid}]: Call exceeded maximum duration ({self.max_call_duration}s). Forcing termination.")
            final_status = "Timeout"
            error_message = "Call exceeded maximum duration."
            await self.telephony_wrapper.end_call(self.call_sid)

        except Exception as e:
             logger.error(f"Agent {self.agent_id} [{self.call_sid}]: Unexpected error in call lifecycle: {e}", exc_info=True)
             final_status = "Error"
             error_message = f"Lifecycle management error: {e}"
             await self.telephony_wrapper.end_call(self.call_sid) # Attempt termination

        finally:
            logger.info(f"Agent {self.agent_id} [{self.call_sid}]: Initiating call cleanup from lifecycle task. Final Status: {final_status}")
            # Ensure cleanup happens even if called multiple times
            if self.is_call_active:
                 await self._cleanup_call(final_status, error_message)

    def signal_call_ended_externally(self, reason: str = "External Stop"):
        """ Called by the WebSocket handler when the 'stop' event is received from telephony. """
        if self.is_call_active:
             logger.info(f"Agent {self.agent_id} [{self.call_sid}]: Received external signal that call ended. Reason: {reason}")
             self._signal_call_end(reason)
        else:
             logger.debug(f"Agent {self.agent_id} [{self.call_sid}]: External call end signal received, but call already inactive.")

    def _signal_call_end(self, reason: str):
        """ Internal method to trigger the call end event. """
        if not self.call_end_event.is_set(): # Prevent setting multiple times
             logger.debug(f"Agent {self.agent_id} [{self.call_sid}]: Setting call end event. Reason: {reason}")
             self.call_end_event.set()

    # --- Core Interaction Logic ---

    async def _handle_incoming_transcript(self, transcript: str):
        """ Handles final transcripts received from the VoiceHandler. """
        if not self.is_call_active or not self.call_sid: return # Ignore if call inactive

        logger.info(f"Agent {self.agent_id} [{self.call_sid}]: Received transcript: '{transcript}'")
        self.conversation_history.append({"role": "client", "text": transcript})

        # --- Barge-in: Cancel any ongoing TTS ---
        await self._cancel_ongoing_tts()

        # --- Initiate Agent's Turn ---
        await self._initiate_agent_turn()

    async def _initiate_agent_turn(self, specific_text: Optional[str] = None):
        """ Generates LLM response (if needed) and starts TTS streaming task. """
        if not self.is_call_active: return

        try:
            llm_response_text = specific_text # Use provided text (e.g., initial greeting) if given
            if not llm_response_text:
                # Generate response from LLM based on history
                llm_response_text = await self.llm_client.generate_response(
                    model=config.OPENROUTER_MODEL_NAME,
                    messages=self._get_formatted_llm_messages()
                )

            if not llm_response_text: # Handle LLM failure
                 logger.error(f"Agent {self.agent_id} [{self.call_sid}]: LLM failed to generate response.")
                 llm_response_text = "I'm sorry, I seem to be having trouble formulating a response right now."

            # Add agent's response to history *before* speaking
            if not specific_text: # Don't double-add if it was an initial greeting
                self.conversation_history.append({"role": "agent", "text": llm_response_text})

            # Start TTS streaming in a background task
            logger.info(f"Agent {self.agent_id} [{self.call_sid}]: Starting TTS task for response: '{llm_response_text[:50]}...'")
            self.tts_task = asyncio.create_task(self._stream_tts(llm_response_text))

        except Exception as e:
            logger.error(f"Agent {self.agent_id} [{self.call_sid}]: Error initiating agent turn: {e}", exc_info=True)
            await self._handle_fatal_error(f"Error initiating agent turn: {e}")


    async def _stream_tts(self, text_to_speak: str):
        """ Task to handle streaming TTS audio back via the provided callback. """
        if not self.call_sid:
             logger.error(f"Agent {self.agent_id}: Cannot stream TTS, call_sid is not set.")
             return
        if not self.send_audio_callback or not self.send_mark_callback:
             logger.error(f"Agent {self.agent_id} [{self.call_sid}]: Cannot stream TTS, required callbacks missing.")
             return

        try:
            # Use VoiceHandler to get audio chunks
            async for audio_chunk in self.voice_handler.speak_text(text_to_speak):
                if audio_chunk:
                    # Use the injected callback to send audio over the correct WebSocket
                    await self.send_audio_callback(self.call_sid, audio_chunk)
                else:
                     logger.warning(f"Agent {self.agent_id} [{self.call_sid}]: TTS generator yielded empty chunk.")

            logger.info(f"Agent {self.agent_id} [{self.call_sid}]: Finished streaming TTS audio.")
            # Send a 'mark' message using the callback to signal end of speech
            await self.send_mark_callback(self.call_sid, f"agent_speech_ended_{int(time.time())}")

        except asyncio.CancelledError:
             logger.info(f"Agent {self.agent_id} [{self.call_sid}]: TTS streaming task was cancelled.")
             # Optionally send a mark even if cancelled? Maybe not necessary.
        except Exception as e:
             logger.error(f"Agent {self.agent_id} [{self.call_sid}]: Error occurred during TTS streaming task: {e}", exc_info=True)
             # Signal a fatal error if TTS fails critically?
             await self._handle_fatal_error(f"TTS Streaming Error: {e}")


    async def _cancel_ongoing_tts(self):
        """ Safely cancels the current TTS task if it's running. """
        if self.tts_task and not self.tts_task.done():
            logger.warning(f"Agent {self.agent_id} [{self.call_sid}]: Cancelling ongoing TTS task.")
            self.tts_task.cancel()
            try:
                await self.tts_task # Allow cancellation to process
            except asyncio.CancelledError:
                logger.info(f"Agent {self.agent_id} [{self.call_sid}]: Ongoing TTS task cancelled successfully.")
            self.tts_task = None


    async def _handle_voice_handler_error(self, error_message: str):
        """ Handles errors reported by the VoiceHandler during an active call. """
        logger.error(f"Agent {self.agent_id} [{self.call_sid}]: Received fatal error from VoiceHandler: {error_message}")
        await self._handle_fatal_error(f"VoiceHandler Error: {error_message}")


    async def _handle_fatal_error(self, error_message: str):
         """ Central handler for errors that should terminate the call. """
         logger.error(f"Agent {self.agent_id} [{self.call_sid}]: Encountered fatal error: {error_message}")
         # Signal the lifecycle manager to end the call and trigger cleanup
         self._signal_call_end(f"Fatal Error: {error_message}")
         # Optionally try to end call immediately
         if self.call_sid:
              await self.telephony_wrapper.end_call(self.call_sid)
         # Trigger orchestrator error callback directly if needed, though cleanup will also do it
         # if self.on_call_error_callback:
         #     await self.on_call_error_callback(self.agent_id, error_message)


    # --- Utility Methods ---

    def _get_formatted_llm_messages(self) -> List[Dict[str, str]]:
        """ Formats conversation history for the LLM API, including system prompt. """
        messages = [{"role": "system", "content": self.initial_prompt}]
        for entry in self.conversation_history:
            role = "user" if entry["role"] == "client" else "assistant"
            messages.append({"role": role, "content": entry["text"]})
        return messages

    async def _update_crm(self, status: str, notes: str):
        """ Logs call outcome using the CRMWrapper (which logs locally). """
        logger.info(f"Agent {self.agent_id} [{self.call_sid}]: Updating CRM Log. Status: {status}")
        try:
            success = await self.crm_wrapper.log_call_outcome(
                phone_number=self.target_phone_number,
                status=status,
                notes=notes,
                agent_id=self.agent_id, # Pass agent ID
                call_sid=self.call_sid,
                conversation_history=self.conversation_history
            )
            if not success:
                 logger.warning(f"Agent {self.agent_id} [{self.call_sid}]: CRM log update reported failure.")
        except Exception as e:
            logger.error(f"Agent {self.agent_id} [{self.call_sid}]: Failed to update CRM log: {e}", exc_info=True)

    async def _cleanup_call(self, final_status: str, error_message: Optional[str]):
        """ Cleans up all resources associated with the call. Ensures idempotency. """
        if not self.is_call_active:
             logger.debug(f"Agent {self.agent_id}: Cleanup called but call already inactive.")
             return
        self.is_call_active = False # Mark inactive FIRST

        logger.info(f"Agent {self.agent_id} [{self.call_sid}]: Cleaning up call resources. Final Status: {final_status}")

        # Cancel ongoing tasks safely
        await self._cancel_ongoing_tts()
        if self.call_lifecycle_task and not self.call_lifecycle_task.done():
             # Avoid cancelling self if called from lifecycle task's finally block
             if asyncio.current_task() is not self.call_lifecycle_task:
                  self.call_lifecycle_task.cancel()
                  try: await self.call_lifecycle_task
                  except asyncio.CancelledError: pass

        # Disconnect voice handler
        await self.voice_handler.disconnect()

        # Ensure telephony call is terminated (might be redundant but safe)
        if self.call_sid:
            await self.telephony_wrapper.end_call(self.call_sid)

        # Update CRM Log
        notes = f"Call ended. Status: {final_status}."
        if error_message: notes += f" Error: {error_message}"
        await self._update_crm(status=final_status, notes=notes)

        # Notify Orchestrator via callbacks
        if error_message and self.on_call_error_callback:
            logger.debug(f"Agent {self.agent_id}: Calling error callback.")
            await self.on_call_error_callback(self.agent_id, error_message)
        elif not error_message and self.on_call_complete_callback:
            logger.debug(f"Agent {self.agent_id}: Calling completion callback.")
            await self.on_call_complete_callback(self.agent_id, final_status, self.conversation_history)

        # Reset state fully
        logger.debug(f"Agent {self.agent_id}: Resetting final state variables.")
        self.call_sid = None
        self.stream_sid = None
        # Optionally clear history after logging: self.conversation_history = []
        self.tts_task = None
        self.call_lifecycle_task = None
        self.call_end_event.clear() # Reset event for next potential call

        logger.info(f"Agent {self.agent_id}: Cleanup complete.")

    # --- Public method to pass audio received from WebSocket ---
    async def handle_incoming_audio(self, audio_chunk: bytes):
         """ Receives audio chunks from the telephony WebSocket and sends to VoiceHandler. """
         if self.is_call_active:
              await self.voice_handler.send_audio_chunk(audio_chunk)
         else:
              logger.warning(f"Agent {self.agent_id}: Received audio chunk but call is not active.")


# Conceptual: How an orchestrator might use this (dependency injection needed)
# See main.py for actual instantiation and management.

if __name__ == "__main__":
    print("SalesAgent class defined (Functional). Instantiate and run via an orchestrator with proper dependencies and callbacks.")


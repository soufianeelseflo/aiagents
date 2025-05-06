# core/agents/sales_agent.py

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Coroutine

# Import dependencies from other modules
# These imports assume the structure core/communication and core/services
from core.communication.voice_handler import VoiceHandler
from core.services.llm_client import LLMClient # Includes caching
from core.services.telephony_wrapper import TelephonyWrapper
from core.services.crm_wrapper import CRMWrapper # Using Supabase version
# DataWrapper might be needed if agent needs to lookup Clay data mid-call
# from core.services.data_wrapper import DataWrapper
# Import configuration centrally - needed for defaults and model name
import config

# Configure logging (inherits level from config.py)
logger = logging.getLogger(__name__)

class SalesAgent:
    """
    Autonomous AI agent responsible for conducting hyper-realistic voice sales calls
    from initiation to potential close for Boutique AI. Integrates various services
    including Supabase for CRM logging and LLM caching.
    Manages its own call lifecycle and interaction flow based on real-time events.
    """

    def __init__(
        self,
        agent_id: str,
        target_phone_number: str,
        # --- Dependency Injection ---
        voice_handler: VoiceHandler,
        llm_client: LLMClient,
        telephony_wrapper: TelephonyWrapper,
        crm_wrapper: CRMWrapper, # Instance of the Supabase CRM wrapper
        # data_wrapper: Optional[DataWrapper] = None, # Optional: If agent needs direct Clay access
        # --- Configuration ---
        initial_prompt: Optional[str] = None,
        target_niche: str = config.AGENT_TARGET_NICHE_DEFAULT,
        max_call_duration: int = config.AGENT_MAX_CALL_DURATION_DEFAULT,
        # --- Callbacks (For Orchestrator interaction) ---
        on_call_complete_callback: Optional[Callable[[str, str, List[Dict[str, str]]], Coroutine[Any, Any, None]]] = None,
        on_call_error_callback: Optional[Callable[[str, str], Coroutine[Any, Any, None]]] = None,
        # --- WebSocket Callbacks (Required for sending TTS audio/marks) ---
        send_audio_callback: Optional[Callable[[str, bytes], Coroutine[Any, Any, None]]] = None, # Args: call_sid, audio_chunk
        send_mark_callback: Optional[Callable[[str, str], Coroutine[Any, Any, None]]] = None # Args: call_sid, mark_name
        ):
        """
        Initializes the Sales Agent instance with injected dependencies and callbacks.
        """
        self.agent_id = agent_id
        self.target_phone_number = target_phone_number
        self.voice_handler = voice_handler
        self.llm_client = llm_client # Now includes caching
        self.telephony_wrapper = telephony_wrapper
        self.crm_wrapper = crm_wrapper # Now uses Supabase
        # self.data_wrapper = data_wrapper # Store if provided
        self.target_niche = target_niche
        self.max_call_duration = max_call_duration
        self.on_call_complete_callback = on_call_complete_callback
        self.on_call_error_callback = on_call_error_callback
        self.send_audio_callback = send_audio_callback
        self.send_mark_callback = send_mark_callback

        # Validate required callbacks for sending audio/marks
        if not self.send_audio_callback or not self.send_mark_callback:
             logger.critical(f"Agent {self.agent_id}: CRITICAL - Missing required WebSocket callbacks. Agent cannot send audio.")
             raise ValueError("SalesAgent requires send_audio_callback and send_mark_callback.")

        # Construct the initial system prompt for the LLM
        self.initial_prompt = initial_prompt or self._get_default_initial_prompt()

        # State variables initialization
        self.call_sid: Optional[str] = None
        self.stream_sid: Optional[str] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.is_call_active: bool = False
        self.call_lifecycle_task: Optional[asyncio.Task] = None
        self.tts_task: Optional[asyncio.Task] = None
        self.call_end_event = asyncio.Event()
        self._final_status_reason: str = "Unknown" # Store reason for call end
        self._final_error_message: Optional[str] = None # Store error if applicable
        self.running = False
        self.call_task: Optional[asyncio.Task] = None
        self.interval: int = config.AGENT_RUN_INTERVAL_SECONDS

        # Link voice handler callbacks to agent methods AFTER initialization
        self.voice_handler.transcript_callback = self._handle_incoming_transcript
        self.voice_handler.error_callback = self._handle_voice_handler_error

        logger.info(f"SalesAgent {self.agent_id} initialized for target {self.target_phone_number} in niche '{self.target_niche}'.")

    def _get_default_initial_prompt(self) -> str:
        """ Generates a default system prompt including niche focus. """
        # This prompt is critical and should be refined based on testing and niche requirements.
        return (f"You are 'Alex', a highly persuasive and articulate AI sales executive representing Boutique AI, specializing in providing autonomous AI sales solutions for {self.target_niche}. "
                "Your primary objective is to conduct a natural, engaging conversation to uncover the prospect's key challenges and goals related to their sales operations and performance. "
                "Clearly articulate the unique value proposition of Boutique AI's fully autonomous agents – emphasizing efficiency gains, scalability, cost reduction, and hyper-realistic performance. "
                "Listen actively, demonstrate empathy, handle objections with data-driven insights and confidence, and strategically guide the conversation towards scheduling a qualified follow-up meeting or demo. "
                "Maintain a professional yet approachable tone. Your speech cadence and intonation must be indistinguishable from a top-performing human sales professional. Be concise.")

    async def start_sales_call(self, call_sid: str, stream_sid: str):
        """
        Activates the agent's logic once the call connects and media stream starts.
        Called by the WebSocket handler upon receiving the 'start' event.
        """
        if self.is_call_active:
            logger.warning(f"Agent {self.agent_id}: start_sales_call invoked but call already active (SID: {self.call_sid}). Ignoring.")
            return

        logger.info(f"Agent {self.agent_id}: Activating for connected call (SID: {call_sid}, Stream: {stream_sid}).")
        self.is_call_active = True
        self.call_sid = call_sid
        self.stream_sid = stream_sid
        self.call_end_event.clear()
        self.conversation_history = []
        self._final_status_reason = "Unknown" # Reset status
        self._final_error_message = None

        try:
            # 1. Connect Voice Handler to Deepgram
            await self.voice_handler.connect()
            await asyncio.sleep(0.5) # Allow time for connection confirmation via callback
            if not self.voice_handler.is_connected:
                 raise ConnectionError(f"Agent {self.agent_id} [{self.call_sid}]: Failed to establish active connection with Deepgram.")
            logger.info(f"Agent {self.agent_id} [{self.call_sid}]: VoiceHandler connected to Deepgram.")

            # 2. Fetch pre-call info from Supabase 'contacts' table
            # This now uses the functional Supabase CRMWrapper
            prospect_data = await self.crm_wrapper.get_contact_info(self.target_phone_number)
            if prospect_data:
                 logger.info(f"Agent {self.agent_id} [{self.call_sid}]: Pre-call contact data found in Supabase.")
                 # TODO: Personalize initial_prompt or first utterance based on prospect_data
                 # Example: Add prospect name to initial prompt if available
                 # first_name = prospect_data.get('first_name')
                 # if first_name: self.initial_prompt += f" You are speaking with {first_name}."
            else:
                 logger.info(f"Agent {self.agent_id} [{self.call_sid}]: No pre-call contact data found in Supabase.")

            # 3. Optionally send an initial greeting
            # greeting = "Hi, this is Alex from Boutique AI. Is this a good time to briefly discuss how autonomous agents could impact your sales?"
            # await self._initiate_agent_turn(greeting)
            logger.info(f"Agent {self.agent_id} [{self.call_sid}]: Ready and waiting for first user utterance.")

            # 4. Start background task to manage call lifecycle
            self.call_lifecycle_task = asyncio.create_task(self._manage_call_lifecycle())
            logger.info(f"Agent {self.agent_id} [{self.call_sid}]: Lifecycle monitoring task started.")

        except Exception as e:
            error_msg = f"Error during agent activation/startup: {e}"
            logger.error(f"Agent {self.agent_id} [{self.call_sid}]: {error_msg}", exc_info=True)
            await self._handle_fatal_error(error_msg) # Ensure cleanup and notification

    async def _manage_call_lifecycle(self):
        """ Background task monitoring call duration and the call_end_event. """
        if not self.call_sid: return
        final_status = "Error: Lifecycle ended prematurely"
        error_message = "Lifecycle task ended without proper signal"
        try:
            logger.info(f"Agent {self.agent_id} [{self.call_sid}]: Call lifecycle monitoring started (Timeout: {self.max_call_duration}s).")
            await asyncio.wait_for(self.call_end_event.wait(), timeout=self.max_call_duration)
            # Retrieve status/error set when event was triggered
            final_status = self._final_status_reason
            error_message = self._final_error_message
            logger.info(f"Agent {self.agent_id} [{self.call_sid}]: Call end signal received. Status: {final_status}")

        except asyncio.TimeoutError:
            logger.warning(f"Agent {self.agent_id} [{self.call_sid}]: Call TIMEOUT ({self.max_call_duration}s). Forcing termination.")
            final_status = "Timeout"
            error_message = "Call exceeded maximum duration."
            await self.telephony_wrapper.end_call(self.call_sid)

        except Exception as e:
             logger.error(f"Agent {self.agent_id} [{self.call_sid}]: Unexpected error in call lifecycle task: {e}", exc_info=True)
             final_status = "Error"
             error_message = f"Lifecycle Exception: {e}"
             await self.telephony_wrapper.end_call(self.call_sid)

        finally:
            logger.info(f"Agent {self.agent_id} [{self.call_sid}]: Lifecycle task ending. Initiating cleanup with status: {final_status}")
            if self.is_call_active: # Ensure cleanup runs only once
                 await self._cleanup_call(final_status, error_message)

    def signal_call_ended_externally(self, reason: str = "External Stop"):
        """ Called by the WebSocket handler on 'stop' event or disconnection. """
        logger.info(f"Agent {self.agent_id} [{self.call_sid}]: Received external signal that call ended. Reason: {reason}")
        self._signal_call_end(reason, is_error=False) # Assume normal completion

    def _signal_call_end(self, reason: str, is_error: bool = False, error_msg: Optional[str] = None):
        """ Internal method to trigger the call end event and store final status. """
        if not self.call_end_event.is_set():
             logger.debug(f"Agent {self.agent_id} [{self.call_sid}]: Setting call end event. Reason: {reason}, IsError: {is_error}")
             self._final_status_reason = "Error" if is_error else reason if reason != "Normal Completion" else "Completed"
             self._final_error_message = error_msg if is_error else None
             self.call_end_event.set()
        else:
             logger.debug(f"Agent {self.agent_id} [{self.call_sid}]: Call end signal received, but event already set.")


    async def _handle_incoming_transcript(self, transcript: str):
        """ Handles final transcripts received from the VoiceHandler. """
        if not self.is_call_active or not self.call_sid: return

        logger.info(f"Agent {self.agent_id} [{self.call_sid}]: Transcript received: '{transcript}'")
        self.conversation_history.append({"role": "client", "text": transcript})

        # Cancel any previous TTS task (barge-in)
        await self._cancel_ongoing_tts()

        # --- Basic Reasoning Step (Placeholder - Expand this) ---
        # Analyze transcript/history to decide next micro-goal before calling LLM
        micro_goal = self._determine_next_goal(transcript)
        logger.debug(f"Agent {self.agent_id} [{self.call_sid}]: Determined micro-goal: {micro_goal}")
        # --- End Reasoning Step ---

        # Initiate Agent's turn (get LLM response and start TTS)
        await self._initiate_agent_turn(micro_goal=micro_goal)

    def _determine_next_goal(self, last_transcript: str) -> str:
        """ Simple placeholder for reasoning about the next step. """
        # TODO: Implement more sophisticated reasoning based on conversation state,
        # sentiment analysis, objection detection, niche logic etc.
        if "schedule" in last_transcript.lower() or "demo" in last_transcript.lower():
            return "Confirm scheduling/demo details."
        elif "price" in last_transcript.lower() or "cost" in last_transcript.lower():
            return "Address pricing concerns with value proposition."
        elif "?" in last_transcript:
             return "Answer the prospect's question clearly and concisely."
        else:
            return "Continue building rapport and uncovering needs." # Default goal

    async def _initiate_agent_turn(self, specific_text: Optional[str] = None, micro_goal: Optional[str] = None):
        """ Generates LLM response (guided by micro_goal) and starts TTS streaming task. """
        if not self.is_call_active: return

        try:
            llm_response_text = specific_text
            if not llm_response_text:
                # Add micro-goal to prompt context if available
                current_messages = self._get_formatted_llm_messages()
                if micro_goal:
                     # Inject goal into the system prompt or as a pseudo-user message
                     # Example: Modify system prompt temporarily (less clean)
                     # Or add a user message like:
                     # current_messages.append({"role": "user", "content": f"[Internal Goal: {micro_goal}]"})
                     # For simplicity, let's slightly modify the system prompt concept here:
                     current_messages[0]["content"] += f" Your immediate goal for the next response is: {micro_goal}."

                llm_response_text = await self.llm_client.generate_response(
                    model=config.OPENROUTER_MODEL_NAME,
                    messages=current_messages
                    # Cache might be less effective if micro-goal changes prompt often
                    # use_cache=not bool(micro_goal) # Example: Disable cache if goal injected
                )

            if not llm_response_text: # Handle LLM failure
                 logger.error(f"Agent {self.agent_id} [{self.call_sid}]: LLM failed to generate response for goal '{micro_goal}'.")
                 llm_response_text = "I'm sorry, I need a moment to process that. Could you say that again?"

            # Add agent's response to history
            if not specific_text:
                self.conversation_history.append({"role": "agent", "text": llm_response_text})

            # Start TTS streaming task
            logger.info(f"Agent {self.agent_id} [{self.call_sid}]: Starting TTS task for response: '{llm_response_text[:50]}...'")
            self.tts_task = asyncio.create_task(self._stream_tts(llm_response_text))

        except Exception as e:
            logger.error(f"Agent {self.agent_id} [{self.call_sid}]: Error initiating agent turn: {e}", exc_info=True)
            await self._handle_fatal_error(f"Error initiating agent turn: {e}")


    async def _stream_tts(self, text_to_speak: str):
        """ Task to stream TTS audio chunks via the send_audio_callback. """
        if not self.call_sid or not self.send_audio_callback or not self.send_mark_callback: return
        tts_start_time = time.monotonic()
        total_bytes_sent = 0
        try:
            async for audio_chunk in self.voice_handler.speak_text(text_to_speak):
                if audio_chunk:
                    await self.send_audio_callback(self.call_sid, audio_chunk)
                    total_bytes_sent += len(audio_chunk)
            tts_duration = time.monotonic() - tts_start_time
            logger.info(f"Agent {self.agent_id} [{self.call_sid}]: Finished streaming TTS ({total_bytes_sent} bytes in {tts_duration:.2f}s).")
            mark_name = f"agent_turn_{len(self.conversation_history)}_{int(tts_start_time)}"
            await self.send_mark_callback(self.call_sid, mark_name)
        except asyncio.CancelledError:
             logger.info(f"Agent {self.agent_id} [{self.call_sid}]: TTS streaming task cancelled.")
        except Exception as e:
             logger.error(f"Agent {self.agent_id} [{self.call_sid}]: Error during TTS streaming task: {e}", exc_info=True)
             await self._handle_fatal_error(f"TTS Streaming Error: {e}")


    async def _cancel_ongoing_tts(self):
        """ Safely cancels the current TTS task if it's running. """
        if self.tts_task and not self.tts_task.done():
            logger.warning(f"Agent {self.agent_id} [{self.call_sid}]: Cancelling ongoing TTS task.")
            self.tts_task.cancel()
            try: await self.tts_task
            except asyncio.CancelledError: logger.info(f"Agent {self.agent_id} [{self.call_sid}]: Ongoing TTS task cancelled successfully.")
            except Exception as e: logger.error(f"Agent {self.agent_id} [{self.call_sid}]: Error awaiting cancelled TTS task: {e}")
            finally: self.tts_task = None


    async def _handle_voice_handler_error(self, error: str):
        """ Handles fatal errors reported by the VoiceHandler. """
        logger.error(f"VoiceHandler error on {self.call_sid}: {error}")
        self._signal_call_end("Error", True, error)


    async def _handle_fatal_error(self, message: str):
         """ Central handler for errors that should terminate the call. """
         logger.error(f"Fatal error: {message}")
         self._signal_call_end("Error", True, message)
         if self.on_call_error_callback:
            await self.on_call_error_callback(self.agent_id, message)


    async def _cleanup_call(self, final_status: str, error_message: Optional[str]):
        logger.info(f"Agent {self.agent_id}[{self.call_sid}]: Cleaning up. Status: {final_status}")
        if self.voice_handler.is_connected:
            await self.voice_handler.disconnect()
        await self.telephony_wrapper.end_call(self.call_sid)
        try:
            await self.crm_wrapper.log_call(self.call_sid, {'status': final_status, 'error': error_message})
            logger.info("Call outcome logged.")
        except Exception as e:
            logger.error(f"Failed to log outcome: {e}", exc_info=True)
        if self.on_call_complete_callback:
            await self.on_call_complete_callback(self.agent_id, self.call_sid, self.conversation_history)


    async def _initiate_agent_turn(self, text: str):
        logger.info(f"Agent {self.agent_id}[{self.call_sid}]: Speaking: {text}")
        await self.send_mark_callback(self.call_sid, 'agent_start')
        async for chunk in self.voice_handler.speak_text(text):
            await self.send_audio_callback(self.call_sid, chunk)
        await self.send_mark_callback(self.call_sid, 'agent_end')
        self.conversation_history.append({'role':'assistant','content':text})


    async def start(self, interval: int = config.AGENT_RUN_INTERVAL_SECONDS):
        """Begin periodic sales cycles."""
        if hasattr(self, 'call_task') and self.call_task and not self.call_task.done():
            logger.warning(f"SalesAgent {self.agent_id}: already running.")
            return
        self.running = True
        self.interval = interval
        self.call_task = asyncio.create_task(self._run_loop())
        logger.info(f"SalesAgent {self.agent_id}: Sales loop started (interval {self.interval}s).")


    async def stop(self):
        """Stop periodic sales cycles."""
        self.running = False
        if hasattr(self, 'call_task') and self.call_task:
            self.call_task.cancel()
            try:
                await self.call_task
            except asyncio.CancelledError:
                logger.info(f"SalesAgent {self.agent_id}: Task cancelled.")
        logger.info(f"SalesAgent {self.agent_id}: Sales loop stopped.")


    async def _run_loop(self):
        while self.running:
            logger.info(f"SalesAgent {self.agent_id}: Starting sales cycle.")
            try:
                count = await self._run_sales_cycle()
                logger.info(f"SalesAgent {self.agent_id}: Completed cycle. Calls made: {count}")
            except Exception as e:
                logger.error(f"Error in sales cycle: {e}", exc_info=True)
                if self.on_call_error_callback:
                    await self.on_call_error_callback(self.agent_id, str(e))
            await asyncio.sleep(self.interval)


    async def _run_sales_cycle(self) -> int:
        """Execute one batch of outbound calls."""
        calls = 0
        call_sid = await self.telephony_wrapper.initiate_call(self.target_phone_number)
        if call_sid:
            calls += 1
            await self.start_sales_call(call_sid, call_sid)
        return calls


    async def handle_incoming_audio(self, audio_chunk: bytes):
         """ Receives audio chunks from the telephony WebSocket and sends to VoiceHandler. """
         if self.is_call_active: await self.voice_handler.send_audio_chunk(audio_chunk)
         else: logger.debug(f"Agent {self.agent_id}: Received audio chunk but call inactive.")


    async def stop_call(self, reason: str = "External Stop Request"):
         """ Allows external stop request. """
         logger.info(f"Agent {self.agent_id} [{self.call_sid}]: Received external stop request. Reason: {reason}")
         self._signal_call_end(reason, is_error=False)


    def _get_formatted_llm_messages(self) -> List[Dict[str, str]]:
        """ Formats conversation history for the LLM API, including system prompt. """
        messages = [{"role": "system", "content": self.initial_prompt}]
        # Simple history inclusion - consider truncation/summarization for long calls
        history_limit = 20 # Limit history length sent to LLM
        start_index = max(0, len(self.conversation_history) - history_limit)
        for entry in self.conversation_history[start_index:]:
            role = "user" if entry["role"] == "client" else "assistant"
            content = str(entry.get("text", ""))
            messages.append({"role": role, "content": content})
        return messages


if __name__ == "__main__":
    print("SalesAgent class defined (Functional - Supabase/Cache - Final). Use via Orchestrator.")

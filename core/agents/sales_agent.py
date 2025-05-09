# /core/agents/sales_agent.py: (Reviewed for Imports and Robustness)
# --------------------------------------------------------------------------------
# boutique_ai_project/core/agents/sales_agent.py

import asyncio
import logging
import json
from enum import Enum, auto
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable, Coroutine, AsyncGenerator # Added missing types

# Local imports
import config
from core.services.llm_client import LLMClient
from core.communication.voice_handler import VoiceHandler, VoiceHandlerError, STTProvider, TTSProvider
from core.services.telephony_wrapper import TelephonyWrapper, TelephonyError
from core.services.crm_wrapper import CRMWrapper, CRMError

logger = logging.getLogger(__name__)

# --- Enums for State Management ---
class SalesCallPhase(Enum):
    NOT_STARTED = auto()
    INITIALIZING = auto()
    INTRODUCTION = auto()
    QUALIFICATION = auto()
    DISCOVERY = auto()
    PITCH = auto()
    OBJECTION_HANDLING = auto()
    CLOSING_ATTEMPT = auto()
    WRAP_UP = auto()
    CALL_ENDED_SUCCESS = auto()
    CALL_ENDED_FAIL_PROSPECT = auto()
    CALL_ENDED_FAIL_INTERNAL = auto()
    CALL_ENDED_MAX_DURATION = auto()
    CALL_ENDED_OPERATOR = auto()

class ProspectSentiment(Enum):
    UNKNOWN = auto()
    POSITIVE = auto()
    NEUTRAL = auto()
    NEGATIVE = auto()
    INTERESTED = auto()
    NOT_INTERESTED = auto()

class CallOutcomeCategory(Enum):
    NO_ANSWER = auto()
    VOICEMAIL_LEFT = auto()
    SHORT_INTERACTION_NO_PITCH = auto()
    CONVERSATION_NO_CLOSE = auto()
    VERBAL_AGREEMENT_TO_NEXT_STEP = auto()
    MEETING_SCHEDULED = auto()
    SALE_CLOSED = auto() # Ambitious for pure voice, but possible
    REQUESTED_CALLBACK = auto()
    WRONG_NUMBER = auto()
    DO_NOT_CALL = auto()
    ERROR_OR_UNKNOWN = auto()


class SalesAgent:
    """
    Manages an individual sales call, including STT/TTS, LLM interaction,
    and call state. (Level 47)
    """
    MAX_SILENCE_DURATION_SECONDS = 20 # How long to wait for prospect response before nudging or ending
    MAX_CALL_DURATION_SECONDS = config.AGENT_MAX_CALL_DURATION_DEFAULT
    MAX_CONSECUTIVE_ERRORS = 3 # Max consecutive STT/TTS/LLM errors before ending call

    def __init__(
        self,
        agent_id: str,
        target_phone_number: str,
        voice_handler: VoiceHandler,
        llm_client: LLMClient,
        telephony_wrapper: TelephonyWrapper,
        crm_wrapper: Optional[CRMWrapper] = None, # CRM is optional but recommended
        initial_prospect_data: Optional[Dict[str, Any]] = None,
        send_audio_callback: Callable[[str, bytes], Coroutine[Any, Any, None]], # Call SID, Audio Bytes
        send_mark_callback: Callable[[str, str], Coroutine[Any, Any, None]],      # Call SID, Mark Name
        on_call_complete_callback: Optional[Callable[[str, SalesCallPhase, List[Dict], Dict], Coroutine[Any, Any, None]]] = None, # agent_id, final_phase, history, summary
        on_call_error_callback: Optional[Callable[[str, str], Coroutine[Any, Any, None]]] = None # agent_id, error_message
    ):
        self.agent_id = agent_id
        self.target_phone_number = target_phone_number
        self.voice_handler = voice_handler
        self.llm_client = llm_client
        self.telephony_wrapper = telephony_wrapper
        self.crm_wrapper = crm_wrapper
        self.initial_prospect_data = initial_prospect_data or {}

        # Callbacks for communication with the WebSocket handler / server
        self._send_audio_to_ws = send_audio_callback
        self._send_mark_to_ws = send_mark_callback
        self._on_call_complete_callback = on_call_complete_callback
        self._on_call_error_callback = on_call_error_callback

        self.call_sid: Optional[str] = None
        self.stream_sid: Optional[str] = None
        self.call_phase = SalesCallPhase.NOT_STARTED
        self.prospect_sentiment = ProspectSentiment.UNKNOWN
        self.conversation_history: List[Dict[str, str]] = [] # [{"role": "user/assistant/system", "content": "..."}]
        self.call_summary: Dict[str, Any] = {}
        self.is_call_active = False
        self.start_time: Optional[datetime] = None
        self.last_prospect_speech_time: Optional[datetime] = None
        self.error_counter = 0

        # Configure VoiceHandler callbacks for this agent instance
        self.voice_handler.set_transcript_callback(self._handle_stt_transcript)
        self.voice_handler.set_error_callback(self._handle_voice_handler_error)

        logger.info(f"SalesAgent [{self.agent_id}] initialized for target: {self.target_phone_number}.")

    async def _handle_stt_transcript(self, transcript: str, is_final: bool, **kwargs):
        """Handles incoming transcripts from the VoiceHandler (STT)."""
        if not self.is_call_active: return
        if not transcript.strip(): return # Ignore empty transcripts

        logger.info(f"[{self.agent_id}/{self.call_sid}] Transcript (Final={is_final}): {transcript}")
        self.last_prospect_speech_time = datetime.now(timezone.utc)

        if is_final:
            self._add_to_history("user", transcript)
            # Trigger LLM response generation
            await self._generate_and_send_llm_response()

    async def _handle_voice_handler_error(self, error: VoiceHandlerError):
        """Handles errors from the VoiceHandler (STT/TTS)."""
        if not self.is_call_active: return
        logger.error(f"[{self.agent_id}/{self.call_sid}] VoiceHandler Error: {error.message}", exc_info=error.original_exception)
        self.error_counter += 1
        if self.error_counter >= self.MAX_CONSECUTIVE_ERRORS:
            await self._handle_fatal_error(f"Max VoiceHandler errors reached: {error.message}")

    async def _handle_fatal_error(self, reason: str, phase: SalesCallPhase = SalesCallPhase.CALL_ENDED_FAIL_INTERNAL):
        """Handles a fatal error that requires ending the call."""
        logger.critical(f"[{self.agent_id}/{self.call_sid}] FATAL ERROR: {reason}. Ending call.")
        self.call_phase = phase
        if self._on_call_error_callback:
            try:
                await self._on_call_error_callback(self.agent_id, reason)
            except Exception as cb_err:
                logger.error(f"Error in on_call_error_callback: {cb_err}", exc_info=True)
        await self.stop_call(reason) # stop_call also handles cleanup


    def _add_to_history(self, role: str, content: str):
        """Adds a message to the conversation history."""
        if not content.strip(): return
        self.conversation_history.append({"role": role, "content": content.strip()})
        # Optional: Log to CRM in real-time or batch
        if self.crm_wrapper and self.call_sid:
            # Fire and forget to avoid blocking call flow
            asyncio.create_task(
                self.crm_wrapper.log_call_interaction(
                    self.call_sid, role, content.strip(), phase=self.call_phase.name
                )
            )

    async def _generate_and_send_llm_response(self):
        """Generates a response using LLM and sends it via TTS."""
        if not self.is_call_active: return
        if not self.llm_client:
            await self._handle_fatal_error("LLMClient not available for generating response.")
            return

        # Basic prompt engineering - this should be much more sophisticated
        # TODO: Implement dynamic prompt strategy based on call_phase, prospect_data, history
        system_prompt = (
            "You are an AI Sales Agent for 'BoutiqueAI', a company offering custom AI agent solutions. "
            "Your goal is to understand the prospect's needs, qualify them, and guide them towards a demo or next step. "
            "Be conversational, empathetic, and professional. Keep responses concise for voice interaction."
            f"Current call phase: {self.call_phase.name}. Prospect info: {json.dumps(self.initial_prospect_data)}"
        )
        # Limit history to keep context window manageable
        recent_history = [{"role": "system", "content": system_prompt}] + self.conversation_history[-10:]

        try:
            logger.debug(f"[{self.agent_id}/{self.call_sid}] Sending to LLM. History snippet: {recent_history[-3:]}")
            llm_response_data = await self.llm_client.get_chat_completion(
                messages=recent_history,
                model=config.OPENROUTER_DEFAULT_CONVERSATIONAL_MODEL,
                temperature=0.7,
                max_tokens=150 # Keep responses relatively short for voice
            )

            if llm_response_data and llm_response_data.get("choices"):
                ai_response_text = llm_response_data["choices"][0]["message"]["content"]
                if not ai_response_text or not ai_response_text.strip():
                     logger.warning(f"[{self.agent_id}/{self.call_sid}] LLM returned empty response. Nudging prospect.")
                     ai_response_text = "Sorry, I missed that. Could you say it again?" # Fallback

                self._add_to_history("assistant", ai_response_text)
                logger.info(f"[{self.agent_id}/{self.call_sid}] LLM Response: {ai_response_text}")

                # Send response via TTS through VoiceHandler
                await self.voice_handler.send_tts_audio(ai_response_text, self.call_sid) # Pass call_sid for callback mapping
                self.error_counter = 0 # Reset error counter on successful LLM/TTS cycle
            else:
                logger.error(f"[{self.agent_id}/{self.call_sid}] LLM returned no response or empty choices.")
                self.error_counter +=1
                if self.error_counter >= self.MAX_CONSECUTIVE_ERRORS:
                     await self._handle_fatal_error("Max LLM errors reached (no response).")
                else: # Try a generic recovery phrase
                     await self.voice_handler.send_tts_audio("I'm having a little trouble processing that, could you repeat?", self.call_sid)


        except Exception as e:
            logger.error(f"[{self.agent_id}/{self.call_sid}] Error in LLM interaction or TTS: {e}", exc_info=True)
            self.error_counter += 1
            if self.error_counter >= self.MAX_CONSECUTIVE_ERRORS:
                await self._handle_fatal_error(f"Max LLM/TTS errors reached: {e}")
            else: # Try a generic recovery phrase
                try:
                    await self.voice_handler.send_tts_audio("I seem to be having a technical difficulty. One moment.", self.call_sid)
                except Exception as tts_e:
                    logger.error(f"[{self.agent_id}/{self.call_sid}] Backup TTS failed: {tts_e}")


    async def start_sales_call(self, call_sid: str, stream_sid: str):
        """Initiates the sales call flow."""
        if self.is_call_active:
            logger.warning(f"[{self.agent_id}/{call_sid}] Call already active. Ignoring duplicate start.")
            return

        self.call_sid = call_sid
        self.stream_sid = stream_sid # Provided by Twilio media stream start event
        self.is_call_active = True
        self.start_time = datetime.now(timezone.utc)
        self.last_prospect_speech_time = self.start_time
        self.call_phase = SalesCallPhase.INITIALIZING
        self.error_counter = 0
        self.conversation_history = [] # Reset for new call
        self.call_summary = {
            "agent_id": self.agent_id, "call_sid": self.call_sid,
            "target_phone": self.target_phone_number, "start_time": self.start_time.isoformat(),
            "initial_prospect_data": self.initial_prospect_data
        }

        logger.info(f"[{self.agent_id}/{self.call_sid}] Starting sales call. Stream SID: {self.stream_sid}")

        try:
            # Connect VoiceHandler to the audio stream (Deepgram, etc.)
            # Pass the audio sending callback to VoiceHandler for TTS output
            await self.voice_handler.connect(
                stt_provider=STTProvider.DEEPGRAM, # Or from config
                tts_provider=TTSProvider.DEEPGRAM, # Or from config
                send_audio_upstream_callback=self._send_audio_to_ws, # Callback to send TTS to Twilio WS
                mark_callback=self._send_mark_to_ws,
                call_identifier=self.call_sid # Used to tag TTS audio for the correct websocket stream
            )
            self.call_phase = SalesCallPhase.INTRODUCTION

            # Initial greeting
            greeting = f"Hello! This is {config.OPENROUTER_APP_NAME or 'your AI assistant'}. Is this a good time to talk briefly?"
            # More personalized greeting if prospect data is available
            prospect_name = self.initial_prospect_data.get('first_name') or self.initial_prospect_data.get('name')
            if prospect_name:
                greeting = f"Hi {prospect_name}, this is {config.OPENROUTER_APP_NAME or 'your AI assistant'}. Is this a good time for a quick chat?"

            self._add_to_history("assistant", greeting)
            await self.voice_handler.send_tts_audio(greeting, self.call_sid)

            # Start silence detection loop
            asyncio.create_task(self._monitor_silence())
            # Start max duration monitor
            asyncio.create_task(self._monitor_max_duration())

        except VoiceHandlerError as e:
            await self._handle_fatal_error(f"Failed to connect VoiceHandler: {e.message}")
        except Exception as e:
            await self._handle_fatal_error(f"Unexpected error starting sales call: {e}")


    async def handle_incoming_audio(self, audio_chunk: bytes):
        """Receives audio chunks from the WebSocket (Twilio Media Stream)."""
        if not self.is_call_active or not self.voice_handler.is_connected():
            # logger.debug(f"[{self.agent_id}/{self.call_sid}] Dropping audio chunk, call not active or voice_handler not connected.")
            return
        await self.voice_handler.receive_audio_from_upstream(audio_chunk)


    async def _monitor_silence(self):
        """Periodically checks for prospect silence and may nudge or end the call."""
        logger.debug(f"[{self.agent_id}/{self.call_sid}] Silence monitor started.")
        while self.is_call_active:
            await asyncio.sleep(self.MAX_SILENCE_DURATION_SECONDS / 2) # Check periodically
            if not self.is_call_active: break

            if self.last_prospect_speech_time:
                silence_duration = (datetime.now(timezone.utc) - self.last_prospect_speech_time).total_seconds()
                if silence_duration > self.MAX_SILENCE_DURATION_SECONDS:
                    logger.warning(f"[{self.agent_id}/{self.call_sid}] Prospect silence detected for over {self.MAX_SILENCE_DURATION_SECONDS}s.")
                    # TODO: Implement a nudge strategy (e.g., "Are you still there?")
                    # For now, we'll end the call if prolonged silence after initial interaction
                    if len(self.conversation_history) > 2: # Avoid ending immediately after greeting
                        await self._handle_fatal_error("Prolonged prospect silence.", SalesCallPhase.CALL_ENDED_FAIL_PROSPECT)
                    else: # Nudge if very early in call
                        nudge_message = "Hello? Are you still there?"
                        self._add_to_history("assistant", nudge_message)
                        await self.voice_handler.send_tts_audio(nudge_message, self.call_sid)
                        self.last_prospect_speech_time = datetime.now(timezone.utc) # Reset nudge timer
        logger.debug(f"[{self.agent_id}/{self.call_sid}] Silence monitor stopped.")

    async def _monitor_max_duration(self):
        """Monitors the call duration and ends it if it exceeds the maximum."""
        logger.debug(f"[{self.agent_id}/{self.call_sid}] Max duration monitor started ({self.MAX_CALL_DURATION_SECONDS}s).")
        await asyncio.sleep(self.MAX_CALL_DURATION_SECONDS)
        if self.is_call_active:
            logger.warning(f"[{self.agent_id}/{self.call_sid}] Call exceeded maximum duration of {self.MAX_CALL_DURATION_SECONDS}s.")
            await self._handle_fatal_error("Maximum call duration reached.", SalesCallPhase.CALL_ENDED_MAX_DURATION)
        logger.debug(f"[{self.agent_id}/{self.call_sid}] Max duration monitor stopped.")

    async def signal_call_ended_externally(self, reason: str = "External Stop Signal"):
        """Called if the WebSocket or Twilio signals the call has ended."""
        if self.is_call_active:
            logger.info(f"[{self.agent_id}/{self.call_sid}] Call ended externally: {reason}. Current phase: {self.call_phase.name}")
            # Determine appropriate end phase if not already an error/success phase
            if self.call_phase not in [
                SalesCallPhase.CALL_ENDED_SUCCESS, SalesCallPhase.CALL_ENDED_FAIL_PROSPECT,
                SalesCallPhase.CALL_ENDED_FAIL_INTERNAL, SalesCallPhase.CALL_ENDED_MAX_DURATION,
                SalesCallPhase.CALL_ENDED_OPERATOR
            ]:
                self.call_phase = SalesCallPhase.CALL_ENDED_OPERATOR # Generic operator/external end
            await self.stop_call(reason)


    async def stop_call(self, reason: str = "Call stopped by agent logic."):
        """Cleans up resources and finalizes the call."""
        if not self.is_call_active:
            logger.debug(f"[{self.agent_id}/{self.call_sid}] Stop_call invoked but call not active or already stopping.")
            return
        self.is_call_active = False # Primary flag to stop other loops/tasks

        logger.info(f"[{self.agent_id}/{self.call_sid}] Stopping call. Reason: {reason}. Final Phase: {self.call_phase.name}")

        # Disconnect VoiceHandler (STT/TTS)
        if self.voice_handler and self.voice_handler.is_connected():
            await self.voice_handler.disconnect()

        # Finalize call summary
        end_time = datetime.now(timezone.utc)
        self.call_summary["end_time"] = end_time.isoformat()
        self.call_summary["duration_seconds"] = (end_time - self.start_time).total_seconds() if self.start_time else 0
        self.call_summary["final_call_phase"] = self.call_phase.name
        self.call_summary["final_prospect_sentiment"] = self.prospect_sentiment.name # TODO: LLM-based sentiment analysis
        self.call_summary["stop_reason"] = reason
        # Determine CallOutcomeCategory based on phase and history
        # This logic can be quite complex; basic mapping for now:
        if self.call_phase == SalesCallPhase.CALL_ENDED_SUCCESS: # Placeholder for more specific success
            self.call_summary["final_call_outcome_category"] = CallOutcomeCategory.CONVERSATION_NO_CLOSE.name # Default success
        elif self.call_phase in [SalesCallPhase.CALL_ENDED_FAIL_PROSPECT, SalesCallPhase.CALL_ENDED_MAX_DURATION, SalesCallPhase.CALL_ENDED_OPERATOR]:
            self.call_summary["final_call_outcome_category"] = CallOutcomeCategory.SHORT_INTERACTION_NO_PITCH.name # Default if prospect ended
        else: # CALL_ENDED_FAIL_INTERNAL or other
            self.call_summary["final_call_outcome_category"] = CallOutcomeCategory.ERROR_OR_UNKNOWN.name

        # Log conversation history and summary
        logger.info(f"[{self.agent_id}/{self.call_sid}] Call Summary: {json.dumps(self.call_summary)}")
        # logger.debug(f"[{self.agent_id}/{self.call_sid}] Full Conversation History: {json.dumps(self.conversation_history, indent=2)}")

        # Update CRM with final call log (if CRM wrapper is available)
        if self.crm_wrapper and self.call_sid:
            try:
                await self.crm_wrapper.upsert_call_log(
                    self.call_sid,
                    self.agent_id,
                    self.target_phone_number,
                    self.initial_prospect_data.get("contact_id"), # Assuming contact_id is fetched/created
                    self.start_time,
                    end_time,
                    (end_time - self.start_time).total_seconds() if self.start_time else 0,
                    self.call_summary["final_call_outcome_category"],
                    self.conversation_history, # Full transcript
                    self.call_summary # Additional structured data
                )
                logger.info(f"[{self.agent_id}/{self.call_sid}] Call log updated in CRM.")
            except CRMError as e:
                logger.error(f"[{self.agent_id}/{self.call_sid}] CRM Error updating call log: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"[{self.agent_id}/{self.call_sid}] Unexpected error updating CRM: {e}", exc_info=True)

        # Notify main server/handler that call is complete
        if self._on_call_complete_callback:
            try:
                await self._on_call_complete_callback(
                    self.agent_id, self.call_phase, self.conversation_history, self.call_summary
                )
            except Exception as e:
                logger.error(f"Error in on_call_complete_callback for {self.agent_id}: {e}", exc_info=True)

        logger.info(f"[{self.agent_id}/{self.call_sid}] Call cleanup finished.")
# --------------------------------------------------------------------------------
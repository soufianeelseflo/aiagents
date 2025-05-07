# core/agents/sales_agent.py

import asyncio
import logging
import time
import json # For LLM interaction and structured logging
from typing import Dict, Any, List, Optional, Callable, Coroutine, Tuple

import config # Root config
from core.communication.voice_handler import VoiceHandler
from core.services.llm_client import LLMClient
from core.services.telephony_wrapper import TelephonyWrapper
from core.services.crm_wrapper import CRMWrapper
# from core.services.data_wrapper import DataWrapper # Optional, if needed mid-call

logger = logging.getLogger(__name__)

# Predefined strategic micro-goals for the LLM to choose from
CONVERSATIONAL_STRATEGIC_GOALS = [
    "BuildRapport", "DeepenPainDiscovery", "QualifyNeedAndBudget", "IntroduceSolutionConcept",
    "MapSolutionToPain", "HandleObjection_Price", "HandleObjection_Competitor", "HandleObjection_Timing",
    "HandleObjection_NotInterested", "HandleObjection_NeedMoreInfo", "RequestNextStep_Demo",
    "RequestNextStep_Meeting", "ClarifyProspectStatement", "VerifyAgreement", "ProvideValue_NoAsk",
    "AttemptGentleClose", "ReiterateValueProposition", "GatherInformation_DecisionProcess", "FinalizeDetails"
]

class SalesAgent:
    """
    Autonomous AI Sales Virtuoso (Level 45). Conducts hyper-realistic, context-aware
    voice sales calls, driven by dynamic LLM-guided strategy and focused on outcomes.
    """

    def __init__(
        self,
        agent_id: str,
        target_phone_number: str,
        voice_handler: VoiceHandler,
        llm_client: LLMClient,
        telephony_wrapper: TelephonyWrapper,
        crm_wrapper: CRMWrapper,
        initial_prospect_data: Optional[Dict[str, Any]] = None, # "Level 25" context
        target_niche: Optional[str] = None, # Can override default from prospect_data
        # Callbacks for server/orchestrator
        send_audio_callback: Callable[[str, bytes], Coroutine[Any, Any, None]],
        send_mark_callback: Callable[[str, str], Coroutine[Any, Any, None]],
        on_call_complete_callback: Optional[Callable[[str, str, List[Dict[str, str]], Dict[str, Any]], Coroutine[Any, Any, None]]] = None, # agent_id, final_status, history, call_summary_obj
        on_call_error_callback: Optional[Callable[[str, str], Coroutine[Any, Any, None]]] = None
        ):
        self.agent_id = agent_id
        self.target_phone_number = target_phone_number
        self.voice_handler = voice_handler
        self.llm_client = llm_client
        self.telephony_wrapper = telephony_wrapper
        self.crm_wrapper = crm_wrapper
        
        self.initial_prospect_data = initial_prospect_data or {}
        self.target_niche = target_niche or self.initial_prospect_data.get("niche_override") or config.AGENT_TARGET_NICHE_DEFAULT
        self.max_call_duration = config.AGENT_MAX_CALL_DURATION_DEFAULT
        
        self.send_audio_callback = send_audio_callback
        self.send_mark_callback = send_mark_callback
        self.on_call_complete_callback = on_call_complete_callback
        self.on_call_error_callback = on_call_error_callback

        # Initial system prompt is now dynamically generated with context
        self.system_prompt: str = self._generate_contextual_system_prompt()

        # State variables
        self.call_sid: Optional[str] = None
        self.stream_sid: Optional[str] = None # From Twilio media stream 'start' event
        self.conversation_history: List[Dict[str, str]] = [] # Standard OpenAI format: {"role": "user/assistant", "content": "..."}
        self.is_call_active: bool = False
        self.call_lifecycle_task: Optional[asyncio.Task] = None
        self.current_tts_task: Optional[asyncio.Task] = None
        self.call_end_event = asyncio.Event()
        self._final_status_reason: str = "Unknown"
        self._final_error_message: Optional[str] = None
        self._current_strategic_goal: str = "BuildRapport" # Initial goal
        self._last_llm_strategic_reasoning: str = "Initial rapport building."

        # Link voice handler callbacks to agent methods
        self.voice_handler.transcript_callback = self._handle_final_transcript
        self.voice_handler.error_callback = self._handle_voice_handler_error

        logger.info(f"SalesAgent {self.agent_id} (Level 45) initialized for {self.target_phone_number} in niche '{self.target_niche}'.")
        logger.debug(f"[{self.agent_id}] Initial system prompt generated (first 100 chars): {self.system_prompt[:100]}...")
        if self.initial_prospect_data:
            logger.debug(f"[{self.agent_id}] Initial prospect data context: {json.dumps(self.initial_prospect_data, default=str, indent=2)[:500]}")


    def _generate_contextual_system_prompt(self) -> str:
        """Generates a highly contextual system prompt using initial prospect data."""
        prospect_name = self.initial_prospect_data.get("first_name", "there")
        company_name = self.initial_prospect_data.get("company_name", "their company")
        inferred_pain_1 = self.initial_prospect_data.get("llm_inferred_pain_point_1", "improving sales effectiveness")
        suggested_hook = self.initial_prospect_data.get("llm_suggested_hook", f"explore how AI can significantly boost your sales outcomes for {company_name}")

        # Base persona and objective
        prompt = (
            f"You are Alex, a highly articulate and empathetic Senior AI Solutions Consultant from Boutique AI. "
            f"You specialize in helping businesses in the '{self.target_niche}' sector transform their sales operations using advanced AI Sales Agents. "
            f"Your voice, cadence, and interaction style must be indistinguishable from a top-tier human consultant – warm, confident, and insightful. Avoid sounding robotic or overly scripted.\n\n"
            f"Your primary objective for this call with {prospect_name} from {company_name} is to deeply understand their current sales challenges, "
            f"align Boutique AI's unique value proposition (hyper-realistic AI agents for end-to-end sales cycles, driving efficiency, scalability, and expert-level performance) to those specific pains, "
            f"and secure a concrete next step, ideally a qualified demo or a strategic follow-up meeting with a specialist.\n\n"
        )

        # Incorporate pre-call intelligence
        prompt += f"Pre-call intelligence suggests {company_name} might be focused on '{inferred_pain_1}'. "
        prompt += f"A potential way to open the conversation or frame value could be around how you can help them '{suggested_hook}'. Use this insight judiciously.\n\n"

        # AI Meta-Awareness
        prompt += (
            "You are an advanced AI. While your goal is a natural human conversation, if a unique AI capability (like rapid data synthesis for insights, or tireless consistency) "
            "becomes directly relevant to solving a prospect's stated problem, you can subtly weave it in to build credibility. Example: 'Because I can process information from thousands of similar scenarios, I can see a pattern here...' "
            "However, prioritize genuine connection and problem-solving over showcasing your AI nature.\n\n"
        )
        
        # Conversational Style & Strategy
        prompt += (
            "Conversational Mandates:\n"
            "- Listen more than you speak. Ask open-ended, insightful questions.\n"
            "- Demonstrate active listening by referencing previous statements from the prospect.\n"
            "- Handle objections with empathy, data-driven responses, and by reframing to value. Do not be dismissive.\n"
            "- Maintain control of the conversation flow, gently guiding it towards your objectives.\n"
            "- Verify key understandings and agreements explicitly (e.g., 'So, to confirm, the main challenge is X, and you're open to exploring Y?').\n"
            "- Always aim to provide value, even if a sale isn't immediate.\n"
            "- Your speech should be fluid, with natural pauses and intonation. Avoid filler words unless they sound exceptionally natural.\n"
            "- Be concise but thorough. Respect the prospect's time.\n"
            "- You will be given an immediate strategic micro-goal for each turn. Fulfill this goal within the broader conversational context.\n"
            "- Sell the 'dream' – the transformation and outcomes – not just features.\n"
        )
        return prompt

    async def start_sales_call(self, call_sid: str, stream_sid: str):
        if self.is_call_active: logger.warning(f"[{self.agent_id}] Start called but already active (SID: {self.call_sid})."); return
        logger.info(f"[{self.agent_id}] Activating for call SID: {call_sid}, Stream SID: {stream_sid}, Prospect: {self.target_phone_number}")
        self.is_call_active = True
        self.call_sid = call_sid
        self.stream_sid = stream_sid # Store the stream_sid from Twilio's 'start' event
        self.call_end_event.clear()
        self.conversation_history = []
        self._final_status_reason = "Call Initiated"
        self._current_strategic_goal = "BuildRapport" # Reset initial goal

        try:
            if not await self.voice_handler.connect():
                raise ConnectionError("Failed to connect VoiceHandler to Deepgram.")
            logger.info(f"[{self.agent_id}] VoiceHandler connected to Deepgram.")

            # Initial greeting or first strategic turn
            # The first turn might be an LLM-generated greeting based on the initial prompt and prospect data.
            logger.info(f"[{self.agent_id}] Initiating first agent turn (goal: {self._current_strategic_goal}).")
            await self._initiate_agent_turn() # Let it determine first utterance based on initial goal

            self.call_lifecycle_task = asyncio.create_task(self._manage_call_lifecycle())
            logger.info(f"[{self.agent_id}] Call lifecycle monitoring started.")
        except Exception as e:
            error_msg = f"Error during SalesAgent activation: {e}"
            logger.error(f"[{self.agent_id}] {error_msg}", exc_info=True)
            await self._handle_fatal_error(error_msg)

    async def _manage_call_lifecycle(self): # Largely same as before, ensures cleanup
        if not self.call_sid: return
        final_status, error_message = "Error: Lifecycle ended prematurely", "Lifecycle task ended without signal"
        try:
            await asyncio.wait_for(self.call_end_event.wait(), timeout=self.max_call_duration)
            final_status, error_message = self._final_status_reason, self._final_error_message
            logger.info(f"[{self.agent_id}] Call end signal received. Status: {final_status}")
        except asyncio.TimeoutError:
            logger.warning(f"[{self.agent_id}] Call TIMEOUT ({self.max_call_duration}s).")
            final_status, error_message = "Timeout", "Call exceeded maximum duration."
            if self.call_sid: await self.telephony_wrapper.end_call(self.call_sid) # Ensure call is hung up
        except Exception as e:
            logger.error(f"[{self.agent_id}] Error in call lifecycle: {e}", exc_info=True)
            final_status, error_message = "Error", f"Lifecycle Exception: {e}"
            if self.call_sid: await self.telephony_wrapper.end_call(self.call_sid)
        finally:
            logger.info(f"[{self.agent_id}] Lifecycle task ending. Cleaning up with status: {final_status}")
            if self.is_call_active: await self._cleanup_call(final_status, error_message)

    def signal_call_ended_externally(self, reason: str = "External Stop Signal"):
        logger.info(f"[{self.agent_id}] Received external signal that call ended. Reason: {reason}")
        self._signal_call_end(reason, is_error=False)

    def _signal_call_end(self, reason: str, is_error: bool = False, error_msg: Optional[str] = None):
        if not self.call_end_event.is_set():
            logger.debug(f"[{self.agent_id}] Setting call end event. Reason: {reason}, IsError: {is_error}")
            self._final_status_reason = "Error" if is_error else reason
            self._final_error_message = error_msg if is_error else None
            self.is_call_active = False # Mark inactive immediately when signaling end
            self.call_end_event.set()

    async def _handle_final_transcript(self, transcript: str):
        if not self.is_call_active or not self.call_sid: return
        logger.info(f"[{self.agent_id}][{self.call_sid}] User Transcript: '{transcript}'")
        self.conversation_history.append({"role": "user", "content": transcript})
        await self._cancel_ongoing_tts() # Barge-in
        await self._initiate_agent_turn()

    async def _determine_next_strategic_goal(self) -> Tuple[str, str]:
        """Uses LLM to determine the next conversational micro-goal."""
        if not self.is_call_active: return "ProvideValue_NoAsk", "Call inactive, defaulting."

        # Create a concise history summary for this specific decision prompt
        # e.g., last 2 agent, last 2 user utterances
        history_summary_parts = []
        last_n_turns = 4 # Consider last 2 full exchanges
        start_index = max(0, len(self.conversation_history) - last_n_turns)
        for entry in self.conversation_history[start_index:]:
            history_summary_parts.append(f"{entry['role'].capitalize()}: {entry['content']}")
        history_summary = "\n".join(history_summary_parts)
        if not history_summary: history_summary = "(No prior conversation history for this decision)"


        prompt = (
            f"You are an AI Sales Strategist guiding a sales call. The overall objective is defined in the main system prompt. "
            f"The current sales niche is '{self.target_niche}'.\n"
            f"Recent conversation turns:\n---\n{history_summary}\n---\n"
            f"Previous strategic goal was: '{self._current_strategic_goal}' with reasoning: '{self._last_llm_strategic_reasoning}'.\n"
            f"Considering the overall objective and recent turns, what is the single most effective strategic micro-goal for the *agent's next response*? "
            f"Choose ONLY ONE from this list: {json.dumps(CONVERSATIONAL_STRATEGIC_GOALS)}.\n"
            f"Provide brief `reasoning` for your choice. "
            f"Respond ONLY with a valid JSON object: {{\"next_goal\": \"CHOSEN_GOAL\", \"reasoning\": \"Your brief reasoning.\"}}"
        )
        
        messages = [{"role": "user", "content": prompt}]
        try:
            # Use a fast, capable model for this strategic decision
            # Temperature can be low for consistency in strategic choices
            response_str = await self.llm_client.generate_response(
                messages=messages, model=config.OPENROUTER_MODEL_NAME, # Or a specific strategy model
                temperature=0.2, max_tokens=150, use_cache=False # Strategy should be fresh
            )
            if response_str:
                response_json = json.loads(response_str)
                next_goal = response_json.get("next_goal")
                reasoning = response_json.get("reasoning", "No reasoning provided by strategy LLM.")
                if next_goal in CONVERSATIONAL_STRATEGIC_GOALS:
                    logger.info(f"[{self.agent_id}] Next strategic goal determined by LLM: '{next_goal}'. Reasoning: '{reasoning}'")
                    return next_goal, reasoning
                else:
                    logger.warning(f"[{self.agent_id}] LLM returned invalid strategic goal: '{next_goal}'. Defaulting. LLM Response: {response_str}")
            else:
                logger.warning(f"[{self.agent_id}] LLM failed to determine next strategic goal. Defaulting.")
        except json.JSONDecodeError:
            logger.error(f"[{self.agent_id}] Failed to parse JSON from strategic goal LLM. Response: {response_str}. Defaulting.")
        except Exception as e:
            logger.error(f"[{self.agent_id}] Error determining next strategic goal: {e}. Defaulting.", exc_info=True)
        
        # Default fallback goal
        return "ClarifyProspectStatement" if self.conversation_history else "BuildRapport", "Defaulting due to error or invalid LLM response."


    async def _initiate_agent_turn(self):
        if not self.is_call_active or not self.call_sid: return

        try:
            # 1. Determine strategic goal for this turn
            self._current_strategic_goal, self._last_llm_strategic_reasoning = await self._determine_next_strategic_goal()

            # 2. Construct messages for LLM to generate speech
            # The system prompt (self.system_prompt) contains persona, overall objectives, and context.
            llm_messages = [{"role": "system", "content": self.system_prompt}]
            llm_messages.extend(self.conversation_history) # Add full conversation history
            
            # Add the specific instruction for this turn based on the strategic goal
            llm_messages.append({
                "role": "system", # Or 'user' if it makes the LLM perform better for this instruction
                "content": f"[Internal Instruction for Alex]: Your immediate strategic micro-goal for this response is: '{self._current_strategic_goal}'. "
                           f"Reasoning: '{self._last_llm_strategic_reasoning}'. "
                           f"Craft your response to naturally achieve this goal while adhering to all conversational mandates."
            })
            
            # 3. Generate response from LLM
            logger.debug(f"[{self.agent_id}] Generating LLM speech. Current strategic goal: {self._current_strategic_goal}")
            llm_response_text = await self.llm_client.generate_response(
                messages=llm_messages, model=config.OPENROUTER_MODEL_NAME, # Use main conversational model
                temperature=0.7, max_tokens=300 # Adjust as needed for speech length
            )

            if not llm_response_text or "error:" in llm_response_text.lower(): # Basic check for LLM error string
                 logger.error(f"[{self.agent_id}] LLM failed to generate response or returned error. Text: '{llm_response_text}'")
                 # Fallback response if LLM fails critically
                 llm_response_text = "I'm sorry, I seem to be having a momentary issue. Could you please repeat that?"
                 self.conversation_history.append({"role": "assistant", "content": llm_response_text}) # Log fallback
                 # Don't update strategic goal if we're just saying sorry
            else:
                self.conversation_history.append({"role": "assistant", "content": llm_response_text})
                logger.info(f"[{self.agent_id}][{self.call_sid}] Agent Speech (Goal: {self._current_strategic_goal}): '{llm_response_text[:100]}...'")


            # 4. Stream TTS
            if self.is_call_active: # Check again, call might have ended during LLM generation
                self.current_tts_task = asyncio.create_task(self._stream_tts(llm_response_text))
            else:
                logger.warning(f"[{self.agent_id}] Call became inactive before TTS could start for goal {self._current_strategic_goal}.")

        except Exception as e:
            error_msg = f"Error initiating agent turn (goal: {self._current_strategic_goal}): {e}"
            logger.error(f"[{self.agent_id}] {error_msg}", exc_info=True)
            # Attempt a graceful error message via TTS if possible
            if self.is_call_active:
                try:
                    fallback_error_speech = "My apologies, I encountered an unexpected issue. Let's try to get back on track."
                    self.conversation_history.append({"role": "assistant", "content": fallback_error_speech})
                    self.current_tts_task = asyncio.create_task(self._stream_tts(fallback_error_speech))
                except Exception as tts_err:
                    logger.error(f"[{self.agent_id}] Failed to even stream fallback error TTS: {tts_err}")
                    await self._handle_fatal_error(error_msg) # If TTS itself fails, signal fatal
            else: # If call not active, just signal fatal error
                 await self._handle_fatal_error(error_msg)


    async def _stream_tts(self, text_to_speak: str): # Mostly same, ensures stream_sid is used
        if not self.is_call_active or not self.call_sid or not self.stream_sid:
            logger.warning(f"[{self.agent_id}] Cannot stream TTS: call/stream inactive or SID missing.")
            return
        
        tts_start_time = time.monotonic()
        logger.info(f"[{self.agent_id}] Starting TTS stream for: '{text_to_speak[:70]}...'")
        await self.send_mark_callback(self.call_sid, f"tts_start_{self._current_strategic_goal}")
        try:
            async for audio_chunk in self.voice_handler.speak_text(text_to_speak):
                if not self.is_call_active: break # Stop if call ended mid-TTS
                if audio_chunk:
                    await self.send_audio_callback(self.call_sid, audio_chunk) # Uses self.stream_sid via callback closure
            
            if self.is_call_active: # Only send end mark if call still active
                await self.send_mark_callback(self.call_sid, f"tts_end_{self._current_strategic_goal}")
            tts_duration = time.monotonic() - tts_start_time
            logger.info(f"[{self.agent_id}] Finished TTS streaming ({tts_duration:.2f}s).")
        except asyncio.CancelledError:
             logger.info(f"[{self.agent_id}] TTS streaming task cancelled (likely barge-in).")
        except Exception as e:
             logger.error(f"[{self.agent_id}] Error during TTS streaming: {e}", exc_info=True)
             # Don't signal fatal error from here directly, let main LLM turn handler decide
             # or rely on voice_handler error callback if it's a Deepgram issue.

    async def _cancel_ongoing_tts(self): # Same as before
        if self.current_tts_task and not self.current_tts_task.done():
            logger.warning(f"[{self.agent_id}] Cancelling ongoing TTS task.")
            self.current_tts_task.cancel()
            try: await self.current_tts_task
            except asyncio.CancelledError: logger.info(f"[{self.agent_id}] TTS task cancelled successfully.")
            finally: self.current_tts_task = None
    
    async def _handle_voice_handler_error(self, error_message: str):
        logger.error(f"[{self.agent_id}] Fatal error from VoiceHandler: {error_message}")
        await self._handle_fatal_error(f"VoiceHandler Error: {error_message}")

    async def _handle_fatal_error(self, error_message: str):
         logger.error(f"[{self.agent_id}][{self.call_sid}] Handling FATAL error: {error_message}")
         self._signal_call_end("Fatal Error", is_error=True, error_msg=error_message)
         # The lifecycle manager will call _cleanup_call.
         # If there's an orchestrator callback for errors, it will be invoked by _cleanup_call.

    async def _generate_call_summary_with_llm(self) -> Dict[str, Any]:
        """Generates a structured summary of the call using LLM for logging and learning."""
        if not self.conversation_history:
            return {"summary_status": "no_conversation_history"}

        logger.info(f"[{self.agent_id}] Generating LLM call summary for call SID {self.call_sid}...")
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_history])
        
        prompt = (
            f"You are a sales call analyst. Analyze the following sales call transcript involving 'Alex' (the AI Sales Agent) and a 'User' (the prospect).\n"
            f"The call was for Boutique AI, selling AI Sales Agent solutions to businesses in the '{self.target_niche}' niche.\n"
            f"Transcript:\n---\n{history_str}\n---\n"
            f"Based on the entire transcript, provide a structured JSON summary with the following keys:\n"
            f"- `overall_call_sentiment` (string: Positive, Neutral, Negative, Mixed)\n"
            f"- `prospect_engagement_level` (string: High, Medium, Low)\n"
            f"- `key_topics_discussed` (list of strings)\n"
            f"- `identified_prospect_pain_points` (list of strings, be specific)\n"
            f"- `objections_raised_by_prospect` (list of strings, verbatim or summarized)\n"
            f"- `agent_objection_handling_effectiveness` (string: Effective, PartiallyEffective, Ineffective, NotApplicable)\n"
            f"- `key_value_propositions_resonated` (list of strings, if any)\n"
            f"- `agreed_next_steps` (string, e.g., 'Demo scheduled for YYYY-MM-DD HH:MM', 'Follow-up email with case studies', 'None')\n"
            f"- `final_call_outcome_category` (string: Meeting_Booked, Strong_Interest_Followup, Mild_Interest_Nurture, Not_Interested_Polite, Not_Interested_Abrupt, Disqualified, Error_Technical, Voicemail)\n"
            f"- `agent_performance_strengths` (list of strings, e.g., 'Good rapport building', 'Clear explanation of X')\n"
            f"- `agent_areas_for_improvement` (list of strings, e.g., 'Could have probed deeper on Y objection', 'Missed cue for Z')\n"
            f"Respond ONLY with the valid JSON object."
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            response_str = await self.llm_client.generate_response(
                messages=messages, model=config.OPENROUTER_MODEL_NAME, # Use a capable model for analysis
                temperature=0.2, max_tokens=1000, use_cache=False # Analysis should be fresh
            )
            if response_str:
                summary_obj = json.loads(response_str)
                logger.info(f"[{self.agent_id}] LLM Call summary generated. Outcome category: {summary_obj.get('final_call_outcome_category')}")
                return summary_obj
            else:
                logger.warning(f"[{self.agent_id}] LLM failed to generate call summary.")
                return {"summary_status": "llm_no_response"}
        except json.JSONDecodeError:
            logger.error(f"[{self.agent_id}] Failed to parse JSON from LLM call summary. Response: {response_str[:200]}")
            return {"summary_status": "llm_json_parse_error", "raw_response": response_str}
        except Exception as e:
            logger.error(f"[{self.agent_id}] Error generating LLM call summary: {e}", exc_info=True)
            return {"summary_status": f"llm_exception_{type(e).__name__}"}


    async def _cleanup_call(self, final_status: str, error_message: Optional[str]):
        if self.call_sid is None: logger.warning(f"[{self.agent_id}] Cleanup called but no call_sid. Already cleaned or never started?"); return
        
        current_call_sid_for_cleanup = self.call_sid # Store before reset
        logger.info(f"[{self.agent_id}][{current_call_sid_for_cleanup}] Cleaning up call resources. Final Status: {final_status}")

        # Ensure is_call_active is false to prevent further actions
        self.is_call_active = False 
        if not self.call_end_event.is_set(): self.call_end_event.set() # Ensure lifecycle task can exit

        await self._cancel_ongoing_tts()
        if self.call_lifecycle_task and not self.call_lifecycle_task.done():
            if asyncio.current_task() is not self.call_lifecycle_task: # Don't cancel self
                self.call_lifecycle_task.cancel()
                try: await self.call_lifecycle_task
                except asyncio.CancelledError: pass

        if self.voice_handler.is_connected: await self.voice_handler.disconnect()
        
        # Ensure telephony call is terminated (TelephonyWrapper should handle if already ended)
        await self.telephony_wrapper.end_call(current_call_sid_for_cleanup)

        # Generate structured call summary with LLM for rich logging
        llm_summary_obj = await self._generate_call_summary_with_llm()
        
        # Update CRM with final outcome and rich summary
        crm_notes = f"Call ended. Final Status: {final_status}."
        if error_message: crm_notes += f" Error: {error_message}."
        if llm_summary_obj.get("summary_status", "").startswith("llm_"): # If summary failed
            crm_notes += f" LLM Summary Status: {llm_summary_obj['summary_status']}."
        
        # Determine contact_id for logging
        contact_id_for_log = self.initial_prospect_data.get("id") if self.initial_prospect_data else None
        if not contact_id_for_log and self.crm_wrapper.supabase: # Try to fetch by phone if ID wasn't in initial data
            contact = await self.crm_wrapper.get_contact_info(self.target_phone_number)
            if contact: contact_id_for_log = contact.get("id")

        await self.crm_wrapper.log_call_outcome(
            call_sid=current_call_sid_for_cleanup,
            contact_id=contact_id_for_log,
            agent_id=self.agent_id,
            status=final_status, # This is the system status (e.g., Completed, Error, Timeout)
            notes=crm_notes,
            conversation_history_json=self.conversation_history,
            call_duration_seconds=int(time.monotonic() - self._call_start_time) if hasattr(self, '_call_start_time') else None,
            llm_call_summary_json=llm_summary_obj if not llm_summary_obj.get("summary_status") else None, # Only log if successful
            key_objections_tags=llm_summary_obj.get("objections_raised_by_prospect"),
            prospect_engagement_signals_tags=llm_summary_obj.get("key_value_propositions_resonated"), # Example mapping
            agent_perceived_call_outcome=llm_summary_obj.get("final_call_outcome_category") # LLM's perception
        )
        
        # Update contact status based on LLM's suggested next action or outcome category
        if contact_id_for_log and llm_summary_obj.get("final_call_outcome_category"):
            new_contact_status = f"PostCall - {llm_summary_obj['final_call_outcome_category']}"
            await self.crm_wrapper.update_contact_status_and_notes(contact_id_for_log, new_contact_status, f"Call {current_call_sid_for_cleanup} ended. Outcome: {llm_summary_obj['final_call_outcome_category']}")

        # Notify Orchestrator/Server
        if self.on_call_complete_callback and not error_message:
            try: await self.on_call_complete_callback(self.agent_id, final_status, self.conversation_history, llm_summary_obj)
            except Exception as cb_e: logger.error(f"Error in on_call_complete_callback: {cb_e}")
        elif self.on_call_error_callback and error_message:
            try: await self.on_call_error_callback(self.agent_id, error_message)
            except Exception as cb_e: logger.error(f"Error in on_call_error_callback: {cb_e}")

        # Reset state for potential reuse by orchestrator (though usually a new instance is made)
        self.call_sid = None; self.stream_sid = None; self.current_tts_task = None
        self.call_lifecycle_task = None; self.conversation_history = []
        # self.call_end_event.clear() # Already cleared in start_sales_call
        logger.info(f"[{self.agent_id}][{current_call_sid_for_cleanup}] Cleanup complete.")

    async def handle_incoming_audio(self, audio_chunk: bytes):
         if self.is_call_active: await self.voice_handler.send_audio_chunk(audio_chunk)

    async def stop_call(self, reason: str = "External Stop Request (Orchestrator)"):
         logger.info(f"[{self.agent_id}][{self.call_sid}] Received external stop request. Reason: {reason}")
         self._signal_call_end(reason, is_error=False)
         # Lifecycle manager will handle cleanup. If task is already running, this signal will make it exit.
         # If lifecycle task isn't running or stuck, direct cleanup might be needed by caller.
         if self.call_lifecycle_task and self.call_lifecycle_task.done() and self.is_call_active:
             logger.warning(f"[{self.agent_id}] stop_call invoked, but lifecycle task is done and call still marked active. Forcing cleanup.")
             await self._cleanup_call(reason, None)


if __name__ == "__main__":
    # This agent is complex and designed to be run by the server.py orchestrator.
    # Direct testing would require mocking many dependencies and callbacks.
    print("SalesAgent (Level 45 - AI Sales Virtuoso) defined. Designed for use via server orchestrator.")
    # Add conceptual test runner if desired, similar to other agents, mocking dependencies.
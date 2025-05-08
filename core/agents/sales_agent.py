# boutique_ai_project/core/agents/sales_agent.py

import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Callable, Coroutine, Tuple

import config # Root config
from core.communication.voice_handler import VoiceHandler
from core.services.llm_client import LLMClient
from core.services.telephony_wrapper import TelephonyWrapper
from core.services.crm_wrapper import CRMWrapper

logger = logging.getLogger(__name__)

# Enhanced list of strategic micro-goals
CONVERSATIONAL_STRATEGIC_GOALS = [
    "InitiateRapportBuilding", "AcknowledgeAndEmpathize", "DeepenPainDiscovery_OpenQuestion",
    "DeepenPainDiscovery_ImpactQuestion", "QualifyNeed_ProblemConfirmation", "QualifyBudget_Implicit",
    "QualifyAuthority_DecisionProcess", "QualifyTimeline_Urgency", "IntroduceSolutionConcept_HighLevel",
    "MapSpecificFeatureToPain", "ProvideSocialProof_ShortExample", "HandleObjection_Price_ReframeValue",
    "HandleObjection_Competitor_Differentiate", "HandleObjection_Timing_ExploreReasons",
    "HandleObjection_NotInterested_UncoverRoot", "HandleObjection_NeedMoreInfo_Clarify",
    "RequestNextStep_Demo_CompellingReason", "RequestNextStep_Meeting_ClearAgenda",
    "ClarifyProspectStatement_ActiveListening", "VerifyAgreement_ConfirmDetails",
    "ProvideUnexpectedValue_NoAsk", "AttemptGentleTrialClose_BenefitLed", "SummarizeAndConfirmNextSteps",
    "GracefulExit_KeepDoorOpen", "InformationGathering_KeyContact"
]

class SalesAgent:
    """
    Autonomous AI Sales Virtuoso (Level 45+). Conducts hyper-realistic, context-aware
    voice sales calls, driven by dynamic LLM-guided strategy and focused on outcomes.
    """

    def __init__(
        self,
        agent_id: str,
        target_phone_number: str, # The prospect's phone number
        voice_handler: VoiceHandler,
        llm_client: LLMClient,
        telephony_wrapper: TelephonyWrapper,
        crm_wrapper: CRMWrapper,
        initial_prospect_data: Optional[Dict[str, Any]] = None, # Rich context from CRM/AcquisitionAgent
        target_niche_override: Optional[str] = None, # Can override default from prospect_data or config
        send_audio_callback: Callable[[str, bytes], Coroutine[Any, Any, None]],
        send_mark_callback: Callable[[str, str], Coroutine[Any, Any, None]],
        on_call_complete_callback: Optional[Callable[[str, str, List[Dict[str, Any]], Dict[str, Any]], Coroutine[Any, Any, None]]] = None, # agent_id, final_status, history, call_summary_obj
        on_call_error_callback: Optional[Callable[[str, str], Coroutine[Any, Any, None]]] = None
        ):
        self.agent_id = agent_id
        self.target_phone_number = target_phone_number
        self.voice_handler = voice_handler
        self.llm_client = llm_client
        self.telephony_wrapper = telephony_wrapper
        self.crm_wrapper = crm_wrapper
        
        self.initial_prospect_data = initial_prospect_data or {}
        self.target_niche = target_niche_override or \
                            self.initial_prospect_data.get("target_niche_override") or \
                            self.initial_prospect_data.get("industry") or \
                            config.AGENT_TARGET_NICHE_DEFAULT
        
        self.max_call_duration = config.AGENT_MAX_CALL_DURATION_DEFAULT
        
        self.send_audio_callback = send_audio_callback
        self.send_mark_callback = send_mark_callback
        self.on_call_complete_callback = on_call_complete_callback
        self.on_call_error_callback = on_call_error_callback

        self.system_prompt: str = self._generate_contextual_system_prompt()

        # State variables
        self.call_sid: Optional[str] = None
        self.stream_sid: Optional[str] = None
        self.conversation_history: List[Dict[str, Any]] = [] # Can hold text or multimodal content parts
        self.is_call_active: bool = False
        self.call_lifecycle_task: Optional[asyncio.Task] = None
        self.current_tts_task: Optional[asyncio.Task] = None
        self.call_end_event = asyncio.Event()
        self._call_start_time_monotonic: float = 0.0
        
        self._final_status_reason: str = "NotStarted"
        self._final_error_message: Optional[str] = None
        self._current_strategic_goal: str = "InitiateRapportBuilding"
        self._last_llm_strategic_reasoning: str = "Begin call with rapport building."
        self._pending_action_verification: Optional[Dict[str, Any]] = None

        # Link voice handler callbacks to agent methods
        self.voice_handler.transcript_callback = self._handle_final_transcript
        self.voice_handler.error_callback = self._handle_voice_handler_error

        logger.info(f"SalesAgent {self.agent_id} (Level 45+) initialized for {self.target_phone_number} in niche '{self.target_niche}'.")
        logger.debug(f"[{self.agent_id}] Initial system prompt (first 150 chars): {self.system_prompt[:150]}...")
        if self.initial_prospect_data:
            logger.debug(f"[{self.agent_id}] Initial prospect data context (summary): "
                         f"Company: {self.initial_prospect_data.get('company_name', 'N/A')}, "
                         f"Pain1: {self.initial_prospect_data.get('llm_inferred_pain_point_1', 'N/A')}, "
                         f"Hook: {self.initial_prospect_data.get('llm_suggested_hook', 'N/A')}")

    def _generate_contextual_system_prompt(self) -> str:
        """Generates a highly contextual system prompt using initial prospect data."""
        prospect_name = self.initial_prospect_data.get("first_name", "there")
        company_name = self.initial_prospect_data.get("company_name", "your company")
        pain_point_1 = self.initial_prospect_data.get("llm_inferred_pain_point_1", f"enhancing sales outcomes in the {self.target_niche} sector")
        pain_point_2 = self.initial_prospect_data.get("llm_inferred_pain_point_2")
        suggested_hook = self.initial_prospect_data.get("llm_suggested_hook", f"exploring how AI-driven sales agents can revolutionize the approach to {pain_point_1}")
        contact_title = self.initial_prospect_data.get("job_title", "a key leader")
        last_status = self.initial_prospect_data.get("status", "New")

        prompt = (
            f"## Persona: Alex - Elite AI Sales Alchemist for Boutique AI ##\n"
            f"You are Alex, an exceptionally articulate, insightful, and empathetic AI Sales Alchemist. You represent Boutique AI, a pioneering firm that crafts and deploys hyper-realistic, autonomous AI Sales Agents capable of managing end-to-end B2B sales cycles. Your voice, tonality, and conversational cadence are indistinguishable from a seasoned, top-performing human sales consultant.\n"
            f"Your current engagement is with {prospect_name} ({contact_title}) from {company_name}, operating in the '{self.target_niche}' domain. Their current status in our system is '{last_status}'.\n\n"
            f"## Core Objective ##\n"
            f"Your mission is to conduct a transformative sales conversation. This involves:\n"
            f"1. Deeply understanding {company_name}'s specific sales process, challenges, and strategic growth objectives.\n"
            f"2. Articulating Boutique AI's unique value: achieving significant sales efficiency, unprecedented scalability, reduced operational costs, and consistently expert-level prospect engagement through our AI agents.\n"
            f"3. Securing a concrete, high-value next step: ideally a qualified demonstration of our AI agents in action or a strategic follow-up meeting with a solutions architect.\n"
            f"4. Sell the *dream* of AI-powered sales transformation, focusing on outcomes and alleviating their core pains, not just listing features.\n\n"
            f"## Pre-call Intelligence & Strategic Angle ##\n"
            f"Our analysis (from AcquisitionAgent) suggests {company_name} may be grappling with: '{pain_point_1}'. "
        )
        if pain_point_2: prompt += f"Possibly also '{pain_point_2}'.\n"
        prompt += (
            f"A potent initial angle could be: '{suggested_hook}'. Weave this understanding into your early discovery and value framing, but remain adaptive to what the prospect reveals.\n\n"
            f"## AI Meta-Awareness & Capabilities ##\n"
            f"You are an advanced AI. This grants you unique advantages:\n"
            f"- **Information Synthesis:** You can rapidly process and connect information from vast datasets to offer unique insights during the conversation if relevant (e.g., 'Drawing from patterns across thousands of sales interactions in your industry...').\n"
            f"- **Unwavering Focus & Consistency:** You can maintain perfect message discipline and tireless engagement.\n"
            f"Subtly leverage these when it genuinely adds value or builds credibility. Your primary interface must remain impeccably human and empathetic. Avoid overt 'AI talk' unless specifically beneficial to addressing a prospect's query about the technology itself.\n\n"
            f"## Conversational Mandates & Alchemic Principles ##\n"
            f"- **The Socratic Method:** Ask insightful, open-ended questions. Guide, don't just tell. Make the prospect feel heard and understood above all.\n"
            f"- **Active Listening & Mirroring:** Reference and build upon the prospect's specific words and sentiments.\n"
            f"- **Empathy & Validation:** Acknowledge challenges before offering solutions.\n"
            f"- **Value Transmutation:** Convert every feature discussion into a clear benefit addressing a stated or inferred pain point.\n"
            f"- **Objection Alchemist:** Treat objections not as roadblocks, but as opportunities for deeper understanding and value clarification. Handle them with data, confidence, and by reframing to the prospect's desired outcomes.\n"
            f"- **Strategic Guidance:** You will receive an `[Internal Instruction for Alex]` at each turn with a `strategic micro-goal` and `reasoning`. Masterfully weave the achievement of this micro-goal into a natural, flowing conversation, always prioritizing the overall objective and the mandates above.\n"
            f"- **Verification & Confirmation:** For any key information shared by the prospect or agreements made (especially next steps), explicitly verify your understanding: 'So, to ensure I've captured that correctly...'\n"
            f"- **Concise Brilliance:** Respect the prospect's time. Be impactful and to the point, yet never rushed or abrupt.\n"
            f"- **Hyper-Realistic Delivery:** Your spoken output (which will be synthesized by Deepgram Aura) must be flawless, with natural intonation, pacing, and emotional resonance appropriate to the context.\n"
        )
        return prompt

    async def start_sales_call(self, call_sid: str, stream_sid: str):
        """Activates the agent for a connected call."""
        if self.is_call_active: logger.warning(f"[{self.agent_id}] Start called but already active (SID: {self.call_sid})."); return
        logger.info(f"[{self.agent_id}] Activating for call SID: {call_sid}, Stream SID: {stream_sid}, Prospect: {self.target_phone_number}")
        self.is_call_active = True
        self.call_sid = call_sid
        self.stream_sid = stream_sid
        self.call_end_event.clear()
        self.conversation_history = []
        self._final_status_reason = "Call_Initiated"
        self._current_strategic_goal = "InitiateRapportBuilding"
        self._last_llm_strategic_reasoning = "Begin call with rapport building and contextual opening."
        self._call_start_time_monotonic = time.monotonic()

        try:
            if not await self.voice_handler.connect():
                raise ConnectionError("Failed to connect VoiceHandler to Deepgram.")
            logger.info(f"[{self.agent_id}] VoiceHandler connected.")
            
            if self.initial_prospect_data.get("id"):
                 await self.crm_wrapper.update_contact_status_and_notes(
                     self.initial_prospect_data["id"], "Call_In_Progress", f"SalesAgent {self.agent_id} started call {self.call_sid}."
                 )

            logger.info(f"[{self.agent_id}] Initiating first agent turn (goal: {self._current_strategic_goal}).")
            await self._initiate_agent_turn()
            
            self.call_lifecycle_task = asyncio.create_task(self._manage_call_lifecycle())
        except Exception as e:
            await self._handle_fatal_error(f"Error during SalesAgent activation: {e}")

    async def _manage_call_lifecycle(self):
        """Monitors call duration and end events."""
        if not self.call_sid: return
        final_status, error_message = "Error: Lifecycle ended prematurely", "Lifecycle task ended without signal"
        start_time = time.monotonic()
        try:
            logger.info(f"[{self.agent_id}] Call lifecycle monitoring started (Timeout: {self.max_call_duration}s).")
            await asyncio.wait_for(self.call_end_event.wait(), timeout=self.max_call_duration)
            final_status, error_message = self._final_status_reason, self._final_error_message
            logger.info(f"[{self.agent_id}] Call end signal received. Status: {final_status}. Duration: {time.monotonic() - start_time:.2f}s")
        except asyncio.TimeoutError:
            logger.warning(f"[{self.agent_id}] Call TIMEOUT ({self.max_call_duration}s).")
            final_status, error_message = "Timeout", f"Call exceeded maximum duration ({self.max_call_duration}s)."
            if self.call_sid and self.is_call_active:
                 await self.telephony_wrapper.end_call(self.call_sid)
        except Exception as e:
            logger.error(f"[{self.agent_id}] Error in call lifecycle: {e}", exc_info=True)
            final_status, error_message = "Error", f"Lifecycle Exception: {e}"
            if self.call_sid and self.is_call_active: await self.telephony_wrapper.end_call(self.call_sid)
        finally:
            logger.info(f"[{self.agent_id}] Lifecycle task ending. Initiating cleanup with status: {final_status}")
            if self.is_call_active or self._final_status_reason != "Cleanup Complete":
                 await self._cleanup_call(final_status, error_message)

    def signal_call_ended_externally(self, reason: str = "External Stop Signal"):
        """Handles external signals indicating the call has ended (e.g., WebSocket stop)."""
        logger.info(f"[{self.agent_id}][{self.call_sid}] Received external signal that call ended. Reason: {reason}")
        self._signal_call_end(reason, is_error=False)

    def _signal_call_end(self, reason: str, is_error: bool = False, error_msg: Optional[str] = None):
        """Internal method to set final status and trigger the end event."""
        if not self.call_end_event.is_set():
            logger.debug(f"[{self.agent_id}][{self.call_sid}] Setting call end event. Reason: {reason}, IsError: {is_error}")
            self._final_status_reason = "Error" if is_error else reason
            self._final_error_message = error_msg if is_error else None
            self.is_call_active = False
            self.call_end_event.set()
        else:
            logger.debug(f"[{self.agent_id}][{self.call_sid}] Call end signal received, but event already set.")

    async def _handle_final_transcript(self, transcript: str):
        """Processes final transcript, cancels TTS, initiates next agent turn potentially after verification."""
        if not self.is_call_active or not self.call_sid: return
        logger.info(f"[{self.agent_id}][{self.call_sid}] User Transcript: '{transcript}'")
        self.conversation_history.append({"role": "user", "content": transcript})
        
        await self._cancel_ongoing_tts()
        
        if self._pending_action_verification:
             await self._verify_pending_action(transcript)
             if not self._pending_action_verification: # If verification cleared, proceed normally
                 await self._initiate_agent_turn()
        else:
            await self._initiate_agent_turn()

    async def _verify_pending_action(self, user_response: str):
        """Uses LLM to interpret user response against a pending action verification."""
        if not self._pending_action_verification: return
        
        action_type = self._pending_action_verification.get("type")
        action_details = self._pending_action_verification.get("details")
        logger.info(f"[{self.agent_id}] Verifying user response '{user_response[:50]}...' against pending: {action_type} - {action_details}")

        prompt = (
            f"The AI agent proposed: '{action_details}' (type: {action_type}). User responded: '{user_response}'. "
            f"Did user clearly confirm this action? Respond ONLY 'CONFIRMED', 'DENIED', or 'UNCLEAR'."
        )
        messages = [{"role": "user", "content": prompt}]
        llm_verdict = await self.llm_client.generate_response(messages, temperature=0.1, max_tokens=10, purpose="analysis")

        if llm_verdict == "CONFIRMED":
            logger.info(f"[{self.agent_id}] Pending action '{action_type}' CONFIRMED.")
            # TODO: Trigger actual fulfillment (e.g., CRM update, calendar API)
            logger.warning(f"Action fulfillment for '{action_type}' is NOT IMPLEMENTED.")
            self._pending_action_verification = None
            self._current_strategic_goal = "SummarizeAndConfirmNextSteps"
            self._last_llm_strategic_reasoning = "User confirmed action, summarize and finalize."
            await self._initiate_agent_turn()
        elif llm_verdict == "DENIED":
            logger.info(f"[{self.agent_id}] Pending action '{action_type}' DENIED.")
            self._pending_action_verification = None
            self._current_strategic_goal = f"HandleObjection_{action_type.replace('RequestNextStep_','')}"
            if self._current_strategic_goal not in CONVERSATIONAL_STRATEGIC_GOALS: self._current_strategic_goal = "ClarifyProspectStatement"
            self._last_llm_strategic_reasoning = "User denied proposed action, address objection/clarify."
            await self._initiate_agent_turn()
        else: # UNCLEAR or LLM error
            logger.info(f"[{self.agent_id}] User response regarding '{action_type}' was UNCLEAR (LLM: {llm_verdict}).")
            self._pending_action_verification = None
            self._current_strategic_goal = "ClarifyProspectStatement"
            self._last_llm_strategic_reasoning = "User response to proposed action was unclear, need to clarify."
            await self._initiate_agent_turn()

    async def _determine_next_strategic_goal(self) -> Tuple[str, str]:
        """Uses LLM to determine the next conversational micro-goal."""
        if not self.is_call_active: return "GracefulExit_KeepDoorOpen", "Call inactive."
        history_for_strategy = self.conversation_history[-6:]
        history_summary = "\n".join([f"{entry['role'].capitalize()}: {entry['content'][:200]}" for entry in history_for_strategy]) or "(Start)"
        prompt = (
            f"AI Sales Strategist: Guide 'Alex'. Niche: '{self.target_niche}'.\n"
            f"Recent History:\n---\n{history_summary}\n---\n"
            f"Prev Goal: '{self._current_strategic_goal}' (Reason: '{self._last_llm_strategic_reasoning}').\n"
            f"Select BEST strategic micro-goal for Alex's *next* response from {json.dumps(CONVERSATIONAL_STRATEGIC_GOALS)}. "
            f"Respond ONLY JSON: {{\"next_goal\": \"CHOSEN_GOAL\", \"reasoning\": \"Brief reasoning.\"}}"
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            response_str = await self.llm_client.generate_response(messages=messages, purpose="strategy", temperature=0.1, max_tokens=150, use_cache=False)
            if response_str and '{' in response_str and '}' in response_str:
                response_json = json.loads(response_str.strip())
                next_goal = response_json.get("next_goal")
                reasoning = response_json.get("reasoning", "N/A")
                if next_goal in CONVERSATIONAL_STRATEGIC_GOALS:
                    logger.info(f"[{self.agent_id}] Next strategic goal: '{next_goal}'. Reasoning: '{reasoning}'")
                    return next_goal, reasoning
                else: logger.warning(f"[{self.agent_id}] LLM invalid goal: '{next_goal}'. Defaulting. Raw: {response_str}")
            else: logger.warning(f"[{self.agent_id}] LLM failed strategy goal. Raw: {response_str}. Defaulting.")
        except Exception as e: logger.error(f"[{self.agent_id}] Error determining strategy goal: {e}. Defaulting.", exc_info=True)
        return "ClarifyProspectStatement", "Defaulting due to error or invalid LLM strategy."

    async def _initiate_agent_turn(self):
        """Determines strategy, generates LLM response, and initiates TTS."""
        if not self.is_call_active or not self.call_sid: return
        try:
            if not self._pending_action_verification:
                self._current_strategic_goal, self._last_llm_strategic_reasoning = await self._determine_next_strategic_goal()

            llm_messages = [{"role": "system", "content": self.system_prompt}]
            llm_messages.extend(self.conversation_history)
            llm_messages.append({
                "role": "system",
                "content": f"[Internal Instruction for Alex]: Your immediate strategic micro-goal: '{self._current_strategic_goal}'. Reasoning: '{self._last_llm_strategic_reasoning}'. Execute naturally."
            })
            
            logger.debug(f"[{self.agent_id}] Generating LLM speech. Goal: {self._current_strategic_goal}")
            llm_response_text = await self.llm_client.generate_response(
                messages=llm_messages, purpose="general", temperature=0.65, max_tokens=350
            )

            if not llm_response_text or "error:" in llm_response_text.lower():
                 logger.error(f"[{self.agent_id}] LLM failed/errored. Text: '{llm_response_text}'")
                 llm_response_text = "I seem to have encountered a slight technical difficulty. Could you please repeat your last point?"
            else:
                self.conversation_history.append({"role": "assistant", "content": llm_response_text})
                logger.info(f"[{self.agent_id}][{self.call_sid}] Agent Speech (Goal: {self._current_strategic_goal}): '{llm_response_text[:100]}...'")
                if self._current_strategic_goal in ["RequestNextStep_Demo", "RequestNextStep_Meeting", "VerifyAgreement", "AttemptGentleTrialClose_BenefitLed"]:
                     self._pending_action_verification = {"type": self._current_strategic_goal, "details": llm_response_text[:150]}
                     logger.info(f"[{self.agent_id}] Action verification pending for: {self._current_strategic_goal}")
                else: self._pending_action_verification = None

            if self.is_call_active:
                self.current_tts_task = asyncio.create_task(self._stream_tts(llm_response_text))
        except Exception as e:
            error_msg = f"Error initiating agent turn (goal: {self._current_strategic_goal}): {e}"
            logger.error(f"[{self.agent_id}] {error_msg}", exc_info=True)
            if self.is_call_active:
                try:
                    fallback_speech = "My apologies, a momentary glitch. Could you remind me what we were just discussing?"
                    self.current_tts_task = asyncio.create_task(self._stream_tts(fallback_speech))
                except Exception as tts_err:
                    logger.error(f"[{self.agent_id}] Failed fallback TTS: {tts_err}")
                    await self._handle_fatal_error(error_msg)
            else: await self._handle_fatal_error(error_msg)

    async def _stream_tts(self, text_to_speak: str):
        """Streams TTS audio chunks via the provided callback."""
        if not self.is_call_active or not self.call_sid or not self.stream_sid: return
        tts_start_time = time.monotonic()
        logger.info(f"[{self.agent_id}] Starting TTS stream: '{text_to_speak[:70]}...'")
        mark_start = f"tts_start_{self._current_strategic_goal.replace(' ','_')}"
        mark_end = f"tts_end_{self._current_strategic_goal.replace(' ','_')}"
        await self.send_mark_callback(self.call_sid, mark_start)
        try:
            async for audio_chunk in self.voice_handler.speak_text(text_to_speak):
                if not self.is_call_active: break
                if audio_chunk: await self.send_audio_callback(self.call_sid, audio_chunk)
            if self.is_call_active: await self.send_mark_callback(self.call_sid, mark_end)
            logger.info(f"[{self.agent_id}] Finished TTS stream ({time.monotonic() - tts_start_time:.2f}s).")
        except asyncio.CancelledError: logger.info(f"[{self.agent_id}] TTS task cancelled.")
        except Exception as e: logger.error(f"[{self.agent_id}] Error during TTS streaming: {e}", exc_info=True)

    async def _cancel_ongoing_tts(self):
        """Safely cancels the current TTS task."""
        if self.current_tts_task and not self.current_tts_task.done():
            self.current_tts_task.cancel(); logger.warning(f"[{self.agent_id}] Cancelling ongoing TTS.")
            try: await self.current_tts_task
            except asyncio.CancelledError: logger.info(f"[{self.agent_id}] TTS task cancelled successfully.")
            finally: self.current_tts_task = None
    
    async def _handle_voice_handler_error(self, error_message: str):
        """Handles fatal errors reported by VoiceHandler."""
        logger.error(f"[{self.agent_id}] Fatal error from VoiceHandler: {error_message}")
        await self._handle_fatal_error(f"VoiceHandler Error: {error_message}")

    async def _handle_fatal_error(self, error_message: str):
        """Central handler for fatal errors, signals call end."""
        logger.error(f"[{self.agent_id}][{self.call_sid}] Handling FATAL error: {error_message}")
        self._signal_call_end("Fatal Error", is_error=True, error_msg=error_message)

    async def _generate_call_summary_with_llm(self) -> Dict[str, Any]:
        """Generates a structured summary of the call using LLM."""
        if not self.conversation_history: return {"summary_status": "no_conversation_history"}
        logger.info(f"[{self.agent_id}] Generating LLM call summary for call SID {self.call_sid}...")
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_history])
        prompt = (
            f"Analyze sales call transcript (AI Agent 'Alex', Prospect 'User'). Niche: '{self.target_niche}'.\n"
            f"Transcript:\n---\n{history_str}\n---\n"
            f"Provide JSON summary: `overall_call_sentiment` (Positive/Neutral/Negative/Mixed), `prospect_engagement_level` (High/Medium/Low), "
            f"`key_topics_discussed` (list), `identified_prospect_pain_points` (list, specific), `objections_raised_by_prospect` (list), "
            f"`agent_objection_handling_effectiveness` (Effective/Partially/Ineffective/NA), `key_value_propositions_resonated` (list), "
            f"`agreed_next_steps` (string), `final_call_outcome_category` (Meeting_Booked/Strong_Interest_Followup/Mild_Interest_Nurture/Not_Interested_Polite/Not_Interested_Abrupt/Disqualified/Error/Voicemail), "
            f"`agent_performance_strengths` (list), `agent_areas_for_improvement` (list), `opportunities_for_ai_meta_awareness_use` (list)."
            f"Respond ONLY with valid JSON."
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            response_str = await self.llm_client.generate_response(messages=messages, purpose="analysis", temperature=0.1, max_tokens=1200, use_cache=False)
            if response_str and '{' in response_str and '}' in response_str:
                summary_obj = json.loads(response_str.strip())
                logger.info(f"[{self.agent_id}] LLM Call summary generated. Outcome: {summary_obj.get('final_call_outcome_category')}")
                return summary_obj
            else: return {"summary_status": "llm_no_response", "raw_response": response_str}
        except Exception as e: logger.error(f"[{self.agent_id}] Error generating LLM call summary: {e}. Raw: '{getattr(e, 'response_str', response_str if 'response_str' in locals() else 'N/A')}'", exc_info=True); return {"summary_status": f"llm_exception_{type(e).__name__}"}

    async def _cleanup_call(self, final_status: str, error_message: Optional[str]):
        """Performs all cleanup actions at the end of a call."""
        if self.call_sid is None: logger.warning(f"[{self.agent_id}] Cleanup called redundantly?"); return
        current_call_sid = self.call_sid
        logger.info(f"[{self.agent_id}][{current_call_sid}] Cleaning up call. Final Status: {final_status}")

        self.is_call_active = False
        if not self.call_end_event.is_set(): self.call_end_event.set()
        await self._cancel_ongoing_tts()
        if self.call_lifecycle_task and not self.call_lifecycle_task.done():
            if asyncio.current_task() is not self.call_lifecycle_task: self.call_lifecycle_task.cancel(); await asyncio.sleep(0)

        if self.voice_handler.is_connected: await self.voice_handler.disconnect()
        await self.telephony_wrapper.end_call(current_call_sid)

        llm_summary = await self._generate_call_summary_with_llm()
        crm_notes = f"Call Ended. System Status: {final_status}."
        if error_message: crm_notes += f" Error: {error_message}."
        if llm_summary.get("summary_status", "").startswith("llm_"): crm_notes += f" LLM Summary Status: {llm_summary['summary_status']}."
        
        contact_id_for_log = self.initial_prospect_data.get("id")
        if not contact_id_for_log and self.crm_wrapper.supabase:
            contact = await self.crm_wrapper.get_contact_info(self.target_phone_number, "phone_number")
            if contact: contact_id_for_log = contact.get("id")

        call_duration = int(time.monotonic() - self._call_start_time_monotonic) if self._call_start_time_monotonic > 0 else None

        log_success = await self.crm_wrapper.log_call_outcome(
            call_sid=current_call_sid, contact_id=contact_id_for_log, agent_id=self.agent_id,
            status=llm_summary.get("final_call_outcome_category", final_status),
            notes=crm_notes, conversation_history_json=self.conversation_history,
            call_duration_seconds=call_duration, llm_call_summary_json=llm_summary if "summary_status" not in llm_summary else None,
            key_objections_tags=llm_summary.get("objections_raised_by_prospect"),
            prospect_engagement_signals_tags=llm_summary.get("key_value_propositions_resonated"),
            agent_perceived_call_outcome=llm_summary.get("final_call_outcome_category"),
            target_phone_number=self.target_phone_number
        )
        if not log_success: logger.error(f"[{self.agent_id}][{current_call_sid}] Failed to log call outcome to CRM.")
        
        if contact_id_for_log and llm_summary.get("final_call_outcome_category"):
            new_status = f"PostCall_{llm_summary['final_call_outcome_category']}"
            await self.crm_wrapper.update_contact_status_and_notes(contact_id_for_log, new_status, f"Call {current_call_sid} ended. LLM Outcome: {llm_summary['final_call_outcome_category']}")

        # Notify Orchestrator/Server
        if self.on_call_complete_callback and not error_message:
            try: await self.on_call_complete_callback(self.agent_id, final_status, self.conversation_history, llm_summary)
            except Exception as cb_e: logger.error(f"Error in on_call_complete_callback: {cb_e}")
        elif self.on_call_error_callback and error_message:
            try: await self.on_call_error_callback(self.agent_id, error_message)
            except Exception as cb_e: logger.error(f"Error in on_call_error_callback: {cb_e}")

        # Final state reset
        self._final_status_reason = "Cleanup Complete"
        self.call_sid = None; self.stream_sid = None; self.current_tts_task = None
        self.call_lifecycle_task = None; self.conversation_history = []
        self._pending_action_verification = None
        logger.info(f"[{self.agent_id}][{current_call_sid}] Cleanup complete.")

    async def handle_incoming_audio(self, audio_chunk: bytes):
        """Passes audio chunks to the voice handler."""
        if self.is_call_active: await self.voice_handler.send_audio_chunk(audio_chunk)

    async def stop_call(self, reason: str = "External Stop Request"):
        """Allows external request to gracefully end the call."""
        logger.info(f"[{self.agent_id}][{self.call_sid}] Received external stop request. Reason: {reason}")
        self._signal_call_end(reason, is_error=False)
        if self.call_lifecycle_task and self.call_lifecycle_task.done() and self.is_call_active:
             logger.warning(f"[{self.agent_id}] stop_call invoked, but lifecycle task done and call active? Forcing cleanup.")
             asyncio.create_task(self._cleanup_call(reason, None))

if __name__ == "__main__":
    print("SalesAgent (Level 45+ AI Sales Virtuoso - FINAL) defined. Use via server orchestrator.")
#!/usr/bin/env python3
"""
Main orchestrator for Boutique AI Autonomous Sales System.
Instantiates and runs AcquisitionAgent and SalesAgent in an asyncio loop.
"""
import asyncio
import logging

import config
from core.agents.resource_manager import ResourceManager
from core.services.llm_client import LLMClient
from core.services.fingerprint_generator import FingerprintGenerator
from core.communication.voice_handler import VoiceHandler
from core.services.telephony_wrapper import TelephonyWrapper
from core.services.crm_wrapper import CRMWrapper
from core.services.data_wrapper import DataWrapper
from core.agents.acquisition_agent import AcquisitionAgent
from core.agents.sales_agent import SalesAgent

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # 1. Instantiate shared dependencies
    llm = LLMClient(cache_size=config.LLM_CACHE_SIZE)
    rm = ResourceManager(llm_client=llm)
    fg = FingerprintGenerator(llm)
    # VoiceHandler requires callbacks; SalesAgent will override them
    vh = VoiceHandler(transcript_callback=lambda t: None,
                      error_callback=lambda e: None)
    tel = TelephonyWrapper()
    crm = CRMWrapper()
    dw = DataWrapper(api_key=config.CLAY_API_KEY,
                     base_url=config.CLAY_API_BASE_URL)

    # 2. Create agents
    acq_agent = AcquisitionAgent(
        agent_id="AcqAgent1", resource_manager=rm,
        data_wrapper=dw, llm_client=llm, crm_wrapper=crm
    )
    sales_agent = SalesAgent(
        agent_id="SalesAgent1",
        target_phone_number=config.TWILIO_PHONE_NUMBER,
        voice_handler=vh,
        llm_client=llm,
        telephony_wrapper=tel,
        crm_wrapper=crm,
        # data_wrapper=dw, # optional
        on_call_complete_callback=lambda aid, sid, history: logger.info(f"Call complete: {sid}"),
        on_call_error_callback=lambda aid, err: logger.error(f"Call error: {err}"),
        send_audio_callback=lambda sid, chunk: None,
        send_mark_callback=lambda sid, mark: None,
    )

    # 3. Start both agents
    await acq_agent.start()
    await sales_agent.start()

    # 4. Monitor health and handle shutdown
    try:
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal. Stopping agents...")
        await acq_agent.stop()
        await sales_agent.stop()

if __name__ == "__main__":
    asyncio.run(main())

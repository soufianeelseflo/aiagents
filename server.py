#!/usr/bin/env python3
"""
Boutique AI Phone Server - FastAPI App
Exposes TwiML webhook (/call_webhook) and WebSocket (/media) endpoints for Twilio Media Streams.
"""
import base64
import logging

import config
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse

from core.services.llm_client import LLMClient
from core.services.fingerprint_generator import FingerprintGenerator
from core.communication.voice_handler import VoiceHandler
from core.services.telephony_wrapper import TelephonyWrapper
from core.services.crm_wrapper import CRMWrapper
from core.agents.sales_agent import SalesAgent

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

app = FastAPI()

# Shared dependencies
tel = TelephonyWrapper()
crm = CRMWrapper()
llm = LLMClient(cache_size=config.LLM_CACHE_SIZE)
fg = FingerprintGenerator(llm)

# Active agents map
call_agents: dict[str, SalesAgent] = {}

@app.post("/call_webhook")
async def call_webhook(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid")
    if not call_sid:
        logger.error("Missing CallSid in webhook request.")
        return Response(status_code=400)
    # Construct WebSocket URL
    ws_url = config.BASE_WEBHOOK_URL.replace("https://", "wss://") + f"/media?callSid={call_sid}"
    resp = VoiceResponse()
    resp.start().stream(url=ws_url)
    logger.info(f"CallWebhook: call {call_sid} streaming to {ws_url}")
    return Response(content=str(resp), media_type="application/xml")

@app.websocket("/media")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    params = websocket.query_params
    call_sid = params.get("callSid")
    if not call_sid:
        await websocket.close(code=1008)
        return
    agent_id = call_sid

    # Callbacks for SalesAgent
    async def send_audio_callback(sid: str, chunk: bytes):
        msg = {"event": "media", "media": {"payload": base64.b64encode(chunk).decode()}}
        await websocket.send_json(msg)
    async def send_mark_callback(sid: str, mark: str):
        msg = {"event": "mark", "mark": mark}
        await websocket.send_json(msg)
    async def transcript_callback(text: str):
        logger.info(f"[{call_sid}] Transcript: {text}")
    async def error_callback(err: str):
        logger.error(f"[{call_sid}] VoiceHandler error: {err}")

    # Instantiate per-call voice handler and SalesAgent
    vh = VoiceHandler(transcript_callback=transcript_callback, error_callback=error_callback)
    sales_agent = SalesAgent(
        agent_id=agent_id,
        target_phone_number="",
        voice_handler=vh,
        llm_client=llm,
        telephony_wrapper=tel,
        crm_wrapper=crm,
        on_call_complete_callback=lambda aid, sid, hist: logger.info(f"[{aid}] Call complete: {sid}"),
        on_call_error_callback=lambda aid, err: logger.error(f"[{aid}] Call error: {err}"),
        send_audio_callback=send_audio_callback,
        send_mark_callback=send_mark_callback,
    )
    call_agents[call_sid] = sales_agent

    try:
        while True:
            msg = await websocket.receive_json()
            event = msg.get("event")
            if event == "start":
                stream_sid = msg.get("streamSid")
                await sales_agent.start_sales_call(call_sid, stream_sid)
            elif event == "media":
                payload = msg.get("media", {}).get("payload")
                if payload:
                    chunk = base64.b64decode(payload)
                    await sales_agent.handle_incoming_audio(chunk)
            elif event == "stop":
                sales_agent.signal_call_ended_externally("Stop event")
                break
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for call {call_sid}")
    finally:
        call_agents.pop(call_sid, None)

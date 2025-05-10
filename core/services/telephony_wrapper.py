# core/services/telephony_wrapper.py

import logging
import asyncio
from typing import Optional, Dict, List, Any # CORRECTED

from twilio.rest import Client as TwilioClient
from twilio.base.exceptions import TwilioRestException

import config # Uses root config

logger = logging.getLogger(__name__)

class TelephonyWrapper:
    """
    Minimal wrapper for Twilio Voice API, focused on initiating and managing outbound calls.
    """
    def __init__(self):
        if not all([config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN, config.TWILIO_PHONE_NUMBER, config.BASE_WEBHOOK_URL]):
            logger.critical("CRITICAL: Twilio configuration incomplete (SID, Token, From Number, or Base URL missing).")
            raise ValueError("Missing required Twilio configuration.")
        try:
            self.client = TwilioClient(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
            logger.info("TelephonyWrapper: Twilio client initialized successfully.")
        except TwilioRestException as e:
            logger.critical(f"TelephonyWrapper: Failed to initialize Twilio client or verify credentials: {e}", exc_info=True)
            raise ConnectionError(f"Could not connect to Twilio: {e}") from e

    async def initiate_call(self, target_number: str, custom_parameters: Optional[Dict[str, str]] = None) -> Optional[str]:
        """
        Initiates an outbound call using Twilio.

        Args:
            target_number: The E.164 formatted phone number to call.
            custom_parameters: Optional dict of custom parameters to pass to TwiML app/webhook.
                               Example: {"prospect_crm_id": "12345"}

        Returns:
            The Twilio Call SID if successful, otherwise None.
        """
        if not config.BASE_WEBHOOK_URL:
             logger.error("Cannot initiate call: BASE_WEBHOOK_URL is not configured.")
             return None

        call_answered_webhook_url = f"{config.BASE_WEBHOOK_URL.rstrip('/')}/call_webhook"
        
        if custom_parameters:
            # Ensure values are simple strings for URL parameters
            safe_custom_parameters = {k: str(v) for k, v in custom_parameters.items()}
            param_string = "&".join([f"Custom_{k}={v}" for k,v in safe_custom_parameters.items()])
            call_answered_webhook_url += f"?{param_string}"

        logger.info(f"Initiating outbound call from {config.TWILIO_PHONE_NUMBER} to {target_number}...")
        logger.debug(f"Using call answered webhook: {call_answered_webhook_url}")

        try:
            call = await asyncio.to_thread( 
                self.client.calls.create,
                to=target_number,
                from_=config.TWILIO_PHONE_NUMBER,
                url=call_answered_webhook_url,
                method="POST",
                machine_detection="Enable", 
                machine_detection_timeout=8,
                # status_callback=f"{config.BASE_WEBHOOK_URL.rstrip('/')}/call_status_webhook", # For detailed call events
                # status_callback_event=["initiated", "ringing", "answered", "completed"],
                # status_callback_method="POST"
            )
            logger.info(f"Twilio call initiated successfully. Call SID: {call.sid}")
            return call.sid
        except TwilioRestException as e:
            logger.error(f"Twilio API error initiating call to {target_number}: {e.code} - {e.msg}", exc_info=False)
            return None
        except Exception as e:
            logger.error(f"Unexpected error initiating Twilio call: {e}", exc_info=True)
            return None

    async def end_call(self, call_sid: str) -> bool:
        """Attempts to terminate an ongoing call."""
        if not call_sid: 
            logger.warning("End call requested but no Call SID provided.")
            return False
        logger.info(f"Requesting termination of call SID: {call_sid}")
        try:
            call = await asyncio.to_thread(self.client.calls(call_sid).update, status='completed')
            logger.info(f"Call {call_sid} termination request successful. Final status: {call.status}")
            return True
        except TwilioRestException as e:
            if e.status == 404: 
                 logger.warning(f"Call {call_sid} not found or already ended when trying to terminate.")
                 return True 
            logger.error(f"Twilio API error terminating call {call_sid}: {e.code} - {e.msg}", exc_info=False)
            return False
        except Exception as e:
            logger.error(f"Unexpected error terminating Twilio call {call_sid}: {e}", exc_info=True)
            return False
# core/services/telephony_wrapper.py

import logging
from typing import Optional
from twilio.rest import Client as TwilioClient
from twilio.base.exceptions import TwilioRestException

# Import configuration centrally
from config import (
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_PHONE_NUMBER, # The "From" number for outbound calls
    BASE_WEBHOOK_URL    # Base URL for callbacks (e.g., call answered webhook)
)

logger = logging.getLogger(__name__)

class TelephonyWrapper:
    """
    Minimal wrapper for Twilio Voice API interactions, focusing on initiating
    and potentially managing outbound calls. Uses the official Twilio SDK minimally.
    """

    def __init__(self):
        """
        Initializes the Twilio client using credentials from config.
        """
        if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, BASE_WEBHOOK_URL]):
            logger.error("Twilio credentials or Base Webhook URL not fully configured.")
            raise ValueError("Missing required Twilio configuration (SID, Token, From Number, Base URL).")

        try:
            self.client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            # Verify connection by fetching account info (optional but good practice)
            # self.client.api.accounts(TWILIO_ACCOUNT_SID).fetch()
            logger.info("Twilio client initialized successfully.")
            logger.debug(f"Twilio Account SID: {TWILIO_ACCOUNT_SID[:5]}...{TWILIO_ACCOUNT_SID[-4:]}")
            logger.debug(f"Twilio 'From' Number: {TWILIO_PHONE_NUMBER}")
            logger.debug(f"Base Webhook URL for Callbacks: {BASE_WEBHOOK_URL}")
        except TwilioRestException as e:
            logger.error(f"Failed to initialize Twilio client or verify credentials: {e}", exc_info=True)
            raise ConnectionError(f"Could not connect to Twilio: {e}") from e
        except Exception as e:
             logger.error(f"Unexpected error initializing Twilio client: {e}", exc_info=True)
             raise ConnectionError(f"Unexpected error initializing Twilio: {e}") from e


    async def initiate_call(self, target_number: str) -> Optional[str]:
        """
        Initiates an outbound call using Twilio.

        Args:
            target_number: The phone number to call (E.164 format).

        Returns:
            The Twilio Call SID if successful, otherwise None.
        """
        # Construct the webhook URL Twilio will call when the recipient answers.
        # This webhook should trigger the connection to our WebSocket for audio streaming.
        # Assumes a specific endpoint structure, adjust if needed.
        call_answered_webhook_url = f"{BASE_WEBHOOK_URL}/call_webhook" # Matches sales_agent example

        logger.info(f"Initiating outbound call from {TWILIO_PHONE_NUMBER} to {target_number}...")
        logger.debug(f"Using call answered webhook: {call_answered_webhook_url}")

        try:
            # Use the Twilio client to create the call resource
            call = self.client.calls.create(
                to=target_number,
                from_=TWILIO_PHONE_NUMBER,
                url=call_answered_webhook_url, # TwiML instruction URL when call connects
                method="POST", # Standard method for webhooks
                # Optional: Add status callbacks for more detailed tracking if needed later
                # status_callback=f"{BASE_WEBHOOK_URL}/status_callback",
                # status_callback_event=["initiated", "ringing", "answered", "completed", "failed"],
                # status_callback_method="POST",
                # Optional: Answering Machine Detection
                machine_detection="Enable", # Detect if machine answers
                machine_detection_timeout=10 # Seconds to wait for detection
            )
            logger.info(f"Twilio call initiated successfully. Call SID: {call.sid}")
            return call.sid
        except TwilioRestException as e:
            logger.error(f"Twilio API error initiating call to {target_number}: {e}", exc_info=False)
            logger.debug(f"Twilio error details: Code={e.code}, Status={e.status}, URI={e.uri}, Msg={e.msg}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error initiating Twilio call: {e}", exc_info=True)
            return None

    async def end_call(self, call_sid: str) -> bool:
        """
        Attempts to terminate an ongoing call using its SID.

        Args:
            call_sid: The SID of the call to terminate.

        Returns:
            True if the termination request was accepted by Twilio, False otherwise.
        """
        if not call_sid:
            logger.warning("End call requested but no Call SID provided.")
            return False

        logger.info(f"Requesting termination of call SID: {call_sid}")
        try:
            # Update the call status to 'completed' to end it
            call = self.client.calls(call_sid).update(status='completed')
            logger.info(f"Call {call_sid} termination request successful. Final status: {call.status}")
            return True
        except TwilioRestException as e:
            # Handle cases where the call might already be completed or doesn't exist
            if e.status == 404:
                 logger.warning(f"Call {call_sid} not found or already ended when trying to terminate.")
                 return False # Or True, depending on desired semantics (already ended is success?)
            logger.error(f"Twilio API error terminating call {call_sid}: {e}", exc_info=False)
            logger.debug(f"Twilio error details: Code={e.code}, Status={e.status}, URI={e.uri}, Msg={e.msg}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error terminating Twilio call {call_sid}: {e}", exc_info=True)
            return False

    # Note: Generating the TwiML for the webhook (/call_webhook) is handled
    # by the web server part receiving the callback (e.g., in sales_agent.py using FastAPI).
    # This wrapper focuses only on the outbound API actions.

# Example Usage (Conceptual - requires running event loop)
async def main():
    print("Testing TelephonyWrapper...")
    try:
        wrapper = TelephonyWrapper()
        target = "+15559998888" # Replace with a valid test number if actually running
        print(f"Attempting to initiate call to {target} (will likely fail without valid credentials/number)...")
        # In a real test, ensure BASE_WEBHOOK_URL is accessible (e.g., via ngrok)
        # call_sid = await wrapper.initiate_call(target)

        # --- Simulate getting a SID ---
        call_sid = "CAxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # Dummy SID
        print(f"Simulated call initiation. Call SID: {call_sid}")
        # --- End Simulation ---

        if call_sid:
            print(f"Call initiated (simulation). SID: {call_sid}")
            await asyncio.sleep(2) # Simulate call duration
            print(f"Attempting to end call {call_sid} (simulation)...")
            # success = await wrapper.end_call(call_sid)
            success = True # Simulate success
            print(f"Call end request successful (simulation): {success}")
        else:
            print("Failed to initiate call (simulation).")

    except ValueError as e: # Catch configuration errors
         print(f"Configuration Error: {e}")
    except ConnectionError as e:
         print(f"Connection Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during test: {e}")

if __name__ == "__main__":
    # asyncio.run(main()) # Uncomment to run test (requires valid config and async context)
    print("TelephonyWrapper structure defined. Run test manually in an async context.")


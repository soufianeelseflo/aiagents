# core/communication/voice_handler.py

import asyncio
import logging
import base64
from typing import Callable, Optional, AsyncGenerator, Coroutine, Any # Added Coroutine, Any
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    SpeakOptions,
    SpeakStreamSource,
    StreamSource # Keep this import if used elsewhere, maybe not needed now
)

# Import configuration centrally
from config import (
    DEEPGRAM_API_KEY,
    DEEPGRAM_STT_MODEL,
    DEEPGRAM_TTS_MODEL
)

logger = logging.getLogger(__name__)

class VoiceHandler:
    """
    Handles real-time bidirectional audio streaming with Deepgram for
    Speech-to-Text (STT) and Text-to-Speech (TTS) via WebSockets.
    Designed to integrate with a telephony media stream.
    """

    def __init__(self,
                 transcript_callback: Callable[[str], Coroutine[Any, Any, None]], # Async callback for final transcripts
                 error_callback: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None # Optional async error callback
                 ):
        """
        Initializes the VoiceHandler.

        Args:
            transcript_callback: An async function to call when a final transcript is ready.
                                 Receives the transcript string as an argument.
            error_callback: An optional async function to call on Deepgram errors.
                            Receives the error message string.
        """
        if not DEEPGRAM_API_KEY:
            logger.error("Deepgram API key not configured.")
            raise ValueError("DEEPGRAM_API_KEY is required.")

        # Store callbacks provided by the agent/orchestrator
        self.transcript_callback = transcript_callback
        self.error_callback = error_callback

        self.dg_client: DeepgramClient
        self.dg_connection: Optional[Any] = None # Stores the active Deepgram connection object
        self.is_connected: bool = False
        # Removed websocket_task as connection management is internal to dg_connection

        # Configure Deepgram client
        dg_config: DeepgramClientOptions = DeepgramClientOptions(verbose=logging.DEBUG) # Enable debug for connection issues
        self.dg_client = DeepgramClient(DEEPGRAM_API_KEY, dg_config)
        logger.info("VoiceHandler initialized with Deepgram client.")


    async def connect(self) -> bool:
        """
        Establishes the connection to Deepgram for live transcription and speech synthesis.

        Returns:
            True if connection initiation was successful, False otherwise.
            Actual connection status is confirmed by the _on_open callback.
        """
        if self.is_connected:
            logger.warning("Deepgram connection attempt ignored: Already connected.")
            return True

        logger.info("Attempting to establish Deepgram live connection...")
        try:
            # Get the connection object using the latest API version
            self.dg_connection = self.dg_client.listen.asynclive.v("1")

            # Register event handlers BEFORE starting the connection
            self._register_event_handlers()

            # Define Live Transcription Options for audio sent TO Deepgram
            live_options = LiveOptions(
                model=DEEPGRAM_STT_MODEL,
                language="en-US",
                encoding="mulaw",       # Standard for telephony (e.g., Twilio)
                sample_rate=8000,       # Standard for telephony
                channels=1,
                interim_results=False,  # Only final transcripts needed for agent response
                smart_format=True,
                endpointing=300,        # ms of silence to detect end of speech
                vad_events=True         # Enable VAD events
            )
            logger.debug(f"Deepgram LiveOptions configured: {live_options}")

            # Start the connection - this opens the WebSocket
            await self.dg_connection.start(live_options)

            logger.info("Deepgram connection process initiated successfully.")
            # is_connected flag will be set True by the _on_open handler
            return True

        except Exception as e:
            logger.error(f"Failed to initiate Deepgram connection: {e}", exc_info=True)
            self.dg_connection = None
            self.is_connected = False
            # Trigger error callback if connection fails during initiation
            await self._handle_error(f"Deepgram connection initiation failed: {e}")
            return False

    def _register_event_handlers(self):
        """ Registers all necessary Deepgram event handlers. """
        if self.dg_connection:
            self.dg_connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
            self.dg_connection.on(LiveTranscriptionEvents.Metadata, self._on_metadata)
            self.dg_connection.on(LiveTranscriptionEvents.SpeechStarted, self._on_speech_started)
            self.dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, self._on_utterance_end)
            self.dg_connection.on(LiveTranscriptionEvents.Error, self._on_error)
            self.dg_connection.on(LiveTranscriptionEvents.Open, self._on_open)
            self.dg_connection.on(LiveTranscriptionEvents.Close, self._on_close)
            logger.debug("Deepgram event handlers registered.")
        else:
             logger.error("Cannot register Deepgram handlers: connection object is None.")


    async def send_audio_chunk(self, audio_chunk: bytes):
        """ Sends a raw audio chunk (bytes) to Deepgram for transcription. """
        if not self.is_connected or not self.dg_connection:
            logger.warning("Cannot send audio chunk, Deepgram connection not active.")
            return

        try:
            await self.dg_connection.send(audio_chunk)
        except Exception as e:
            logger.error(f"Error sending audio chunk to Deepgram: {e}", exc_info=True)
            await self._handle_error(f"Failed to send audio: {e}")


    async def speak_text(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Uses Deepgram Speak Stream API to synthesize speech and yields audio chunks (mulaw/8k).

        Args:
            text: The text to synthesize.

        Yields:
            Raw mulaw audio chunks (bytes).
        """
        if not self.is_connected or not self.dg_connection:
            logger.error("Cannot speak text, Deepgram connection not active.")
            return # Stop the generator gracefully

        logger.info(f"Requesting TTS stream from Deepgram for text: '{text[:70]}...'")
        speak_options = SpeakOptions(
            model=DEEPGRAM_TTS_MODEL, # Use Aura model from config
            encoding="mulaw",         # Match telephony requirements
            sample_rate=8000,         # Match telephony requirements
            container="none"          # Request raw audio stream
        )
        source: SpeakStreamSource = {'text': text}

        try:
            async for chunk_result in self.dg_connection.speak_stream(source, speak_options):
                audio_chunk = chunk_result.get("stream")
                if audio_chunk:
                    yield audio_chunk
                elif chunk_result.get('type') == 'Error':
                     error_msg = chunk_result.get('description', 'Unknown TTS Error')
                     logger.error(f"Deepgram TTS stream error: {error_msg}")
                     await self._handle_error(f"TTS Error: {error_msg}")
                     break # Stop generation on error
            logger.info("Finished receiving TTS stream from Deepgram.")
        except Exception as e:
            logger.error(f"Error during Deepgram TTS streaming request: {e}", exc_info=True)
            await self._handle_error(f"TTS Streaming Failed: {e}")


    async def disconnect(self):
        """ Gracefully closes the Deepgram connection. """
        if not self.is_connected or not self.dg_connection:
            logger.info("Deepgram disconnect requested, but connection already inactive.")
            return

        logger.info("Requesting Deepgram connection closure...")
        try:
            await self.dg_connection.finish()
            logger.info("Deepgram finish signal sent.")
            # Actual state change happens in _on_close
        except Exception as e:
            logger.error(f"Error during Deepgram connection finish: {e}", exc_info=True)
            # Force state update if finish fails
            self.is_connected = False
            self.dg_connection = None
            # Trigger error callback? Maybe not, as connection might still close.

    # --- Deepgram Event Handlers ---

    async def _on_open(self, *args, **kwargs):
        """ Called when the Deepgram WebSocket connection is successfully opened. """
        self.is_connected = True
        logger.info("Deepgram connection successfully opened and active.")

    async def _on_transcript(self, *args, **kwargs):
        """ Handles incoming transcript messages from Deepgram. """
        # dg_connection = args[0] # First argument is usually the connection object
        result = args[1] if len(args) > 1 else kwargs.get('result') # Get the result object
        if not result:
             logger.warning("Received transcript event with no result data.")
             return
        try:
            # Check if 'channel' and 'alternatives' exist and are structured as expected
            if (hasattr(result, 'channel') and
                hasattr(result.channel, 'alternatives') and
                isinstance(result.channel.alternatives, list) and
                len(result.channel.alternatives) > 0 and
                hasattr(result.channel.alternatives[0], 'transcript')):

                transcript = result.channel.alternatives[0].transcript
                if transcript and result.is_final:
                    logger.info(f"Received FINAL transcript: '{transcript}'")
                    if self.transcript_callback: # Ensure callback exists
                        await self.transcript_callback(transcript)
                    else:
                        logger.warning("Transcript received but no callback registered.")
            else:
                 logger.warning("Received transcript event with unexpected structure.")
                 logger.debug(f"Unexpected transcript result structure: {result}")

        except Exception as e:
            logger.error(f"Error processing transcript message: {e}", exc_info=True)
            logger.debug(f"Problematic transcript result object: {result}")


    async def _on_metadata(self, *args, **kwargs):
        """ Handles metadata messages. """
        metadata = args[1] if len(args) > 1 else kwargs.get('metadata')
        logger.debug(f"Received Deepgram metadata: {metadata}")

    async def _on_speech_started(self, *args, **kwargs):
        """ Handles speech started VAD events. """
        logger.debug("Deepgram detected speech started.")

    async def _on_utterance_end(self, *args, **kwargs):
        """ Handles utterance ended VAD events. """
        logger.debug("Deepgram detected utterance ended.")

    async def _on_error(self, *args, **kwargs):
        """ Handles error messages received from Deepgram. """
        error = args[1] if len(args) > 1 else kwargs.get('error')
        error_message = "Unknown Deepgram Error"
        if isinstance(error, dict):
            error_message = str(error.get('message', error))
        elif error:
             error_message = str(error)

        logger.error(f"Deepgram connection error reported: {error_message}")
        logger.debug(f"Full Deepgram error object: {error}")
        # Assume connection is lost on error
        self.is_connected = False
        self.dg_connection = None
        await self._handle_error(error_message)


    async def _on_close(self, *args, **kwargs):
        """ Called when the Deepgram WebSocket connection is closed. """
        logger.info("Deepgram connection closed.")
        self.is_connected = False
        self.dg_connection = None

    async def _handle_error(self, error_message: str):
         """ Utility to safely call the optional error callback. """
         if self.error_callback:
              try:
                   logger.debug(f"Calling registered error callback for: {error_message}")
                   await self.error_callback(error_message)
              except Exception as cb_err:
                   logger.error(f"Error executing error callback: {cb_err}", exc_info=True)

# (Conceptual Test Runner - Keep as is)
async def main():
    print("Testing VoiceHandler...")
    # ... (rest of test logic remains the same) ...
    async def dummy_transcript_handler(transcript: str): print(f"--- Transcript Received: {transcript} ---")
    async def dummy_error_handler(error: str): print(f"--- Error Received: {error} ---")
    handler = VoiceHandler(transcript_callback=dummy_transcript_handler, error_callback=dummy_error_handler)
    # ... (rest of test logic remains the same) ...
    print("Connecting...")
    connected = await handler.connect()
    if connected:
        print("Waiting for connection...")
        await asyncio.sleep(2) # Give time for _on_open
        if handler.is_connected:
            print("Connected. Simulating TTS...")
            async for chunk in handler.speak_text("Test"): print(f"Got TTS chunk: {len(chunk)} bytes")
            print("Disconnecting...")
            await handler.disconnect()
        else: print("Connection failed after initiation.")
    else: print("Initiation failed.")


if __name__ == "__main__":
    # import config # Ensure config is loaded
    # import asyncio
    # asyncio.run(main()) # Uncomment to run test (requires valid config and async context)
    print("VoiceHandler structure defined. Run test manually in an async context.")


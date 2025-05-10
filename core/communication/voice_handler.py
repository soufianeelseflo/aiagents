# core/communication/voice_handler.py

import asyncio
import logging
import base64
from typing import Callable, Optional, AsyncGenerator, Coroutine, Any # CORRECTED

# Import configuration centrally
from config import (
    DEEPGRAM_API_KEY,
    DEEPGRAM_STT_MODEL,
    DEEPGRAM_TTS_MODEL
)

# Attempt to import specific items from deepgram
try:
    import deepgram # Import the top-level package first
    # Now try to import specific components
    from deepgram import (
        DeepgramClient,
        DeepgramClientOptions,
        LiveTranscriptionEvents,
        LiveOptions,
        SpeakOptions
    )
    # SpeakStreamSource is part of the Prerecorded API in some versions,
    # but for live TTS, the source is typically a dict.
    # For speak_stream on the live connection, the source is {'text': '...'}
    # Let's check the SDK version to be sure.
    logger_vh = logging.getLogger(__name__) # Use a local logger instance
    logger_vh.info(f"Deepgram SDK Version (in voice_handler): {deepgram.__version__}")

    # SpeakStreamSource might not be needed if we construct the source dict directly
    # For listen.asynclive.v("1").speak_stream, the source is Dict[str, str]
    # If SpeakStreamSource is truly needed for another part of deepgram SDK not used here,
    # this import might still fail if the version is <3.1 or so.
    # For now, let's assume it's not strictly needed for the current speak_text implementation.
    # If it IS needed by a specific version for the live speak_stream, the error will point it out.
    try:
        from deepgram import SpeakStreamSource
    except ImportError:
        logger_vh.warning("SpeakStreamSource could not be imported from top-level deepgram. This might be okay if using dict for speak_stream source.")
        SpeakStreamSource = dict # Fallback to dict type for type hinting if not found

    DEEPGRAM_SDK_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).error(f"Failed to import from Deepgram SDK: {e}. VoiceHandler may not function correctly.")
    DEEPGRAM_SDK_AVAILABLE = False
    DeepgramClient = None # type: ignore
    DeepgramClientOptions = None # type: ignore
    LiveTranscriptionEvents = None # type: ignore
    LiveOptions = None # type: ignore
    SpeakOptions = None # type: ignore
    SpeakStreamSource = None # type: ignore


logger = logging.getLogger(__name__)

class VoiceHandler:
    """
    Handles real-time bidirectional audio streaming with Deepgram for
    Speech-to-Text (STT) and Text-to-Speech (TTS) via WebSockets.
    Designed to integrate with a telephony media stream.
    """

    def __init__(self,
                 transcript_callback: Callable[[str], Coroutine[Any, Any, None]],
                 error_callback: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None
                 ):
        if not DEEPGRAM_SDK_AVAILABLE:
            msg = "Deepgram SDK components not available. VoiceHandler cannot operate."
            logger.critical(msg)
            raise ImportError(msg)

        if not DEEPGRAM_API_KEY:
            logger.error("Deepgram API key not configured.")
            raise ValueError("DEEPGRAM_API_KEY is required.")

        self.transcript_callback = transcript_callback
        self.error_callback = error_callback

        self.dg_client: Optional[DeepgramClient] = None # Make it optional initially
        self.dg_connection: Optional[Any] = None 
        self.is_connected: bool = False
        
        try:
            # Configure Deepgram client options
            # verbose can be set to a log level like logging.DEBUG or an integer
            # Check DeepgramClientOptions documentation for exact accepted types for verbose
            verbose_level = logging.WARNING
            if logger.isEnabledFor(logging.DEBUG):
                verbose_level = logging.DEBUG
            
            # Example of setting options if needed, consult SDK docs for all available options
            # client_options_dict = {"api_url": "your-custom-deepgram-host"} if you have one
            client_options_dict = {} 

            dg_client_options = DeepgramClientOptions(verbose=verbose_level, **client_options_dict)
            self.dg_client = DeepgramClient(DEEPGRAM_API_KEY, dg_client_options)
            logger.info("VoiceHandler initialized with Deepgram client.")
        except Exception as e_dg_init:
            logger.error(f"Failed to initialize DeepgramClient: {e_dg_init}", exc_info=True)
            # Do not raise here, let connect() fail if dg_client is None

    async def connect(self) -> bool:
        if self.is_connected:
            logger.warning("Deepgram connection attempt ignored: Already connected.")
            return True
        
        if not self.dg_client: # Check if client was initialized
            logger.error("Deepgram client not initialized. Cannot connect.")
            await self._handle_error("Deepgram client not initialized.")
            return False

        logger.info("Attempting to establish Deepgram live connection...")
        try:
            # For listen.live.v("1") / listen.asynclive.v("1")
            self.dg_connection = self.dg_client.listen.asynclive.v("1")
            self._register_event_handlers()

            live_options = LiveOptions(
                model=DEEPGRAM_STT_MODEL,
                language="en-US",
                encoding="mulaw",
                sample_rate=8000,
                channels=1,
                interim_results=False,
                smart_format=True,
                endpointing=300, # milliseconds
                vad_events=True
            )
            logger.debug(f"Deepgram LiveOptions configured: {live_options}")
            await self.dg_connection.start(live_options)
            # is_connected will be set by _on_open
            logger.info("Deepgram connection process initiated successfully.")
            return True

        except Exception as e:
            logger.error(f"Failed to initiate Deepgram connection: {e}", exc_info=True)
            self.dg_connection = None
            self.is_connected = False
            await self._handle_error(f"Deepgram connection initiation failed: {e}")
            return False

    def _register_event_handlers(self):
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
        if not self.is_connected or not self.dg_connection:
            # logger.warning("Cannot send audio chunk, Deepgram connection not active.") # Can be too noisy
            return
        try:
            await self.dg_connection.send(audio_chunk)
        except Exception as e:
            logger.error(f"Error sending audio chunk to Deepgram: {e}", exc_info=True)
            await self._handle_error(f"Failed to send audio: {e}")

    async def speak_text(self, text: str) -> AsyncGenerator[bytes, None]:
        if not self.is_connected or not self.dg_connection:
            logger.error("Cannot speak text, Deepgram connection not active.")
            return 
        
        if not self.dg_client: # Should have been caught earlier
            logger.error("Cannot speak text, Deepgram client not initialized.")
            return

        logger.info(f"Requesting TTS stream from Deepgram for text: '{text[:70]}...'")
        
        # For SDK v3.x, speak_stream is part of the live connection object (self.dg_connection)
        # The `source` is a dictionary.
        source_data: Dict[str, str] = {'text': text}
        
        speak_options = SpeakOptions(
            model=DEEPGRAM_TTS_MODEL,
            encoding="mulaw",
            sample_rate=8000,
            container="none" 
        )

        try:
            # Use the speak_stream method on the live connection object
            async for chunk_result in self.dg_connection.speak_stream(source_data, speak_options):
                audio_chunk = chunk_result.get("stream")
                if audio_chunk:
                    yield audio_chunk
                elif chunk_result.get('type') == 'Error': # Check for explicit error type in stream
                     error_msg = chunk_result.get('description', 'Unknown TTS Error')
                     logger.error(f"Deepgram TTS stream error: {error_msg}")
                     await self._handle_error(f"TTS Error: {error_msg}")
                     break 
            logger.info("Finished receiving TTS stream from Deepgram.")
        except AttributeError as ae:
            if "speak_stream" in str(ae) and "AsyncLiveTranscriptionEvents" in str(ae): # More specific check
                logger.error("Deepgram SDK structure mismatch: 'speak_stream' not found on the live connection object. This indicates an issue with the SDK version or usage for live TTS.", exc_info=True)
                await self._handle_error("TTS failed: SDK structure mismatch for speak_stream.")
            else:
                logger.error(f"AttributeError during Deepgram TTS streaming: {ae}", exc_info=True)
                await self._handle_error(f"TTS Streaming Failed (AttributeError): {ae}")
        except Exception as e:
            logger.error(f"Error during Deepgram TTS streaming request: {e}", exc_info=True)
            await self._handle_error(f"TTS Streaming Failed: {e}")


    async def disconnect(self):
        if not self.is_connected or not self.dg_connection:
            logger.info("Deepgram disconnect requested, but connection already inactive.")
            return

        logger.info("Requesting Deepgram connection closure...")
        try:
            await self.dg_connection.finish()
            logger.info("Deepgram finish signal sent.")
        except Exception as e:
            logger.error(f"Error during Deepgram connection finish: {e}", exc_info=True)
            self.is_connected = False # Force state update
            self.dg_connection = None

    async def _on_open(self, *args, **kwargs): # Keep *args, **kwargs for SDK compatibility
        self.is_connected = True
        logger.info("Deepgram connection successfully opened and active.")

    async def _on_transcript(self, *args, **kwargs):
        # The first argument is often the connection instance, the second is the result
        result = args[1] if len(args) > 1 and args[1] else kwargs.get('result')
        if not result:
             logger.warning("Received transcript event with no result data.")
             return
        try:
            if (hasattr(result, 'channel') and
                hasattr(result.channel, 'alternatives') and
                isinstance(result.channel.alternatives, list) and
                len(result.channel.alternatives) > 0 and
                hasattr(result.channel.alternatives[0], 'transcript')):

                transcript = result.channel.alternatives[0].transcript
                if transcript and result.is_final: # Check for is_final attribute
                    logger.info(f"Received FINAL transcript: '{transcript}'")
                    if self.transcript_callback:
                        await self.transcript_callback(transcript)
            else:
                 logger.warning(f"Received transcript event with unexpected structure: {result}")

        except Exception as e:
            logger.error(f"Error processing transcript message: {e}", exc_info=True)
            logger.debug(f"Problematic transcript result object: {result}")

    async def _on_metadata(self, *args, **kwargs):
        metadata = args[1] if len(args) > 1 and args[1] else kwargs.get('metadata')
        logger.debug(f"Received Deepgram metadata: {metadata}")

    async def _on_speech_started(self, *args, **kwargs):
        logger.debug("Deepgram detected speech started.")

    async def _on_utterance_end(self, *args, **kwargs):
        logger.debug("Deepgram detected utterance ended.")

    async def _on_error(self, *args, **kwargs):
        error_obj = args[1] if len(args) > 1 and args[1] else kwargs.get('error')
        error_message = "Unknown Deepgram Error"
        if isinstance(error_obj, dict): # Error object might be a dict
            error_message = str(error_obj.get('message', error_obj.get('reason', str(error_obj))))
        elif error_obj: # Or it might be an exception object or other type
             error_message = str(error_obj)

        logger.error(f"Deepgram connection error reported: {error_message}")
        logger.debug(f"Full Deepgram error object: {error_obj}")
        self.is_connected = False
        self.dg_connection = None
        await self._handle_error(error_message)

    async def _on_close(self, *args, **kwargs):
        logger.info("Deepgram connection closed.")
        self.is_connected = False
        self.dg_connection = None

    async def _handle_error(self, error_message: str):
         if self.error_callback:
              try:
                   logger.debug(f"Calling registered error callback for: {error_message}")
                   await self.error_callback(error_message)
              except Exception as cb_err:
                   logger.error(f"Error executing error callback: {cb_err}", exc_info=True)

if __name__ == "__main__":
    # This is for basic testing if run directly.
    # Ensure config.py is loadable and DEEPGRAM_API_KEY is set in .env
    # Example: python -m core.communication.voice_handler
    
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s')
    
    async def dummy_transcript_cb(transcript: str):
        print(f"TEST CB - Transcript: {transcript}")

    async def dummy_error_cb(error: str):
        print(f"TEST CB - Error: {error}")

    async def test_run():
        print("VoiceHandler Test Run...")
        if not DEEPGRAM_API_KEY:
            print("DEEPGRAM_API_KEY not set. Aborting test.")
            return
        
        handler = VoiceHandler(transcript_callback=dummy_transcript_cb, error_callback=dummy_error_cb)
        if await handler.connect():
            print("Connected to Deepgram. Waiting a few seconds for events...")
            # To test STT, you would need to send audio chunks here.
            # Example: await handler.send_audio_chunk(b'some_mulaw_audio_bytes')
            
            # Test TTS
            print("Attempting TTS for 'Hello from Boutique AI'")
            tts_chunks_received = 0
            async for chunk in handler.speak_text("Hello from Boutique AI, this is a test message."):
                print(f"Received TTS audio chunk, length: {len(chunk)}")
                tts_chunks_received +=1
            print(f"TTS finished. Received {tts_chunks_received} chunks.")

            await asyncio.sleep(5) # Keep connection open for a bit
            await handler.disconnect()
            print("Disconnected.")
        else:
            print("Failed to connect to Deepgram.")

    # asyncio.run(test_run()) # Uncomment to run test
    print("VoiceHandler defined. To test, uncomment asyncio.run(test_run()) and ensure .env is configured.")
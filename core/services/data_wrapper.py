# core/services/data_wrapper.py

import logging
import json
import asyncio
from typing import Optional, Dict, List, Any
import aiohttp # For asynchronous HTTP requests

# Assuming config.py is at the project root
# import config # Not strictly needed if webhook URLs are passed directly

logger = logging.getLogger(__name__)

class ClayWebhookError(Exception):
    """Custom exception for errors sending data to Clay webhooks."""
    def __init__(self, message, status_code=None, response_text=None, webhook_url_snippet=None):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text
        self.webhook_url_snippet = webhook_url_snippet

    def __str__(self):
        return f"{super().__str__()} (Status: {self.status_code}, URL Snippet: {self.webhook_url_snippet}, Response: {self.response_text[:100] if self.response_text else 'N/A'})"

class DataWrapper:
    """
    Wrapper for sending data TO Clay.com tables via their specific Webhook URLs.
    Based on Clay documentation indicating webhooks are the primary method for
    programmatic data input.
    """

    _http_session: Optional[aiohttp.ClientSession] = None # Class-level shared session

    def __init__(self):
        # API key from config.CLAY_API_KEY might be used if Clay webhooks
        # are configured to require an 'x-clay-webhook-auth' header.
        # self.clay_auth_token = config.CLAY_API_KEY # If using x-clay-webhook-auth
        logger.info("DataWrapper initialized for sending data via Webhooks to Clay.com.")

    @classmethod
    async def _get_session(cls) -> aiohttp.ClientSession:
        """Initializes or returns the shared aiohttp session."""
        if cls._http_session is None or cls._http_session.closed:
            # You can configure connector limits here if needed
            # connector = aiohttp.TCPConnector(limit_per_host=config.MAX_CONCURRENT_CLAY_REQUESTS or 10)
            # cls._http_session = aiohttp.ClientSession(connector=connector)
            cls._http_session = aiohttp.ClientSession()
            logger.info("Initialized shared aiohttp.ClientSession for DataWrapper.")
        return cls._http_session

    @classmethod
    async def close_session(cls):
        """Closes the shared aiohttp session. Call during application shutdown."""
        if cls._http_session and not cls._http_session.closed:
            await cls._http_session.close()
            cls._http_session = None
            logger.info("Closed shared aiohttp.ClientSession for DataWrapper.")

    async def send_data_to_clay_webhook(
        self,
        webhook_url: str,
        data_payload: Dict[str, Any],
        # Optional: if you configure 'x-clay-webhook-auth' in Clay for the webhook
        # clay_auth_token: Optional[str] = None,
        timeout_seconds: int = 30
    ) -> bool:
        """
        Sends a single data record (JSON payload) to a specific Clay table's webhook URL.

        Args:
            webhook_url: The unique webhook URL for the target Clay table.
                         (Found in Clay UI: Table -> Import -> Monitor Webhook)
            data_payload: The dictionary representing the data record to send.
                          Keys should match expected input fields for the Clay table.
            timeout_seconds: Request timeout in seconds.

        Returns:
            True if the request was accepted by Clay (typically 200 OK), False otherwise.
        Raises:
            ClayWebhookError: If the request fails significantly or returns an unexpected status.
        """
        if not webhook_url or not webhook_url.startswith("https://hooks.clay.com/"):
            msg = "Invalid or missing Clay webhook_url."
            logger.error(msg)
            raise ValueError(msg)
        if not data_payload:
            msg = "data_payload cannot be empty for Clay webhook."
            logger.error(msg)
            raise ValueError(msg)

        session = await self._get_session()
        url_snippet = webhook_url[:30] + "..." + webhook_url[-10:]
        log_payload_summary = json.dumps(data_payload)[:150] + ("..." if len(json.dumps(data_payload)) > 150 else "")
        logger.info(f"Sending data to Clay Webhook: {url_snippet} Payload: {log_payload_summary}")

        headers = {"Content-Type": "application/json"}
        # if clay_auth_token: # If you set up x-clay-webhook-auth in Clay
        #     headers["x-clay-webhook-auth"] = clay_auth_token
        # elif config.CLAY_API_KEY: # Fallback to global key if webhook needs it
        #     logger.debug("Using global CLAY_API_KEY for x-clay-webhook-auth header (if webhook is configured to need it).")
        #     headers["x-clay-webhook-auth"] = config.CLAY_API_KEY


        try:
            async with session.post(
                webhook_url,
                json=data_payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout_seconds)
            ) as response:
                response_text = await response.text()

                # Clay webhooks typically return 200 OK on successful receipt.
                # Some might return 202 Accepted. VERIFY this.
                if response.status in [200, 201, 202, 204]:
                    logger.info(f"Data successfully sent to Clay webhook {url_snippet}. Status: {response.status}")
                    logger.debug(f"Clay webhook response body for {url_snippet}: {response_text[:500]}")
                    return True
                else:
                    err_msg = f"Error sending data to Clay webhook {url_snippet}. Status: {response.status}."
                    logger.error(f"{err_msg} Response: {response_text[:500]}")
                    raise ClayWebhookError(err_msg, status_code=response.status, response_text=response_text, webhook_url_snippet=url_snippet)

        except asyncio.TimeoutError:
            logger.error(f"Timeout sending to Clay webhook {url_snippet} after {timeout_seconds}s.")
            raise ClayWebhookError("Request timed out", status_code=408, webhook_url_snippet=url_snippet)
        except aiohttp.ClientError as e: # Covers ClientConnectorError, ServerDisconnectedError etc.
            logger.error(f"AIOHTTP client error sending to Clay webhook {url_snippet}: {e}", exc_info=True)
            raise ClayWebhookError(f"HTTP Client error: {e}", webhook_url_snippet=url_snippet) from e
        except Exception as e: # Catch-all for other unexpected errors
            logger.error(f"Unexpected error sending to Clay webhook {url_snippet}: {e}", exc_info=True)
            raise ClayWebhookError(f"Unexpected error: {e}", webhook_url_snippet=url_snippet) from e

    async def send_batch_to_clay_webhook(
        self,
        webhook_url: str,
        batch_records: List[Dict[str, Any]],
        concurrent_sends: int = 5,
        delay_between_sends_ms: int = 100
    ) -> tuple[int, int]:
        """
        Sends a batch of data records to a Clay webhook, one request per record, with concurrency.

        Args:
            webhook_url: The unique webhook URL for the target Clay table.
            batch_records: A list of dictionaries, each representing a record.
            concurrent_sends: Max number of concurrent POST requests.
            delay_between_sends_ms: Delay in milliseconds between starting each request.

        Returns:
            A tuple (successful_sends, failed_sends).
        """
        if not webhook_url: raise ValueError("webhook_url must be provided.")
        if not batch_records: return (0, 0)

        success_count = 0
        fail_count = 0
        tasks = []
        # Semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrent_sends)
        url_snippet = webhook_url[:30] + "..." + webhook_url[-10:]

        async def _send_one_record_with_semaphore(record_data: Dict[str, Any]):
            nonlocal success_count, fail_count
            async with semaphore:
                try:
                    # The send_data_to_clay_webhook now raises on error, so we catch it.
                    await self.send_data_to_clay_webhook(webhook_url, record_data)
                    success_count += 1
                except ClayWebhookError as e: # Catch specific errors from the send function
                    logger.warning(f"Failed to send record to {url_snippet} due to ClayWebhookError: {e}")
                    fail_count += 1
                except Exception as e: # Catch any other unexpected errors during the send_one_record task
                     logger.error(f"Unexpected error processing record in batch send to {url_snippet}: {record_data.get('id', 'N/A')}", exc_info=True)
                     fail_count += 1
                # Apply delay *after* request attempt, before releasing semaphore for next task effectively
                if delay_between_sends_ms > 0:
                    await asyncio.sleep(delay_between_sends_ms / 1000.0)

        logger.info(f"Sending batch of {len(batch_records)} records to Clay webhook {url_snippet}. "
                    f"Concurrency: {concurrent_sends}, Delay: {delay_between_sends_ms}ms")

        for i, record in enumerate(batch_records):
            # Add a unique identifier to the record if it doesn't have one, for better tracking
            if "_batch_item_id" not in record:
                record["_batch_item_id"] = f"batch_item_{i}"
            tasks.append(_send_one_record_with_semaphore(record))

        await asyncio.gather(*tasks, return_exceptions=False) # Errors are handled within _send_one_record

        logger.info(f"Batch send to Clay webhook {url_snippet} complete. "
                    f"Total: {len(batch_records)}, Success: {success_count}, Failed: {fail_count}")
        return success_count, fail_count

# --- Main for testing ---
async def main_test_dw_webhook_researched():
    print("Testing DataWrapper (Clay.com Webhook Implementation - Researched)...")

    # !!! IMPORTANT: Get this from your Clay table's settings !!!
    # In Clay: Open your table -> Click "Import" (or "+ Add Source" then "Monitor Webhook") -> Copy the Webhook URL.
    TEST_CLAY_TABLE_WEBHOOK_URL = os.getenv("TEST_CLAY_TABLE_WEBHOOK_URL_FROM_ENV") # Get from .env for testing
    # Example .env entry:
    # TEST_CLAY_TABLE_WEBHOOK_URL_FROM_ENV="https://hooks.clay.com/v1/workflows/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx/sync"

    if not TEST_CLAY_TABLE_WEBHOOK_URL or \
       "YOUR_UNIQUE_WEBHOOK_ID" in TEST_CLAY_TABLE_WEBHOOK_URL or \
       "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" in TEST_CLAY_TABLE_WEBHOOK_URL: # Check for placeholder
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! CRITICAL: Set TEST_CLAY_TABLE_WEBHOOK_URL_FROM_ENV in your .env file  !!!")
        print("!!! with YOUR REAL Clay table webhook URL before running this test.       !!!")
        print("!!! Find it in Clay: Table -> Import -> Monitor Webhook.                  !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        return

    wrapper = DataWrapper()

    print(f"\n--- Sending Single Record to Webhook: {TEST_CLAY_TABLE_WEBHOOK_URL[:50]}... ---")
    # Payload keys MUST match the input fields expected by your Clay table's workflow
    # For example, if your Clay table starts with a "Find Company from Domain" enrichment:
    single_payload = {
        "domain": f"testdomain{int(time.time())}.com", # Clay will use this as input
        "company_name_guess": f"Test Corp {int(time.time())}", # Additional data
        "source_reference": "dw_single_test"
    }
    try:
        success = await wrapper.send_data_to_clay_webhook(TEST_CLAY_TABLE_WEBHOOK_URL, single_payload)
        print(f"Single send successful: {success}")
        if success: print("Check your Clay table. The workflow should have run for this new row.")
    except ClayWebhookError as e:
        print(f"Single send failed: {e}")
    except Exception as e:
        print(f"Unexpected error during single send test: {e}", exc_info=True)

    print(f"\n--- Sending Batch Records to Webhook: {TEST_CLAY_TABLE_WEBHOOK_URL[:50]}... ---")
    batch_payload = []
    for i in range(3): # Create 3 sample records
        batch_payload.append({
            "domain": f"batchdomain{int(time.time()) + i}.com",
            "company_name_guess": f"Batch Corp {chr(65+i)}", # A, B, C
            "source_reference": f"dw_batch_test_{i}"
        })
    try:
        success_count, fail_count = await wrapper.send_batch_to_clay_webhook(
            TEST_CLAY_TABLE_WEBHOOK_URL,
            batch_payload,
            concurrent_sends=2,
            delay_between_sends_ms=200
            )
        print(f"Batch send complete. Success: {success_count}, Failed: {fail_count}")
        if success_count > 0: print("Check your Clay table for the new rows.")
    except Exception as e:
        print(f"Unexpected error during batch send test: {e}", exc_info=True)

    await DataWrapper.close_session() # Important to close the session when done

if __name__ == "__main__":
    # To run this test:
    # 1. Ensure async environment.
    # 2. `config.py` accessible.
    # 3. Create a table in Clay.com, configure its workflow (e.g., accept 'domain', 'company_name_guess').
    # 4. Go to the table settings in Clay (Import -> Monitor Webhook) and get its unique Webhook URL.
    # 5. Add this URL to your .env file as TEST_CLAY_TABLE_WEBHOOK_URL_FROM_ENV="your_clay_webhook_url"
    # 6. Install aiohttp: pip install aiohttp
    # Example:
    # import asyncio
    # load_dotenv() # Make sure .env is loaded if running this file directly for tests
    # asyncio.run(main_test_dw_webhook_researched())
    print("DataWrapper (Clay.com Webhook - Researched) defined. "
          "YOU MUST SET UP a Clay table, get its webhook URL, and put it in your .env "
          "as TEST_CLAY_TABLE_WEBHOOK_URL_FROM_ENV to run the test. Run test manually.")
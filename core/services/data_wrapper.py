# core/services/data_wrapper.py

import logging
import json
import time # Keep for potential future use (e.g., rate limiting delays)
from typing import Optional, Dict, List, Any
import requests # Using requests for direct API calls

# Import configuration centrally
from config import CLAY_API_BASE_URL # Using configured base URL

logger = logging.getLogger(__name__)

class DataWrapper:
    """
    Wrapper for interacting with Clay.com's API via direct HTTP requests,
    focusing on table-specific operations like lookups and writes, using an API key.
    Requires user to have tables set up in their Clay account and to verify/adjust
    the specific API endpoints and payload structures used.
    """

    def __init__(self):
        """
        Initializes the data wrapper. API key is set dynamically before use.
        """
        self.session = requests.Session() # Use a session for potential reuse
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        # Authentication header (API Key) will be set dynamically via set_api_key
        self.api_key: Optional[str] = None
        # Use base URL from config, ensure it's correctly set in .env
        self.base_url = CLAY_API_BASE_URL
        logger.info(f"DataWrapper initialized for Clay.com (Base URL: {self.base_url}). API key needs to be set.")

    def set_api_key(self, api_key: Optional[str]):
        """
        Sets the Clay.com API key found in user settings for subsequent requests.

        Args:
            api_key: The Clay.com API key.
        """
        if api_key:
            self.api_key = api_key
            # Assuming Clay uses a standard Bearer token scheme based on common practice.
            # ** VERIFY THIS AUTHENTICATION METHOD **
            auth_header = f"Bearer {self.api_key}"
            self.session.headers.update({"Authorization": auth_header})
            logger.info("Clay.com API key set for DataWrapper.")
            logger.debug(f"Clay API Key (masked): {self.api_key[:4]}...{self.api_key[-4:]}")
        else:
             self.api_key = None
             # Remove auth header if key is removed or invalid
             if "Authorization" in self.session.headers:
                  del self.session.headers["Authorization"]
             logger.warning("Clay.com API key removed or not provided to DataWrapper.")


    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None
        ) -> Optional[Dict[str, Any]]:
        """
        Internal helper to make HTTP requests to the Clay.com API.
        Handles actual network request, error checking, and JSON parsing.
        """
        if not self.api_key:
            logger.error("Cannot make Clay API request: API key not set.")
            return None

        # Ensure URL is formed correctly
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        # Prepare JSON data payload if provided
        json_data_payload = json.dumps(data) if data else None

        logger.debug(f"Making Clay API Request: {method.upper()} {url}")
        if params: logger.debug(f"  Params: {params}")
        if json_data_payload: logger.debug(f"  Data: {json_data_payload[:500]}...") # Log truncated data

        try:
            response = self.session.request(
                method=method.upper(),
                url=url,
                params=params,
                data=json_data_payload, # Send JSON string in data field
                timeout=30 # Allow reasonable timeout for data operations
            )
            # Raise an exception for bad status codes (4xx or 5xx)
            response.raise_for_status()

            # Handle successful responses
            if response.status_code == 204: # No Content
                 logger.info(f"Clay API Request successful ({response.status_code} No Content).")
                 return {} # Indicate success with no body

            # Attempt to parse JSON response for other success codes (e.g., 200 OK, 201 Created)
            response_json = response.json()
            logger.debug(f"Clay API Response ({response.status_code}): {str(response_json)[:500]}...")
            return response_json

        except requests.exceptions.HTTPError as e:
            # Log detailed HTTP errors
            logger.error(f"Clay API HTTP Error: {e.response.status_code} {e.response.reason} for {method.upper()} {url}")
            try:
                # Try to get more specific error info from response body
                error_details = e.response.json()
                logger.error(f"  Error details from API: {error_details}")
            except json.JSONDecodeError:
                # Log raw text if response is not JSON
                logger.error(f"  Error body (non-JSON): {e.response.text[:500]}...")
            return None # Indicate failure
        except requests.exceptions.RequestException as e:
            # Handle other request errors (connection, timeout, etc.)
            logger.error(f"Clay API Request Failed ({method.upper()} {url}): {e}", exc_info=True)
            return None
        except json.JSONDecodeError as e:
             # Handle errors parsing a successful (e.g., 200 OK) response
             logger.error(f"Failed to decode apparently successful Clay API JSON response ({response.status_code}) from {method.upper()} {url}: {e}")
             logger.debug(f"Raw Response Text: {response.text[:500]}...")
             return None # Indicate failure to parse


    async def lookup_row_in_table(
        self,
        table_id: str,
        lookup_column_header: str,
        lookup_value: Any, # Value can be string, number etc.
        select_columns: Optional[List[str]] = None
        ) -> Optional[Dict[str, Any]]:
        """
        Looks up a SINGLE row in a specific Clay table based on a column value.
        **NOTE:** The specific endpoint and payload structure are HYPOTHETICAL
        and MUST be verified against actual Clay.com API behavior.

        Args:
            table_id: The ID of the Clay table (e.g., 't_xxxxxxxx').
            lookup_column_header: The exact header/name of the column to match against.
            lookup_value: The value to search for in the lookup column.
            select_columns: Optional list of column headers to return. If None, returns all accessible.

        Returns:
            A dictionary representing the found row's data (keys are column headers),
            or None if not found or an error occurs.
        """
        if not table_id or not lookup_column_header:
             logger.error("lookup_row_in_table requires table_id and lookup_column_header.")
             return None

        logger.info(f"Looking up row in Clay table '{table_id}' where '{lookup_column_header}' matches value.")
        if not self.api_key: return None # Ensure API key is set

        # --- ### START HYPOTHETICAL CLAY API IMPLEMENTATION ### ---
        # This endpoint structure is a guess based on common patterns and documented key capabilities.
        # It assumes a POST request for lookup flexibility. VERIFY THIS.
        endpoint = f"/tables/{table_id}/rows/lookup" # HYPOTHETICAL ENDPOINT
        payload = {
            "filter": {
                "column": lookup_column_header,
                "value": lookup_value # Send the value directly
            },
            "limit": 1 # We only want the first match for a single lookup
        }
        if select_columns:
            payload["select"] = select_columns

        logger.warning(f"lookup_row_in_table using HYPOTHETICAL endpoint '{endpoint}' and payload structure. VERIFY!")

        # Make the actual API call (asynchronously if needed, but using sync requests here)
        # Consider wrapping _make_request in asyncio.to_thread if calling from async code frequently
        response_data = self._make_request("POST", endpoint, data=payload)

        # Parse the response - This structure is also HYPOTHETICAL
        if response_data and isinstance(response_data.get("results"), list):
            results = response_data["results"]
            if len(results) > 0:
                found_row = results[0] # Assume first result is the one we want
                logger.info(f"Found row in table '{table_id}'.")
                # Ensure the result is a dictionary (representing the row data)
                return found_row if isinstance(found_row, dict) else None
            else:
                logger.info(f"No row found in table '{table_id}' matching criteria.")
                return None # Explicitly return None for not found
        elif response_data is not None: # Request succeeded but unexpected format or no results field
             logger.warning(f"Clay lookup request succeeded but response format unexpected or empty: {response_data}")
             return None
        else: # Request failed entirely
             logger.error(f"Failed to lookup row in table '{table_id}' due to API request error.")
             return None
        # --- ### END HYPOTHETICAL CLAY API IMPLEMENTATION ### ---


    async def write_row_to_table(self, table_id: str, row_data: Dict[str, Any]) -> bool:
        """
        Writes (appends) a new row to a specific Clay table.
        **NOTE:** The specific endpoint and payload structure are HYPOTHETICAL
        and MUST be verified against actual Clay.com API behavior. This assumes append-only.
        Update logic might require a different endpoint or method (e.g., PATCH).

        Args:
            table_id: The ID of the Clay table (e.g., 't_xxxxxxxx').
            row_data: A dictionary where keys are exact column headers and values are cell values.

        Returns:
            True if the write operation was likely successful (e.g., 200 OK or 201 Created), False otherwise.
        """
        if not table_id or not row_data:
             logger.error("write_row_to_table requires table_id and non-empty row_data.")
             return False

        logger.info(f"Writing row to Clay table '{table_id}'...")
        if not self.api_key: return False

        # --- ### START HYPOTHETICAL CLAY API IMPLEMENTATION ### ---
        # This endpoint structure is a guess. Assumes POST to add a new row.
        endpoint = f"/tables/{table_id}/rows" # HYPOTHETICAL ENDPOINT
        # Payload structure might be simple like {"columns": row_data} or just row_data
        payload = row_data # Simplest assumption - VERIFY THIS
        logger.warning(f"write_row_to_table using HYPOTHETICAL endpoint '{endpoint}' and payload structure. VERIFY!")

        # Make the actual API call
        response_data = self._make_request("POST", endpoint, data=payload)

        # Check response - Success might be 200 OK, 201 Created, or 204 No Content
        # We consider any non-None response (meaning no request/HTTP error) as potential success here.
        # A more robust check would inspect response_data for specific success indicators if available.
        if response_data is not None:
            logger.info(f"Write request to table '{table_id}' completed (Status Code implies success).")
            # Optionally check response_data for specific confirmation if API provides it
            # e.g., if response_data.get("status") == "success": return True
            return True
        else:
            logger.error(f"Failed to write row to table '{table_id}' due to API request error.")
            return False
        # --- ### END HYPOTHETICAL CLAY API IMPLEMENTATION ### ---


# (Conceptual Test Runner - Keep as is for structure demonstration)
async def main():
    print("Testing DataWrapper (Clay.com Functional - Table Interaction v3)...")
    # Requires CLAY_API_KEY in .env and knowledge of actual table/column IDs
    import config # Load config
    if not config.CLAY_API_KEY:
         print("Skipping test: CLAY_API_KEY environment variable not set.")
         return

    wrapper = DataWrapper()
    wrapper.set_api_key(config.CLAY_API_KEY)

    # --- Replace with ACTUAL Table ID and Column Names from YOUR Clay account ---
    TEST_TABLE_ID = "t_YOUR_TABLE_ID_HERE" # e.g., t_abc123xyz
    LOOKUP_COLUMN = "Company Domain"       # e.g., Email, LinkedIn URL, Company Domain
    LOOKUP_VALUE = "example.com"           # Value to search for
    WRITE_COLUMN_1 = "Lead Status"         # Column to write to
    WRITE_COLUMN_2 = "Timestamp Added"     # Another column to write to
    # --- ---

    print(f"\nLooking up row in table '{TEST_TABLE_ID}' where '{LOOKUP_COLUMN}'='{LOOKUP_VALUE}'")
    # Note: Use await if calling from async context, otherwise adapt _make_request or use sync wrapper
    # For simplicity in example, assuming sync call here, but use await in real async code
    # row = wrapper.lookup_row_in_table(TEST_TABLE_ID, LOOKUP_COLUMN, LOOKUP_VALUE) # Sync version needed for this test block if not async
    # --- Using await as main is async ---
    row = await wrapper.lookup_row_in_table(TEST_TABLE_ID, LOOKUP_COLUMN, LOOKUP_VALUE)
    if row is not None:
        print(f"Lookup result (requires real API & table): {row}")
    else:
        print("Lookup failed or not found (requires real API & table).")

    print(f"\nWriting row to table '{TEST_TABLE_ID}'")
    write_data = {
        LOOKUP_COLUMN: f"new_company_{int(time.time())}.com", # Ensure lookup column has value
        WRITE_COLUMN_1: "New Lead",
        WRITE_COLUMN_2: datetime.utcnow().isoformat()
    }
    # success = wrapper.write_row_to_table(TEST_TABLE_ID, write_data) # Sync version
    # --- Using await ---
    success = await wrapper.write_row_to_table(TEST_TABLE_ID, write_data)
    print(f"Write successful (requires real API & table): {success}")

if __name__ == "__main__":
    # import asyncio
    # asyncio.run(main()) # Uncomment to run test (requires valid config, async context, and configured Clay table)
    print("DataWrapper structure defined (Functional Table Interaction v3). Requires valid Clay API key and user-configured tables/verified endpoints for real use.")


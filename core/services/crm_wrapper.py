# core/services/crm_wrapper.py

import logging
import json
from typing import Optional, Dict, Any, List
import requests # Using requests for direct, simple HTTP calls

# Import configuration centrally - These MUST be set in .env or passed during init
# based on the target CRM for a specific client/campaign.
try:
    # Attempt to load defaults, but they likely need overriding
    from config import CRM_API_BASE_URL, CRM_API_KEY, CRM_TYPE
except ImportError:
    # Define fallbacks if config isn't fully set up yet
    CRM_API_BASE_URL = "https://api.needs-configuration.com"
    CRM_API_KEY = None
    CRM_TYPE = "Unknown"
    LOG_FILE = None # Needed for log dir path below

logger = logging.getLogger(__name__)


class CRMWrapper:
    """
    Minimal CRM wrapper using direct HTTP requests (via `requests` library)
    for essential contact lookup and call logging. Avoids heavy SDKs/ORMs.
    Requires specific endpoint/payload implementation within methods based on the target CRM's API documentation.
    """

    def __init__(self, crm_type: str = CRM_TYPE, api_base_url: str = CRM_API_BASE_URL, api_key: Optional[str] = CRM_API_KEY):
        """
        Initializes the CRM wrapper with configuration for a specific CRM.

        Args:
            crm_type: Identifier for the target CRM (e.g., "SalesforceAPI", "HubSpotAPI", "Custom"). Used for potential logic branching.
            api_base_url: The base URL for the target CRM's API.
            api_key: The API key or token for authentication. MUST be provided.
        """
        self.crm_type = crm_type
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.session = requests.Session() # Use a session for potential connection reuse

        # --- Authentication Header Setup (MUST BE CUSTOMIZED PER CRM) ---
        if not self.api_key:
             # Allow initialization without key, but log a strong warning. Calls will fail.
             logger.error(f"CRMWrapper initialized for '{self.crm_type}' WITHOUT API KEY. API calls WILL FAIL.")
        else:
            # This logic needs to be adapted based on the target CRM's auth method.
            # Defaulting to Bearer token, but this is often incorrect.
            auth_header_value = f"Bearer {self.api_key}"
            # Example for other types (modify based on CRM docs):
            # if "salesforce" in self.crm_type.lower():
            #     auth_header_value = f"Bearer {self.api_key}" # Assumes OAuth token
            # elif "hubspot" in self.crm_type.lower():
            #     auth_header_value = f"Bearer {self.api_key}" # Assumes private app token
            # elif "apikey_header" in self.crm_type.lower():
            #     self.session.headers.update({"X-Api-Key": self.api_key}) # Example custom header
            #     auth_header_value = None # Auth handled by specific header

            if auth_header_value:
                self.session.headers.update({"Authorization": auth_header_value})

            logger.info(f"CRMWrapper initialized for type '{self.crm_type}' at base URL '{self.api_base_url}'.")
            logger.debug("API Key loaded (ensure correct auth header logic is implemented).")
        # --- End Authentication Setup ---

        # Set standard content/accept headers
        self.session.headers.update({"Content-Type": "application/json", "Accept": "application/json"})


    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None
        ) -> Optional[Dict[str, Any]]:
        """
        Internal helper to make HTTP requests to the configured CRM API.
        Handles the actual network request, error checking, and JSON parsing.
        """
        if not self.api_key:
            logger.error(f"Cannot make CRM request ({method} {endpoint}): API key not set.")
            return None
        if not self.api_base_url or self.api_base_url == "https://api.needs-configuration.com":
             logger.error(f"Cannot make CRM request ({method} {endpoint}): CRM API Base URL not configured.")
             return None

        # Ensure URL is formed correctly, preventing double slashes
        url = f"{self.api_base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        # Prepare JSON data payload if provided
        json_data_payload = json.dumps(data) if data else None

        logger.debug(f"Making CRM Request: {method.upper()} {url}")
        if params: logger.debug(f"  Params: {params}")
        if json_data_payload: logger.debug(f"  Data: {json_data_payload[:500]}...") # Log truncated data

        try:
            response = self.session.request(
                method=method.upper(),
                url=url,
                params=params,
                data=json_data_payload,
                timeout=20 # Increased timeout slightly for CRM operations
            )
            # Raise an exception for bad status codes (4xx or 5xx)
            response.raise_for_status()

            # Handle successful responses
            if response.status_code == 204: # No Content
                 logger.info(f"CRM Request successful ({response.status_code} No Content).")
                 return {} # Indicate success with no body

            # Attempt to parse JSON response for other success codes (e.g., 200 OK, 201 Created)
            response_json = response.json()
            logger.debug(f"CRM Response ({response.status_code}): {str(response_json)[:500]}...")
            return response_json

        except requests.exceptions.HTTPError as e:
            logger.error(f"CRM HTTP Error: {e.response.status_code} {e.response.reason} for {method.upper()} {url}")
            try: logger.error(f"  Error details from API: {e.response.json()}")
            except json.JSONDecodeError: logger.error(f"  Error body (non-JSON): {e.response.text[:500]}...")
            return None # Indicate failure
        except requests.exceptions.RequestException as e:
            logger.error(f"CRM Request Failed ({method.upper()} {url}): {e}", exc_info=True)
            return None
        except json.JSONDecodeError as e:
             logger.error(f"Failed to decode apparently successful CRM API JSON response ({response.status_code}) from {method.upper()} {url}: {e}")
             logger.debug(f"Raw Response Text: {response.text[:500]}...")
             return None


    async def get_contact_info(self, phone_number: str) -> Optional[Dict[str, Any]]:
        """
        Searches for a contact/lead by phone number.
        **MUST BE IMPLEMENTED with logic specific to the target CRM's API.**

        Args:
            phone_number: The phone number to search for (E.164 format preferred).

        Returns:
            A dictionary containing relevant contact info (e.g., crm_id, name, company)
            or None if not found or an error occurs.
        """
        logger.info(f"Attempting to get contact info from CRM for phone: {phone_number}")
        if not self.api_key: return None

        # --- ### START CRM-SPECIFIC IMPLEMENTATION ### ---
        # This block MUST be replaced with logic for the actual target CRM.
        # 1. Determine the correct API endpoint for searching contacts by phone.
        # 2. Determine the correct HTTP method (GET, POST).
        # 3. Construct the necessary query parameters or request body payload
        #    according to the CRM's API documentation (e.g., filter syntax).
        # 4. Define which contact fields are needed (e.g., ID, name, company).
        # 5. Call self._make_request(method, endpoint, params=params, data=data).
        # 6. Parse the JSON response to extract the desired contact fields, ensuring a unique ID is captured.
        # 7. Handle cases where no contact is found (return None).

        logger.error(f"CRMWrapper.get_contact_info() is not implemented for CRM type '{self.crm_type}'. Cannot fetch contact.")
        # Example Structure (REMOVE OR REPLACE):
        # endpoint = "/api/vX/contacts/search" # Replace with actual endpoint
        # params = {"query": f"phone_number={phone_number}", "fields":"id,firstName,lastName,company"} # Replace with actual params/fields
        # response_data = self._make_request("GET", endpoint, params=params)
        # if response_data and response_data.get('results'): # Replace with actual parsing
        #     return response_data['results'][0] # Return first match
        # else:
        #     return None
        # --- ### END CRM-SPECIFIC IMPLEMENTATION ### ---

        return None # Return None because it's not implemented


    async def log_call_outcome(
        self,
        phone_number: str,
        status: str,
        notes: str,
        agent_id: str = "UNKNOWN_AGENT",
        call_sid: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
        ) -> bool:
        """
        Logs the outcome of a call (e.g., creates a Task, Activity, Note).
        **MUST BE IMPLEMENTED with logic specific to the target CRM's API.**

        Args:
            phone_number: Phone number to associate the log with (used to find contact ID).
            status: Short status description (e.g., "Meeting Booked", "Voicemail Left").
            notes: Detailed call summary or key outcome notes provided by the agent/system.
            agent_id: The ID of the agent handling the call.
            call_sid: Optional telephony identifier (e.g., from Twilio).
            conversation_history: Optional full transcript history (may be truncated for logging).

        Returns:
            True if logging was likely successful (API call didn't return error), False otherwise.
        """
        logger.info(f"Attempting to log call outcome to CRM for {phone_number}. Status: {status}")
        if not self.api_key: return False

        # 1. Find CRM contact ID (using the method above, which needs implementation)
        # It's crucial this step works for proper association.
        contact_info = await self.get_contact_info(phone_number)
        contact_id = contact_info.get("id") if contact_info else None # Assumes 'id' is the key for the CRM record ID

        if not contact_id:
             logger.warning(f"Could not find contact ID for {phone_number} via get_contact_info. Attempting to log call without association or skipping.")
             # Depending on CRM, you might still log a general task, or this might be a failure.
             # For now, we'll indicate failure if contact isn't found, as association is usually key.
             # return False # Option 1: Fail if no contact ID
             pass # Option 2: Proceed and try to log without association (depends on CRM API)


        # --- ### START CRM-SPECIFIC IMPLEMENTATION ### ---
        # This block MUST be replaced with logic for the actual target CRM.
        # 1. Determine the correct API endpoint for creating activities/tasks/notes.
        # 2. Determine the correct HTTP method (usually POST).
        # 3. Construct the request body payload according to the CRM's API doc:
        #    - Map 'status' to the CRM's disposition field.
        #    - Include subject (e.g., f"Boutique AI Call: {status}").
        #    - Include description/body (containing agent_id, notes, call_sid, truncated transcript).
        #    - Include association to the contact_id (if found and required).
        #    - Set relevant fields like activity type ('Call'), date, task status ('Completed').
        # 4. Call self._make_request("POST", endpoint, data=payload).
        # 5. Check the response to confirm successful creation (e.g., status 201 or specific success field).

        logger.error(f"CRMWrapper.log_call_outcome() is not implemented for CRM type '{self.crm_type}'. Cannot log outcome.")
        # Example Structure (REMOVE OR REPLACE):
        # endpoint = "/api/vX/activities" # Replace with actual endpoint
        # log_body = f"Agent: {agent_id}\nStatus: {status}\nSID: {call_sid}\nNotes: {notes}\n\nSummary:\n"
        # if conversation_history: log_body += " | ".join([f"{m['role'][:1]}:{json.dumps(m['text'][:80])}" for m in conversation_history[-4:]])
        # payload = {
        #     "type": "Call",
        #     "subject": f"Boutique AI Call: {status}",
        #     "description": log_body[:10000], # Truncate
        #     "contactId": contact_id, # Association
        #     "status": "Completed" # Activity status
        # }
        # response_data = self._make_request("POST", endpoint, data=payload)
        # return response_data is not None # Basic check
        # --- ### END CRM-SPECIFIC IMPLEMENTATION ### ---

        return False # Return False because it's not implemented


# (Conceptual Test Runner - Keep as is for structure demonstration)
async def main():
    print("Testing CRMWrapper (Abstracted Direct API - Final)...")
    # This test requires CRM_API_BASE_URL and CRM_API_KEY to be set in config/env
    # AND the methods get_contact_info/log_call_outcome to be implemented for that CRM.
    try:
        import config # Load config
        # Instantiate with potentially configured values
        wrapper = CRMWrapper(
            crm_type=config.CRM_TYPE,
            api_base_url=config.CRM_API_BASE_URL,
            api_key=config.CRM_API_KEY
        )
    except ImportError:
        print("Warning: config.py not found or incomplete. Using dummy wrapper.")
        wrapper = CRMWrapper(api_key=None) # Will show errors
    except ValueError as e:
        print(f"Configuration Error: {e}")
        return

    test_phone = "+15551112222"
    print(f"\nAttempting get_contact_info for: {test_phone} (Requires Implementation)")
    contact = await wrapper.get_contact_info(test_phone)
    if contact: print(f"Found contact: {contact}")
    else: print("Contact not found or method not implemented.")

    print(f"\nAttempting log_call_outcome for: {test_phone} (Requires Implementation)")
    success = await wrapper.log_call_outcome(
        phone_number=test_phone,
        status="Meeting Booked",
        notes="Test log via abstract wrapper.",
        agent_id="TestAgentAbstract",
        call_sid="CAabstract"
    )
    print(f"CRM log successful: {success} (Requires Implementation)")

if __name__ == "__main__":
    # import asyncio
    # asyncio.run(main()) # Uncomment to run test (requires async context and implemented methods)
    print("CRMWrapper structure defined (Abstracted Direct API - Final). Requires CRM-specific implementation in methods for real use.")


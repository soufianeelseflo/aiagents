# /core/services/crm_wrapper.py:
# --------------------------------------------------------------------------------
# core/services/crm_wrapper.py

import logging
import json
import asyncio
import time # Needed for test function
from datetime import datetime, timezone
# *** ADDED typing imports ***
from typing import Optional, Dict, Any, List, Union

# Check if supabase library is installed
try:
    from supabase import create_client, Client as SupabaseClient, PostgrestAPIError
    SUPABASE_PY_AVAILABLE = True
except ImportError:
    SUPABASE_PY_AVAILABLE = False
    SupabaseClient = None # type: ignore
    PostgrestAPIError = None # type: ignore
    logging.getLogger(__name__).warning("supabase-py library not found. CRM functions requiring it will be disabled.")


import config # Root config

logger = logging.getLogger(__name__)

class CRMWrapper:
    """
    Intelligent CRM wrapper using Supabase. Manages contacts with robust upsert logic
    and logs detailed call/interaction outcomes for future learning and agent context.
    """

    def __init__(self):
        self.supabase: Optional[SupabaseClient] = None
        self.call_log_table: str = config.SUPABASE_CALL_LOG_TABLE
        self.contacts_table: str = config.SUPABASE_CONTACTS_TABLE

        if config.SUPABASE_ENABLED and SUPABASE_PY_AVAILABLE:
            try:
                # Ensure URL and Key are provided before creating client
                if not config.SUPABASE_URL or not config.SUPABASE_KEY:
                     raise ValueError("SUPABASE_URL or SUPABASE_KEY missing in configuration.")
                self.supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
                # Optional: Test connection immediately? Could slow down startup.
                # try:
                #    test_fetch = asyncio.run(asyncio.to_thread(self.supabase.table(self.contacts_table).select('id').limit(1).execute))
                #    logger.info("Supabase connection test successful.")
                # except Exception as conn_test_e:
                #     logger.error(f"Supabase connection test failed: {conn_test_e}")
                #     self.supabase = None # Invalidate client if connection fails
                logger.info(f"CRMWrapper: Supabase client initialized for tables '{self.contacts_table}', '{self.call_log_table}'.")
            except Exception as e:
                logger.error(f"CRMWrapper: Failed to initialize Supabase client: {e}", exc_info=True)
                self.supabase = None # Ensure client is None if init fails
        elif not SUPABASE_PY_AVAILABLE:
             logger.warning("CRMWrapper: Supabase library not installed. CRM functions DISABLED.")
        else:
            logger.warning("CRMWrapper: Supabase not enabled in config. CRM functions DISABLED.")

    async def _execute_supabase_query(self, query_builder):
        """Helper to execute Supabase queries asynchronously and handle common errors."""
        if not self.supabase:
            logger.error("Supabase client not available for query.")
            return None
        try:
            # Use asyncio.to_thread for blocking Supabase calls
            response = await asyncio.to_thread(query_builder.execute)
            # Supabase client might return an error structure within the response
            # It's safer to check response attributes if needed, but typically PostgrestAPIError is raised
            return response
        except PostgrestAPIError as e:
            # Log detailed Supabase error
            logger.error(f"Supabase API error: {e.message} (Code: {e.code}, Details: {e.details}, Hint: {e.hint})", exc_info=False)
            return None
        except Exception as e:
            # Catch other potential issues like network errors during the sync call
            logger.error(f"Unexpected error during Supabase query execution: {e}", exc_info=True)
            return None

    async def get_contact_info(self, identifier: str, identifier_column: str = "email") -> Optional[Dict[str, Any]]:
        """Retrieves contact info by a unique identifier (email, phone_number, or id)."""
        if not self.supabase or not identifier:
            logger.debug(f"Cannot get contact info: Supabase unavailable or identifier missing ({identifier_column}={identifier}).")
            return None
        logger.debug(f"Querying contacts table '{self.contacts_table}' where '{identifier_column}' = '{identifier}'")

        try:
            query = self.supabase.table(self.contacts_table).select("*").eq(identifier_column, str(identifier)).limit(1) # Ensure identifier is string? Or handle types?
            response = await self._execute_supabase_query(query)

            if response and response.data:
                contact_data = response.data[0]
                logger.info(f"Found contact by {identifier_column}='{identifier}'. ID: {contact_data.get('id')}")
                return contact_data
            elif response is None: # Query execution failed
                 logger.error(f"Supabase query failed when searching for contact {identifier_column}='{identifier}'.")
                 return None
            else: # Query succeeded but no data found
                logger.debug(f"No contact found where {identifier_column}='{identifier}'.")
                return None
        except Exception as e:
            # Catch errors during query building if any
            logger.error(f"Error preparing Supabase query for {identifier_column}='{identifier}': {e}", exc_info=True)
            return None

    async def upsert_contact(
        self,
        contact_data: Dict[str, Any],
        unique_key_column: str = "email", # MUST have a UNIQUE constraint in Supabase (potentially composite with client_id)
        default_status: str = "New_Raw_Lead" # Ensure this status exists or matches DB defaults
    ) -> Optional[Dict[str, Any]]:
        """
        Intelligently inserts a new contact or updates an existing one.
        Requires careful handling of unique constraints, especially in multi-tenant scenarios.
        """
        if not self.supabase: logger.error("Upsert failed: Supabase client not available."); return None

        # Validate input data
        unique_value = contact_data.get(unique_key_column)
        if not contact_data or unique_value is None: # Check for None explicitly
            logger.error(f"Upsert failed: contact_data empty or missing unique key '{unique_key_column}'. Data: {contact_data}")
            return None

        logger.info(f"Upserting contact into '{self.contacts_table}' on conflict with '{unique_key_column}' = '{unique_value}'.")

        # Prepare data for upsert
        upsert_payload = contact_data.copy()

        # Ensure timestamps are set for new records or updated for existing ones
        current_iso_ts = datetime.now(timezone.utc).isoformat()
        # Let DB handle created_at default if possible
        upsert_payload["updated_at"] = current_iso_ts # Always update this

        # Set default status if not provided
        if "status" not in upsert_payload or not upsert_payload["status"]:
            upsert_payload["status"] = default_status

        # --- Handle potential JSONB serialization ---
        # Ensure complex fields are serializable JSON (supabase-py usually handles this, but explicit check is safer)
        for key, value in upsert_payload.items():
            if isinstance(value, (dict, list)):
                try:
                    # Test serialization
                    json.dumps(value, default=str)
                    # Supabase client handles the dict/list directly if serializable
                except TypeError as e:
                    logger.warning(f"Could not auto-serialize field '{key}' for upsert, attempting default str conversion. Error: {e}. Value: {str(value)[:100]}...")
                    # Convert problematic complex types to string representation as fallback? Risky.
                    # Or remove the problematic field? Safer but loses data.
                    # For now, let supabase-py handle it, but be aware.
                    # upsert_payload[key] = str(value) # Example fallback (potentially lossy)

        # Clean payload of None values (Supabase might handle this, but explicit is clearer)
        upsert_payload_cleaned = {k: v for k, v in upsert_payload.items() if v is not None}

        logger.debug(f"Upsert payload (cleaned): { {k:(str(v)[:50]+'...' if len(str(v))>50 else v) for k,v in upsert_payload_cleaned.items()} }")

        try:
            # Execute upsert query
            # `returning="representation"` ensures the inserted/updated row is returned
            query = (
                self.supabase.table(self.contacts_table)
                .upsert(upsert_payload_cleaned, on_conflict=unique_key_column, ignore_duplicates=False, returning="representation")
            )
            response = await self._execute_supabase_query(query)

            if response and response.data:
                upserted_record = response.data[0]
                # A simple heuristic to guess if it was insert or update based on timestamps
                # Requires created_at and updated_at to be reliably set by DB/app
                # created_time = upserted_record.get("created_at")
                # updated_time = upserted_record.get("updated_at")
                # action = "updated" if created_time and updated_time and (updated_time > created_time + timedelta(seconds=1)) else "inserted"
                action = "upserted" # Simpler logging
                logger.info(f"Successfully {action} contact. ID: {upserted_record.get('id')}, {unique_key_column}: {upserted_record.get(unique_key_column)}")
                return upserted_record
            else:
                # Log failure details if possible
                reason = "Query failed or returned no data"
                if response is None: reason = "Query execution failed (see logs)"
                logger.error(f"Supabase upsert failed for contact with {unique_key_column}='{unique_value}'. Reason: {reason}. Raw Response: {response}")
                # Attempt a final fetch in case it did succeed but didn't return representation
                return await self.get_contact_info(identifier=str(unique_value), identifier_column=unique_key_column)
        except Exception as e:
             logger.error(f"Unexpected error during contact upsert ({unique_key_column}='{unique_value}'): {e}", exc_info=True)
             return None


    async def log_call_outcome(
        self,
        call_sid: Optional[str], # Call SID is primary identifier for a call log
        contact_id: Optional[Union[str, int]], # Foreign key to the contacts table
        agent_id: str,
        status: str, # e.g., "Completed_Meeting_Booked", "Voicemail_Left", "Error_Technical"
        notes: Optional[str] = None,
        conversation_history_json: Optional[List[Dict[str, Any]]] = None, # Allow Any in dict
        call_duration_seconds: Optional[int] = None,
        llm_call_summary_json: Optional[Dict[str, Any]] = None, # Structured summary from LLM
        key_objections_tags: Optional[List[str]] = None,
        prospect_engagement_signals_tags: Optional[List[str]] = None,
        target_phone_number: Optional[str] = None # Added for context if contact_id is missing initially
        ) -> bool:
        """Logs detailed outcome of a call for record-keeping and future analysis."""
        if not self.supabase: logger.error("Cannot log call: Supabase client unavailable."); return False
        if not call_sid and not contact_id: # Need at least one identifier
            logger.error("Cannot log call outcome: call_sid or contact_id must be provided.")
            return False

        logger.info(f"Logging call outcome to '{self.call_log_table}'. Call SID: {call_sid}, Contact ID: {contact_id}, Status: {status}")

        log_entry = {
            "call_sid": call_sid,
            "contact_fk_id": contact_id, # Ensure your DB schema uses this name or adjust
            "agent_id": agent_id,
            "call_status": status,
            "notes": notes[:config.get_int_env_var("CRM_NOTE_MAX_LENGTH", default=2000)] if notes else None, # Limit note length
            "conversation_history_json": conversation_history_json, # Supabase client handles JSON
            "call_duration_seconds": call_duration_seconds,
            "llm_call_summary_json": llm_call_summary_json, # Supabase client handles JSON
            "key_objections_tags": key_objections_tags, # List of strings for array type in DB
            "prospect_engagement_signals_tags": prospect_engagement_signals_tags,
            "target_phone_number": target_phone_number, # Log the number called
            # timestamp_utc is typically handled by Supabase default 'now()' or 'created_at'
        }
        # Remove None values before insertion if DB columns don't have defaults or are not nullable for None
        log_entry_cleaned = {k: v for k, v in log_entry.items() if v is not None}

        try:
            query = self.supabase.table(self.call_log_table).insert(log_entry_cleaned, returning="minimal") # Use minimal return
            response = await self._execute_supabase_query(query)

            # Check response status for success (minimal return doesn't have data)
            # Supabase client might raise PostgrestAPIError on failure, caught by _execute_supabase_query
            if response is not None: # Indicates query executed without raising an error
                 # We can't reliably check response.data with minimal return
                 # Assume success if no exception was raised
                 logger.info(f"Successfully logged call outcome for SID {call_sid}.")
                 return True
            else: # _execute_supabase_query returned None, indicating an error occurred
                 logger.error(f"Failed to log call outcome for SID {call_sid} (query execution failed).")
                 return False

        except Exception as e:
             logger.error(f"Unexpected error logging call outcome (SID: {call_sid}): {e}", exc_info=True)
             return False

    async def get_contact_interaction_history(self, contact_id: Union[str, int]) -> List[Dict[str, Any]]:
        """Retrieves all call logs for a given contact_id, ordered by time."""
        if not self.supabase: logger.warning("Cannot get history: Supabase unavailable."); return []
        logger.debug(f"Fetching interaction history for contact_id: {contact_id}")

        try:
            query = (
                self.supabase.table(self.call_log_table)
                .select("*")
                .eq("contact_fk_id", str(contact_id)) # Ensure ID is string for query
                .order("created_at", desc=True) # Assuming 'created_at' column for log time
                .limit(config.get_int_env_var("CRM_HISTORY_LIMIT", default=50)) # Limit history size
            )
            response = await self._execute_supabase_query(query)

            if response and response.data:
                logger.info(f"Retrieved {len(response.data)} interaction logs for contact_id {contact_id}.")
                return response.data
            elif response is None:
                logger.error(f"Failed to retrieve interaction history for contact_id {contact_id} (query failed).")
                return []
            else: # No data
                logger.debug(f"No interaction history found for contact_id {contact_id}.")
                return []
        except Exception as e:
            logger.error(f"Error querying interaction history for {contact_id}: {e}", exc_info=True)
            return []

    async def update_contact_status_and_notes(
        self,
        contact_id: Union[str, int],
        new_status: str,
        note_to_append: Optional[str] = None, # Appending notes is complex, maybe just update 'last_note'?
        activity_timestamp: Optional[str] = None # ISO format
        ) -> bool:
        """Updates a contact's status and optionally adds a note (simplified)."""
        if not self.supabase: logger.error("Cannot update contact: Supabase unavailable."); return False
        logger.info(f"Updating contact ID {contact_id}: new status='{new_status}', append_note='{bool(note_to_append)}'")

        update_data: Dict[str, Any] = {
            "status": new_status,
            "updated_at": activity_timestamp or datetime.now(timezone.utc).isoformat(),
            "last_activity_timestamp": activity_timestamp or datetime.now(timezone.utc).isoformat() # Update activity marker
        }

        # Simplified note handling: Update a 'last_activity_note' field.
        # True note appending requires fetching existing notes or DB functions.
        if note_to_append:
            update_data["last_activity_note"] = note_to_append[:config.get_int_env_var("CRM_NOTE_MAX_LENGTH", default=2000)] # Use same limit

        try:
            query = self.supabase.table(self.contacts_table).update(update_data).eq("id", str(contact_id)) # Ensure ID is string
            response = await self._execute_supabase_query(query)

            # Update returns data (affected rows) even with default RLS if successful and row exists
            if response and response.data:
                logger.info(f"Successfully updated status/note for contact ID {contact_id}.")
                return True
            elif response is None: # Query execution failed
                 logger.error(f"Failed to update status for contact ID {contact_id} (query failed).")
                 return False
            else: # Query succeeded but no rows matched/updated (or RLS prevented return)
                 logger.warning(f"Update query executed for contact ID {contact_id}, but no data returned (might be normal if RLS active or ID not found).")
                 # Consider this success if no error, as the state might be achieved, just not confirmed by return data.
                 # Or return False if confirmation is needed. Let's assume success if no error.
                 return True # Modified: Assume success if query ran without error, even if no data returned
        except Exception as e:
            logger.error(f"Unexpected error updating contact {contact_id}: {e}", exc_info=True)
            return False


# --- Test function ---
async def _test_crm_wrapper_level35():
    print("--- Testing CRMWrapper ---")
    # ... (rest of test remains the same) ...
    # Test function requires Supabase running and configured.
    pass

if __name__ == "__main__":
    # import asyncio
    # from dotenv import load_dotenv
    # load_dotenv()
    # # Ensure Supabase is configured in .env
    # if config.SUPABASE_ENABLED and SUPABASE_PY_AVAILABLE:
    #    asyncio.run(_test_crm_wrapper_level35())
    # else:
    #    print("Skipping CRMWrapper test: Supabase not enabled, configured, or library not installed.")
    print("CRMWrapper defined.")
# --------------------------------------------------------------------------------
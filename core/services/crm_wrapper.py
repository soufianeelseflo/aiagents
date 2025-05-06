# core/services/crm_wrapper.py

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import json # For sanitizing conversation history if needed

# Import Supabase client library
from supabase import create_client, Client as SupabaseClient, PostgrestAPIError

# Import configuration centrally
try:
    from config import (
        SUPABASE_URL, SUPABASE_KEY,
        SUPABASE_CALL_LOG_TABLE, SUPABASE_CONTACTS_TABLE
    )
    SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_KEY)
except ImportError:
    logger.error("Failed to import Supabase config. Supabase integration disabled.")
    SUPABASE_URL, SUPABASE_KEY = None, None
    SUPABASE_CALL_LOG_TABLE, SUPABASE_CONTACTS_TABLE = "call_logs", "contacts" # Defaults won't matter
    SUPABASE_ENABLED = False

logger = logging.getLogger(__name__)

class CRMWrapper:
    """
    CRM wrapper using Supabase for persistence.
    Logs call outcomes to a 'call_logs' table and can retrieve contact info
    from a 'contacts' table (assumed populated by AcquisitionAgent).
    """

    def __init__(self):
        """ Initializes the Supabase client if configuration is provided. """
        self.supabase: Optional[SupabaseClient] = None
        self.call_log_table: str = SUPABASE_CALL_LOG_TABLE
        self.contacts_table: str = SUPABASE_CONTACTS_TABLE

        if SUPABASE_ENABLED:
            try:
                self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
                logger.info(f"Supabase client initialized successfully for URL: {SUPABASE_URL[:20]}...")
                # Optionally test connection here if needed (e.g., try fetching schema)
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}", exc_info=True)
                self.supabase = None # Ensure client is None if init fails
        else:
            logger.warning("Supabase URL or Key not configured. CRMWrapper will not interact with database.")

    async def get_contact_info(self, phone_number: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves contact info from the Supabase 'contacts' table based on phone number.
        Assumes a 'contacts' table exists with a 'phone_number' column (unique index recommended)
        and columns like 'id', 'first_name', 'last_name', 'company_name', 'email', etc.

        Args:
            phone_number: The phone number (E.164 format preferred) to search for.

        Returns:
            A dictionary containing contact data or None if not found or error.
        """
        if not self.supabase:
            logger.warning("Supabase not configured, cannot get contact info.")
            return None
        if not phone_number:
            logger.warning("get_contact_info called with empty phone number.")
            return None

        logger.debug(f"Querying Supabase contacts table '{self.contacts_table}' for phone: {phone_number}")
        try:
            # Use Supabase client to select data
            # Assumes 'phone_number' is the column name in your Supabase table
            response = await asyncio.to_thread(
                self.supabase.table(self.contacts_table)
                .select("*") # Select all columns for the contact
                .eq("phone_number", phone_number) # Filter by phone number
                .limit(1) # Expect only one match
                .execute
            )

            # Check response and data
            if response.data:
                contact_data = response.data[0]
                logger.info(f"Found contact in Supabase for {phone_number}. ID: {contact_data.get('id')}")
                return contact_data
            else:
                logger.info(f"No contact found in Supabase for {phone_number}.")
                return None

        except PostgrestAPIError as e:
            logger.error(f"Supabase API error querying contacts table '{self.contacts_table}': {e.message}", exc_info=False)
            logger.debug(f"Supabase error details: {e.details}, code: {e.code}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error querying Supabase contacts: {e}", exc_info=True)
            return None

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
        Logs the outcome of a call by inserting a record into the Supabase 'call_logs' table.
        Assumes 'call_logs' table exists with columns matching CRM_LOG_HEADERS.

        Args:
            phone_number: The phone number contacted.
            status: Short status description (e.g., "Meeting Booked").
            notes: Detailed call summary or key outcome notes.
            agent_id: The ID of the agent handling the call.
            call_sid: Optional telephony identifier (e.g., from Twilio).
            conversation_history: Optional full transcript history (stored as JSON).

        Returns:
            True if logging to Supabase was successful, False otherwise.
        """
        if not self.supabase:
            logger.warning("Supabase not configured, cannot log call outcome.")
            return False

        logger.info(f"Logging call outcome to Supabase table '{self.call_log_table}' for {phone_number}. Status: {status}")

        # Prepare data for insertion
        # Ensure data types match Supabase table schema
        log_entry = {
            # 'timestamp_utc' is often handled by Supabase default 'now()' or 'created_at' timestamp
            "agent_id": agent_id,
            "call_sid": call_sid if call_sid else None, # Use None if not provided
            "target_phone_number": phone_number,
            "call_status": status,
            "notes": notes[:2000] if notes else None, # Truncate notes if necessary, use None if empty
            # Store conversation history as JSONB in Supabase for flexibility
            "conversation_history_json": json.dumps(conversation_history) if conversation_history else None
            # Add contact_id if found and table schema requires it
            # 'contact_id': contact_id (Need to fetch this first if association is needed)
        }
        # Remove keys with None values if the Supabase table doesn't handle them well
        log_entry = {k: v for k, v in log_entry.items() if v is not None}


        try:
            # Use Supabase client to insert data
            # Use asyncio.to_thread to run the synchronous Supabase client call in a separate thread
            response = await asyncio.to_thread(
                self.supabase.table(self.call_log_table)
                .insert(log_entry)
                .execute
            )

            # Check if insertion was successful (Supabase typically returns data on success)
            if response.data:
                logger.info(f"Successfully logged call outcome to Supabase for {phone_number}. Record ID: {response.data[0].get('id')}")
                return True
            else:
                # This case might indicate an issue even without an explicit error
                logger.warning(f"Supabase insert into '{self.call_log_table}' completed but returned no data. Response: {response}")
                # Consider it a failure if no data is returned after insert
                return False

        except PostgrestAPIError as e:
            logger.error(f"Supabase API error inserting into call_logs table '{self.call_log_table}': {e.message}", exc_info=False)
            logger.debug(f"Supabase error details: {e.details}, code: {e.code}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error logging call outcome to Supabase: {e}", exc_info=True)
            return False

# (Conceptual Test Runner - Requires Supabase Setup)
async def main():
    print("Testing CRMWrapper (Supabase - Final)...")
    # Requires SUPABASE_URL and SUPABASE_KEY in .env
    # Requires 'contacts' and 'call_logs' tables in Supabase matching expected schema
    if not SUPABASE_ENABLED:
         print("Skipping test: Supabase not configured in .env")
         return

    wrapper = CRMWrapper()
    if not wrapper.supabase: # Check if client initialized
         print("Skipping test: Supabase client failed to initialize.")
         return

    test_phone = "+15556667777" # Use a unique number for testing

    # --- Test Contact Lookup (Assumes contact exists or table handles not found) ---
    print(f"\nAttempting get_contact_info for: {test_phone}")
    # Pre-insert a contact manually in Supabase for this test to succeed:
    # supabase.table("contacts").insert({"phone_number": test_phone, "first_name": "Supabase", "last_name": "Test"}).execute()
    contact = await wrapper.get_contact_info(test_phone)
    if contact: print(f"Found contact: {contact}")
    else: print("Contact not found (or Supabase error). Ensure contact exists for lookup test.")

    # --- Test Call Logging ---
    print(f"\nAttempting log_call_outcome for: {test_phone}")
    success = await wrapper.log_call_outcome(
        phone_number=test_phone,
        status="Test Log Entry",
        notes="This is a test log generated by crm_wrapper_v5_supabase.",
        agent_id="SupabaseTestAgent",
        call_sid="CATestSupabase",
        conversation_history=[{"role": "test", "text": "Log entry test."}]
    )
    print(f"CRM log successful: {success}")
    if success:
         print(f"Check Supabase table '{wrapper.call_log_table}' for the new entry.")

if __name__ == "__main__":
    # import asyncio
    # import config # Ensure config is loaded
    # asyncio.run(main()) # Uncomment to run test (requires async context and Supabase setup)
    print("CRMWrapper structure defined (Supabase - Final). Requires Supabase setup and tables for real use.")


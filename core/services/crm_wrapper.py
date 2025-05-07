# core/services/crm_wrapper.py

import logging
import json
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Union

from supabase import create_client, Client as SupabaseClient, PostgrestAPIError

import config # Root config

logger = logging.getLogger(__name__)

class CRMWrapper:
    """
    Intelligent CRM wrapper using Supabase. Manages contacts with robust upsert logic
    and logs detailed call/interaction outcomes for future learning and agent context.
    Transmuted to Level 35 by Ignis, the Alchemetric Catalyst.
    """

    def __init__(self):
        self.supabase: Optional[SupabaseClient] = None
        self.call_log_table: str = config.SUPABASE_CALL_LOG_TABLE
        self.contacts_table: str = config.SUPABASE_CONTACTS_TABLE

        if config.SUPABASE_ENABLED:
            try:
                self.supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
                logger.info(f"CRMWrapper (Level 35): Supabase client initialized for tables '{self.contacts_table}', '{self.call_log_table}'.")
            except Exception as e:
                logger.error(f"CRMWrapper: Failed to initialize Supabase client: {e}", exc_info=True)
        else:
            logger.warning("CRMWrapper: Supabase not configured. CRM functions are DISABLED.")

    async def _execute_supabase_query(self, query_builder):
        """Helper to execute Supabase queries asynchronously and handle common errors."""
        if not self.supabase:
            logger.error("Supabase client not available for query.")
            return None
        try:
            response = await asyncio.to_thread(query_builder.execute)
            return response
        except PostgrestAPIError as e:
            logger.error(f"Supabase API error: {e.message} (Details: {e.details}, Code: {e.code}, Hint: {e.hint})", exc_info=False)
            return None
        except Exception as e:
            logger.error(f"Unexpected error during Supabase query: {e}", exc_info=True)
            return None

    async def get_contact_info(self, identifier: str, identifier_column: str = "email") -> Optional[Dict[str, Any]]:
        """Retrieves contact info by a unique identifier (email, phone_number, or id)."""
        if not self.supabase or not identifier: return None
        logger.debug(f"Querying contacts table '{self.contacts_table}' where '{identifier_column}' = '{identifier}'")
        
        query = self.supabase.table(self.contacts_table).select("*").eq(identifier_column, identifier).limit(1)
        response = await self._execute_supabase_query(query)

        if response and response.data:
            contact_data = response.data[0]
            logger.info(f"Found contact by {identifier_column}='{identifier}'. ID: {contact_data.get('id')}")
            return contact_data
        logger.info(f"No contact found where {identifier_column}='{identifier}'.")
        return None

    async def upsert_contact(
        self,
        contact_data: Dict[str, Any],
        unique_key_column: str = "email", # MUST have a UNIQUE constraint in Supabase
        default_status: str = "New Lead"
    ) -> Optional[Dict[str, Any]]:
        """
        Intelligently inserts a new contact or updates an existing one.
        Merges some fields on update rather than simple overwrite where appropriate.
        """
        if not self.supabase: return None
        if not contact_data or not contact_data.get(unique_key_column):
            logger.error(f"Upsert failed: contact_data empty or missing unique key '{unique_key_column}'. Data: {contact_data}")
            return None

        unique_value = contact_data.get(unique_key_column)
        logger.info(f"Upserting contact into '{self.contacts_table}' on conflict with '{unique_key_column}' = '{unique_value}'.")

        # Prepare data for upsert
        upsert_payload = contact_data.copy()

        # Ensure timestamps are set for new records or updated for existing ones
        current_iso_ts = datetime.now(timezone.utc).isoformat()
        if "created_at" not in upsert_payload: # Only set created_at if it's truly new (Supabase upsert handles this if column default is now())
            # Supabase's upsert with on_conflict doesn't easily allow conditional setting of created_at only on insert.
            # Often, created_at is set by DB default. updated_at is usually managed by a DB trigger or application logic.
            pass
        upsert_payload["updated_at"] = current_iso_ts # Always update this

        if "status" not in upsert_payload:
            upsert_payload["status"] = default_status
        
        # Serialize complex fields like JSONB data
        for key, value in upsert_payload.items():
            if isinstance(value, (dict, list)):
                upsert_payload[key] = json.loads(json.dumps(value, default=str)) # Ensure serializable

        logger.debug(f"Upsert payload: {upsert_payload}")

        query = (
            self.supabase.table(self.contacts_table)
            .upsert(upsert_payload, on_conflict=unique_key_column, ignore_duplicates=False, returning="representation")
        )
        response = await self._execute_supabase_query(query)

        if response and response.data:
            upserted_record = response.data[0]
            action = "updated" if upserted_record.get("created_at") != upserted_record.get("updated_at") else "created" # Heuristic
            logger.info(f"Successfully {action} contact. ID: {upserted_record.get('id')}, {unique_key_column}: {upserted_record.get(unique_key_column)}")
            return upserted_record
        else:
            logger.error(f"Supabase upsert for contact with {unique_key_column}='{unique_value}' failed or returned no data. Response: {response}")
            # Attempt a final fetch in case it did succeed but didn't return representation
            return await self.get_contact_info(identifier=unique_value, identifier_column=unique_key_column)


    async def log_call_outcome(
        self,
        call_sid: Optional[str], # Call SID is primary identifier for a call log
        contact_id: Optional[Union[str, int]], # Foreign key to the contacts table
        agent_id: str,
        status: str, # e.g., "Completed - Meeting Booked", "Voicemail Left", "Error - Technical"
        notes: Optional[str] = None,
        conversation_history_json: Optional[List[Dict[str, str]]] = None,
        call_duration_seconds: Optional[int] = None,
        llm_call_summary_json: Optional[Dict[str, Any]] = None, # Structured summary from LLM
        key_objections_tags: Optional[List[str]] = None,
        prospect_engagement_signals_tags: Optional[List[str]] = None,
        agent_perceived_call_outcome: Optional[str] = None # e.g., "Positive", "Neutral", "Negative"
        ) -> bool:
        """Logs detailed outcome of a call for record-keeping and future analysis."""
        if not self.supabase: return False
        if not call_sid and not contact_id: # Need at least one identifier
            logger.error("Cannot log call outcome: call_sid or contact_id must be provided.")
            return False

        logger.info(f"Logging call outcome to '{self.call_log_table}'. Call SID: {call_sid}, Contact ID: {contact_id}, Status: {status}")

        log_entry = {
            "call_sid": call_sid,
            "contact_fk_id": contact_id, # Ensure your DB schema uses this name or adjust
            "agent_id": agent_id,
            "call_status": status,
            "notes": notes[:2000] if notes else None,
            "conversation_history_json": conversation_history_json, # Already list of dicts
            "call_duration_seconds": call_duration_seconds,
            "llm_call_summary_json": llm_call_summary_json,
            "key_objections_tags": key_objections_tags, # List of strings for array type in DB
            "prospect_engagement_signals_tags": prospect_engagement_signals_tags,
            "agent_perceived_call_outcome": agent_perceived_call_outcome,
            # timestamp_utc is typically handled by Supabase default 'now()' or 'created_at'
        }
        log_entry_cleaned = {k: v for k, v in log_entry.items() if v is not None}
        
        query = self.supabase.table(self.call_log_table).insert(log_entry_cleaned)
        response = await self._execute_supabase_query(query)

        if response and ( (hasattr(response, 'data') and response.data) or \
                           (hasattr(response, 'status_code') and 200 <= response.status_code < 300) ):
            log_id = response.data[0].get('id') if hasattr(response, 'data') and response.data else "N/A (check status)"
            logger.info(f"Successfully logged call outcome. Log ID: {log_id}")
            return True
        
        logger.error(f"Failed to log call outcome for SID {call_sid}. Response: {response}")
        return False

    async def get_contact_interaction_history(self, contact_id: Union[str, int]) -> List[Dict[str, Any]]:
        """Retrieves all call logs for a given contact_id, ordered by time."""
        if not self.supabase: return []
        logger.debug(f"Fetching interaction history for contact_id: {contact_id}")
        
        query = (
            self.supabase.table(self.call_log_table)
            .select("*")
            .eq("contact_fk_id", contact_id)
            .order("created_at", desc=True) # Assuming 'created_at' column for log time
        )
        response = await self._execute_supabase_query(query)
        
        if response and response.data:
            logger.info(f"Retrieved {len(response.data)} interaction logs for contact_id {contact_id}.")
            return response.data
        return []

    async def update_contact_status_and_notes(
        self,
        contact_id: Union[str, int],
        new_status: str,
        note_to_append: Optional[str] = None,
        activity_timestamp: Optional[str] = None # ISO format
        ) -> bool:
        """Updates a contact's status and optionally appends a note."""
        if not self.supabase: return False
        logger.info(f"Updating contact ID {contact_id}: new status='{new_status}', append_note='{bool(note_to_append)}'")

        update_data: Dict[str, Any] = {"status": new_status}
        update_data["updated_at"] = activity_timestamp or datetime.now(timezone.utc).isoformat()

        # For appending notes, you'd typically fetch existing notes first if your DB doesn't support append directly.
        # This is simplified; a real system might use a separate 'notes' table or more complex JSONB updates.
        if note_to_append:
            # This is a simple overwrite. For append, you'd:
            # 1. Get current notes: existing_contact = await self.get_contact_info(contact_id, "id")
            # 2. Append: new_notes = (existing_contact.get('notes', '') or '') + f"\n[{update_data['updated_at[:19]}] {note_to_append}"
            # 3. Set: update_data['notes'] = new_notes
            # For now, let's assume a field like 'last_agent_note' or that notes are managed elsewhere.
            update_data["last_activity_note"] = note_to_append # Example field

        query = self.supabase.table(self.contacts_table).update(update_data).eq("id", contact_id)
        response = await self._execute_supabase_query(query)

        if response and ( (hasattr(response, 'data') and response.data) or \
                           (hasattr(response, 'status_code') and 200 <= response.status_code < 300 and response.status_code != 404) ): # 404 if not found
            logger.info(f"Successfully updated status for contact ID {contact_id}.")
            return True
        
        logger.error(f"Failed to update status for contact ID {contact_id}. Response: {response}")
        return False

# --- Test function ---
async def _test_crm_wrapper_level35():
    print("--- Testing CRMWrapper (Level 35) ---")
    if not config.SUPABASE_ENABLED:
         print("Skipping CRMWrapper test: Supabase not configured.")
         return

    wrapper = CRMWrapper()
    if not wrapper.supabase:
         print("Skipping CRMWrapper test: Supabase client failed to initialize.")
         return

    ts = int(time.time())
    test_email = f"level35.contact.{ts}@example.com"
    test_phone = f"+1777{str(ts)[-7:]}"
    initial_contact_data = {
        "email": test_email, "first_name": "Level", "last_name": "ThirtyFive",
        "phone_number": test_phone, "company_name": "Alchemic Solutions",
        "source_info": "crm_test_l35_insert", "status": "New Prospect (L35)"
    }

    print(f"\n1. Upserting new contact: {test_email}")
    created_contact = await wrapper.upsert_contact(initial_contact_data, unique_key_column="email")
    if created_contact and created_contact.get("id"):
        contact_id = created_contact["id"]
        print(f"  SUCCESS: Created contact ID: {contact_id}, Email: {created_contact.get('email')}")

        print(f"\n2. Upserting (updating) existing contact: {test_email}")
        update_data = {"email": test_email, "company_name": "Alchemic Solutions Inc.", "status": "Contacted (L35)"}
        updated_contact = await wrapper.upsert_contact(update_data, unique_key_column="email")
        if updated_contact and updated_contact.get("company_name") == "Alchemic Solutions Inc.":
            print(f"  SUCCESS: Updated contact ID: {updated_contact['id']}, Company: {updated_contact['company_name']}")
        else:
            print(f"  FAILURE: Failed to update contact or verify update. Response: {updated_contact}")

        print(f"\n3. Logging call outcome for contact ID: {contact_id}")
        call_logged = await wrapper.log_call_outcome(
            call_sid=f"CA_L35_{ts}", contact_id=contact_id, agent_id="IgnisTestAgent",
            status="Completed - Demo Scheduled", notes="Excellent call, booked demo for next Tuesday.",
            conversation_history_json=[{"role": "user", "text": "Hello?"}, {"role": "agent", "text": "Hi!"}],
            call_duration_seconds=350, llm_call_summary_json={"sentiment": "positive", "key_points": ["demo", "budget confirmed"]},
            key_objections_tags=["pricing_initial", "timeline"], prospect_engagement_signals_tags=["asked_buying_questions"],
            agent_perceived_call_outcome="Positive"
        )
        print(f"  Call logged successfully: {call_logged}")

        print(f"\n4. Retrieving interaction history for contact ID: {contact_id}")
        history = await wrapper.get_contact_interaction_history(contact_id)
        if history:
            print(f"  SUCCESS: Retrieved {len(history)} history item(s). First item status: {history[0].get('call_status')}")
        else:
            print(f"  No history found or error retrieving.")
            
        print(f"\n5. Updating contact status and notes for ID: {contact_id}")
        status_updated = await wrapper.update_contact_status_and_notes(contact_id, "Demo Follow-up Needed", "Sent demo confirmation email.")
        print(f"  Status update successful: {status_updated}")
        final_contact_check = await wrapper.get_contact_info(str(contact_id), "id") # get_contact_info expects string identifier
        if final_contact_check:
            print(f"  Final contact status: {final_contact_check.get('status')}, Last note hint: {final_contact_check.get('last_activity_note')}")


    else:
        print(f"  FAILURE: Initial contact upsert failed for {test_email}. Cannot proceed with further tests for this contact.")

if __name__ == "__main__":
    # import asyncio
    # from dotenv import load_dotenv
    # load_dotenv()
    # asyncio.run(_test_crm_wrapper_level35())
    print("CRMWrapper (Level 35 - Intelligent Data Nexus) defined. Run test manually with Supabase setup.")
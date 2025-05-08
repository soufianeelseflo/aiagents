# boutique_ai_project/core/database_setup.py

import logging
import asyncio
from typing import List, Dict, Any, Optional

from supabase import create_client, Client as SupabaseClient, PostgrestAPIError

import config # Root config

logger = logging.getLogger(__name__)

# --- Table Definitions (Updated for Multi-Tenancy & Agent Factory) ---

# Helper function to add common columns
def get_common_columns(include_client_id: bool = True):
    cols = [
        {"name": "id", "type": "uuid", "primary_key": True, "default": "extensions.uuid_generate_v4()"},
        {"name": "created_at", "type": "timestamptz", "default": "now()", "nullable": False},
        {"name": "updated_at", "type": "timestamptz", "default": "now()", "nullable": False},
    ]
    if include_client_id:
        # Assuming a central 'clients' table or using user IDs if Supabase Auth is integrated for clients
        # For simplicity now, let's assume a client_id (UUID or TEXT) identifies the tenant.
        # This column MUST be used in RLS policies.
        cols.append({"name": "client_id", "type": "uuid", "nullable": True, "index": True})
        # If clients are users in Supabase Auth:
        # cols.append({"name": "user_id", "type": "uuid", "nullable": True, "foreign_key": {"table": "auth.users", "column": "id", "on_delete": "CASCADE"}, "index": True})
    return cols

TABLE_DEFINITIONS = {
    # --- Core CRM Tables (Now Multi-Tenant) ---
    config.SUPABASE_CONTACTS_TABLE: get_common_columns() + [
        {"name": "email", "type": "text", "unique": False, "nullable": True, "index": True}, # Uniqueness should be per client_id, handled by RLS/app logic or composite key
        {"name": "phone_number", "type": "text", "unique": False, "nullable": True, "index": True}, # Unique per client_id
        {"name": "first_name", "type": "text", "nullable": True},
        {"name": "last_name", "type": "text", "nullable": True},
        {"name": "company_name", "type": "text", "nullable": True, "index": True},
        {"name": "domain", "type": "text", "nullable": True, "index": True},
        {"name": "job_title", "type": "text", "nullable": True},
        {"name": "linkedin_url", "type": "text", "nullable": True},
        {"name": "company_linkedin_url", "type": "text", "nullable": True},
        {"name": "status", "type": "text", "default": "'New_Raw_Lead'", "index": True},
        {"name": "source_info", "type": "text", "nullable": True},
        {"name": "correlation_id_clay", "type": "text", "nullable": True, "unique": False, "index": True}, # Unique per client_id likely
        {"name": "last_activity_timestamp", "type": "timestamptz", "nullable": True},
        {"name": "raw_lead_data_json", "type": "jsonb", "nullable": True},
        {"name": "clay_enrichment_data_json", "type": "jsonb", "nullable": True},
        {"name": "llm_qualification_score", "type": "integer", "nullable": True},
        {"name": "llm_inferred_pain_point_1", "type": "text", "nullable": True},
        {"name": "llm_inferred_pain_point_2", "type": "text", "nullable": True},
        {"name": "llm_suggested_hook", "type": "text", "nullable": True},
        {"name": "llm_qualification_reasoning", "type": "text", "nullable": True},
        {"name": "llm_analysis_confidence", "type": "text", "nullable": True},
        {"name": "llm_suggested_next_action", "type": "text", "nullable": True},
        {"name": "llm_full_analysis_json", "type": "jsonb", "nullable": True},
        {"name": "assigned_sales_agent_id", "type": "text", "nullable": True, "index": True}, # Could be internal agent or client's deployed agent ID
        {"name": "tags", "type": "text[]", "nullable": True},
        {"name": "notes", "type": "text", "nullable": True},
        {"name": "company_size_range", "type": "text", "nullable": True},
        {"name": "industry", "type": "text", "nullable": True},
        {"name": "company_description", "type": "text", "nullable": True},
        {"name": "last_activity_note", "type": "text", "nullable": True},
        # --- Fields for Client Management ---
        {"name": "is_client_account", "type": "boolean", "default": "false", "index": True}, # Flag if this contact represents a paying client
        {"name": "client_config_json", "type": "jsonb", "nullable": True} # Stores agent config for this client
    ],
    config.SUPABASE_CALL_LOGS_TABLE: get_common_columns() + [
        {"name": "call_sid", "type": "text", "unique": True, "nullable": False, "index": True},
        {"name": "contact_fk_id", "type": "uuid", "nullable": True, "foreign_key": {"table": config.SUPABASE_CONTACTS_TABLE, "column": "id", "on_delete": "SET NULL"}},
        {"name": "agent_id", "type": "text", "nullable": False, "index": True}, # ID of the specific SalesAgent instance
        {"name": "target_phone_number", "type": "text", "nullable": True},
        {"name": "call_status", "type": "text", "nullable": False},
        {"name": "call_outcome_category_llm", "type": "text", "nullable": True},
        {"name": "call_duration_seconds", "type": "integer", "nullable": True},
        {"name": "conversation_history_json", "type": "jsonb", "nullable": True},
        {"name": "notes", "type": "text", "nullable": True},
        {"name": "llm_call_summary_json", "type": "jsonb", "nullable": True},
        {"name": "key_objections_tags", "type": "text[]", "nullable": True},
        {"name": "prospect_engagement_signals_tags", "type": "text[]", "nullable": True}
        # Removed agent_perceived_call_outcome as it's redundant with call_outcome_category_llm
    ],
    config.SUPABASE_RESOURCES_TABLE: get_common_columns() + [ # Resources might be client-specific or shared
        {"name": "service_name", "type": "text", "nullable": False, "index": True},
        {"name": "resource_type", "type": "text", "nullable": False, "index": True},
        {"name": "resource_data", "type": "jsonb", "nullable": True}, # Encrypt sensitive data here if possible
        {"name": "status", "type": "text", "default": "'active'", "nullable": False, "index": True},
        {"name": "expiry_timestamp", "type": "timestamptz", "nullable": True, "index": True},
        {"name": "notes", "type": "text", "nullable": True},
        {"name": "fingerprint_profile_summary", "type": "jsonb", "nullable": True},
        # Unique constraint might be (client_id, service_name, resource_type) if resources are per-client
    ],

    # --- NEW Tables for Agent Factory ---
    "agent_templates": get_common_columns(include_client_id=False) + [ # Templates are global
        {"name": "template_name", "type": "text", "unique": True, "nullable": False}, # e.g., "SaaS_Sales_Agent_V1"
        {"name": "agent_type", "type": "text", "nullable": False}, # e.g., "SalesAgent", "AcquisitionAgent"
        {"name": "description", "type": "text", "nullable": True},
        {"name": "base_system_prompt", "type": "text", "nullable": True},
        {"name": "core_logic_config", "type": "jsonb", "nullable": True}, # e.g., default strategy goals, analysis params
        {"name": "required_config_params", "type": "text[]", "nullable": True}, # List of keys client needs to provide
        {"name": "version", "type": "integer", "default": "1"}
    ],
    "client_agent_instances": get_common_columns() + [ # Tracks deployed agents for clients
        {"name": "client_contact_fk_id", "type": "uuid", "nullable": False, "foreign_key": {"table": config.SUPABASE_CONTACTS_TABLE, "column": "id", "on_delete": "CASCADE"}},
        {"name": "agent_template_fk_id", "type": "uuid", "nullable": False, "foreign_key": {"table": "agent_templates", "column": "id", "on_delete": "RESTRICT"}},
        {"name": "instance_name", "type": "text", "nullable": True}, # e.g., Client A's Sales Agent
        {"name": "deployment_id", "type": "text", "nullable": True, "index": True}, # ID from DeploymentManager (e.g., container ID)
        {"name": "deployment_status", "type": "text", "default": "'pending'", "index": True}, # pending, deploying, running, stopped, failed, deleted
        {"name": "agent_configuration_override_json", "type": "jsonb", "nullable": True}, # Client-specific overrides for the template
        {"name": "last_health_check_timestamp", "type": "timestamptz", "nullable": True},
        {"name": "performance_metrics_json", "type": "jsonb", "nullable": True} # Store KPIs here
    ]
}

# --- SQL Commands ---
ENABLE_UUID_OSSP_SQL = "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\" WITH SCHEMA extensions;"

CREATE_UPDATE_TIMESTAMP_FUNCTION_SQL = """
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'update_updated_at_column') THEN
    CREATE FUNCTION public.update_updated_at_column()
    RETURNS TRIGGER AS $function$
    BEGIN
       -- Check if the 'updated_at' column exists before trying to set it
       IF TG_OP = 'UPDATE' AND (NEW IS DISTINCT FROM OLD) THEN
          NEW.updated_at = now();
       END IF;
       -- For INSERT, rely on the column default 'now()'
       RETURN NEW;
    END;
    $function$ language 'plpgsql';
    COMMENT ON FUNCTION public.update_updated_at_column() IS 'Updates the updated_at timestamp on modification if column exists';
  END IF;
END $$;
"""

# --- Helper Functions ---
def create_client() -> Optional[SupabaseClient]:
    """Creates Supabase client using SERVICE ROLE KEY from config."""
    if not config.SUPABASE_ENABLED: logger.warning("Supabase disabled."); return None
    if not config.SUPABASE_KEY or not config.SUPABASE_URL: logger.error("SUPABASE_URL or SUPABASE_KEY (Service Role) missing."); return None
    try:
        # Explicitly use service role key for DDL
        client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
        logger.info("Supabase client created for database setup using Service Role Key.")
        return client
    except Exception as e: logger.error(f"Failed to create Supabase client: {e}", exc_info=True); return None

async def execute_sql_via_rpc(supabase: SupabaseClient, sql: str):
    """Executes raw SQL via Supabase RPC (requires 'execute_sql' function)."""
    # Requires: See previous version's comments for function creation SQL
    logger.debug(f"Attempting SQL via RPC: {sql[:150]}...")
    try:
        # Using asyncio.to_thread as supabase-py's rpc might be blocking
        await asyncio.to_thread(supabase.rpc('execute_sql', {'sql_command': sql}).execute)
        logger.debug(f"SQL executed successfully via RPC: {sql[:50]}...")
        return True
    except Exception as e:
        if isinstance(e, PostgrestAPIError) and e.code == '42883': # undefined_function
             logger.error("RPC function 'execute_sql' not found in Supabase. Cannot execute DDL programmatically.")
             logger.error("Please create the function manually in Supabase SQL Editor (see database_setup.py comments).")
        elif isinstance(e, PostgrestAPIError) and e.code == '42501': # permission_denied
             logger.error("Permission denied for RPC function 'execute_sql'. Ensure 'service_role' has execute grant.")
        else: logger.error(f"Error executing SQL via RPC: {e}", exc_info=True)
        logger.warning(f"Manual Action Required: Execute SQL in Supabase Editor:\n{sql}")
        return False

async def check_table_exists(supabase: SupabaseClient, table_name: str) -> bool:
    """Checks if table exists using RPC call to information_schema."""
    # Requires function:
    # CREATE OR REPLACE FUNCTION check_table_exists(schema_name text, table_name_input text)
    # RETURNS boolean AS $$
    # BEGIN
    #   RETURN EXISTS (
    #      SELECT FROM information_schema.tables
    #      WHERE table_schema = schema_name
    #      AND table_name = table_name_input
    #   );
    # END;
    # $$ LANGUAGE plpgsql SECURITY DEFINER;
    # GRANT EXECUTE ON FUNCTION public.check_table_exists(text, text) TO service_role;
    logger.debug(f"Checking existence of table: public.{table_name}")
    try:
        response = await asyncio.to_thread(
            supabase.rpc('check_table_exists', {'schema_name': 'public', 'table_name_input': table_name}).execute
        )
        if response.data is True:
            logger.debug(f"Table 'public.{table_name}' exists.")
            return True
        elif response.data is False:
             logger.debug(f"Table 'public.{table_name}' does not exist.")
             return False
        else: # RPC succeeded but returned unexpected data
             logger.warning(f"RPC check_table_exists returned unexpected data: {response.data}")
             return False # Assume not exists if check is inconclusive
    except PostgrestAPIError as e:
        if e.code == '42883': # undefined_function
            logger.error("RPC function 'check_table_exists' not found. Cannot reliably check table existence programmatically.")
            logger.error("Please create the function manually (see database_setup.py comments). Assuming table might not exist.")
            return False # Assume not exists if we can't check
        else:
            logger.error(f"Error checking table existence via RPC for '{table_name}': {e.message}")
            return False # Assume not exists on error
    except Exception as e:
        logger.error(f"Unexpected error checking table '{table_name}' existence: {e}", exc_info=True)
        return False # Assume not exists on error


async def create_table(supabase: SupabaseClient, table_name: str, columns: List[Dict[str, Any]]):
    """Constructs and attempts to execute CREATE TABLE statement via RPC."""
    logger.info(f"Constructing CREATE TABLE statement for '{table_name}'...")
    column_defs = []
    constraints = []
    primary_key_col = None

    for col in columns:
        col_sql = f"\"{col['name']}\" {col['type']}"
        if col.get("primary_key"): primary_key_col = col['name']
        if not col.get("nullable", True) and not col.get("primary_key"): col_sql += " NOT NULL"
        if col.get("default") is not None: col_sql += f" DEFAULT {col['default']}"
        # Unique constraints handled separately below for clarity if needed as composite later
        # if col.get("unique"): constraints.append(f"UNIQUE (\"{col['name']}\")")
        column_defs.append(col_sql)
        if col.get("foreign_key"):
            fk = col["foreign_key"]
            on_delete = fk.get("on_delete", "NO ACTION").upper()
            on_update = fk.get("on_update", "NO ACTION").upper()
            constraints.append(
                f"CONSTRAINT fk_{table_name}_{col['name']}_{fk['table']} "
                f"FOREIGN KEY (\"{col['name']}\") REFERENCES public.\"{fk['table']}\" (\"{fk['column']}\") "
                f"ON DELETE {on_delete} ON UPDATE {on_update}"
            )
        if col.get("unique"): # Add unique constraints here
             constraints.append(f"CONSTRAINT uq_{table_name}_{col['name']} UNIQUE (\"{col['name']}\")")


    if primary_key_col: constraints.insert(0, f"PRIMARY KEY (\"{primary_key_col}\")")

    # Use standard CREATE TABLE (without IF NOT EXISTS, as we check first)
    create_sql = f"CREATE TABLE public.\"{table_name}\" ({', '.join(column_defs + constraints)});"
    
    if not await execute_sql_via_rpc(supabase, create_sql):
        logger.error(f"Failed to programmatically create table '{table_name}'. Manual creation required.")

async def apply_trigger(supabase: SupabaseClient, table_name: str, trigger_name: str, trigger_sql: str):
    """Applies a trigger using CREATE OR REPLACE."""
    logger.info(f"Ensuring trigger '{trigger_name}' exists for table '{table_name}'.")
    if not await execute_sql_via_rpc(supabase, trigger_sql):
         logger.warning(f"Failed to programmatically apply trigger '{trigger_name}' to '{table_name}'. Verify manually.")

async def create_index(supabase: SupabaseClient, table_name: str, column_name: str):
    """Creates an index if it doesn't exist."""
    index_name = f"idx_{table_name}_{column_name}"
    index_sql = f"CREATE INDEX IF NOT EXISTS \"{index_name}\" ON public.\"{table_name}\" (\"{column_name}\");"
    logger.info(f"Ensuring index '{index_name}' exists on '{table_name}'.")
    if not await execute_sql_via_rpc(supabase, index_sql):
        logger.warning(f"Failed to programmatically create index '{index_name}'. Verify manually.")

async def setup_supabase_tables(supabase: SupabaseClient):
    """Main function to set up extensions, functions, tables, triggers, and indexes."""
    if not supabase: logger.error("Supabase client invalid for setup."); return
    logger.info("Starting Supabase database setup/verification...")

    # 1. Enable Extensions
    await execute_sql_via_rpc(supabase, ENABLE_UUID_OSSP_SQL)

    # 2. Create Trigger Function
    await execute_sql_via_rpc(supabase, CREATE_UPDATE_TIMESTAMP_FUNCTION_SQL)

    # 3. Create Tables, Triggers, Indexes
    # Define order based on foreign key dependencies (e.g., contacts before call_logs)
    table_creation_order = [
        config.SUPABASE_CONTACTS_TABLE,
        config.SUPABASE_RESOURCES_TABLE,
        "agent_templates", # New table
        config.SUPABASE_CALL_LOGS_TABLE,
        "client_agent_instances" # New table
    ]

    for table_name in table_creation_order:
        if table_name not in TABLE_DEFINITIONS:
            logger.error(f"Table '{table_name}' is in creation order but not defined in TABLE_DEFINITIONS.")
            continue
        
        columns = TABLE_DEFINITIONS[table_name]
        try:
            table_already_exists = await check_table_exists(supabase, table_name)
            if not table_already_exists:
                await create_table(supabase, table_name, columns)
                # Apply trigger only after successful table creation attempt
                if "updated_at" in [c["name"] for c in columns]:
                    trigger_name = f"set_timestamp_on_{table_name}_update"
                    trigger_sql = f"""
                    CREATE OR REPLACE TRIGGER "{trigger_name}"
                    BEFORE UPDATE ON public."{table_name}"
                    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
                    """
                    await apply_trigger(supabase, table_name, trigger_name, trigger_sql)
                # Apply indexes
                for col in columns:
                    if col.get("index"): await create_index(supabase, table_name, col['name'])
            else:
                 logger.info(f"Table '{table_name}' already exists. Skipping creation, ensuring triggers/indexes.")
                 # Ensure trigger and indexes exist even if table exists
                 if "updated_at" in [c["name"] for c in columns]:
                    trigger_name = f"set_timestamp_on_{table_name}_update"
                    trigger_sql = f"""
                    CREATE OR REPLACE TRIGGER "{trigger_name}"
                    BEFORE UPDATE ON public."{table_name}"
                    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
                    """
                    await apply_trigger(supabase, table_name, trigger_name, trigger_sql)
                 for col in columns:
                    if col.get("index"): await create_index(supabase, table_name, col['name'])

        except Exception as table_setup_err:
            logger.error(f"Error during setup for table '{table_name}': {table_setup_err}", exc_info=True)

    logger.info("Supabase database setup/verification process complete.")
    logger.warning("Review logs for any SQL execution warnings requiring manual intervention (e.g., if RPC functions are missing).")
    logger.warning("Ensure Row Level Security (RLS) is enabled and appropriate policies are created in Supabase Studio for production security.")

# --- Test Function ---
async def main_db_setup_test():
    if not config.SUPABASE_ENABLED: print("Supabase disabled. Skipping DB setup test."); return
    print("Attempting Supabase connection and DB setup...")
    supabase_client = create_client()
    if supabase_client:
        await setup_supabase_tables(supabase_client)
        print("Database setup attempt finished. Check logs and Supabase dashboard.")
    else: print("Failed to create Supabase client for test.")

if __name__ == "__main__":
    # import asyncio
    # from dotenv import load_dotenv
    # load_dotenv()
    # print("Running database setup directly...")
    # asyncio.run(main_db_setup_test())
    print("Database setup script defined. Call setup_supabase_tables(supabase_client) during app startup.")
    print("Ensure the 'execute_sql' and 'check_table_exists' RPC functions exist in Supabase for programmatic DDL.")
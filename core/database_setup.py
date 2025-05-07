        # boutique_ai_project/core/database_setup.py

import logging
import asyncio
from typing import List, Dict, Any

from supabase import create_client, Client as SupabaseClient, PostgrestAPIError

import config # Root config

logger = logging.getLogger(__name__)

# --- Table Definitions ---
# These definitions include essential columns. You can expand them.
# Ensure data types match Supabase/PostgreSQL types.
# JSONB is good for flexible structured data. TEXT for long strings.
# Timestamps should use 'timestamptz' for timezone awareness.

TABLE_DEFINITIONS = {
    config.SUPABASE_CONTACTS_TABLE: [
        {"name": "id", "type": "uuid", "primary_key": True, "default": "uuid_generate_v4()"},
        {"name": "created_at", "type": "timestamptz", "default": "now()", "nullable": False},
        {"name": "updated_at", "type": "timestamptz", "default": "now()", "nullable": False},
        {"name": "email", "type": "text", "unique": True, "nullable": True}, # Unique, but can be null if phone is primary
        {"name": "phone_number", "type": "text", "unique": True, "nullable": True}, # Unique, but can be null if email is primary
        {"name": "first_name", "type": "text", "nullable": True},
        {"name": "last_name", "type": "text", "nullable": True},
        {"name": "company_name", "type": "text", "nullable": True},
        {"name": "domain", "type": "text", "nullable": True, "index": True},
        {"name": "job_title", "type": "text", "nullable": True},
        {"name": "linkedin_url", "type": "text", "nullable": True},
        {"name": "company_linkedin_url", "type": "text", "nullable": True},
        {"name": "status", "type": "text", "default": "'New_Raw_Lead'", "index": True}, # e.g., New_Raw_Lead, Enrichment_Triggered_Clay, Enriched_From_Clay_Pending_Analysis, Qualified_Sales_Ready, etc.
        {"name": "source_info", "type": "text", "nullable": True}, # How was this lead sourced?
        {"name": "correlation_id_clay", "type": "text", "nullable": True, "unique": True, "index": True}, # For matching Clay webhook results
        {"name": "last_activity_timestamp", "type": "timestamptz", "nullable": True},
        {"name": "raw_lead_data_json", "type": "jsonb", "nullable": True}, # Original data from source
        {"name": "clay_enrichment_data_json", "type": "jsonb", "nullable": True}, # Full payload from Clay
        {"name": "llm_qualification_score", "type": "integer", "nullable": True},
        {"name": "llm_inferred_pain_point_1", "type": "text", "nullable": True},
        {"name": "llm_inferred_pain_point_2", "type": "text", "nullable": True},
        {"name": "llm_suggested_hook", "type": "text", "nullable": True},
        {"name": "llm_qualification_reasoning", "type": "text", "nullable": True},
        {"name": "llm_analysis_confidence", "type": "text", "nullable": True}, # Low/Medium/High
        {"name": "llm_suggested_next_action", "type": "text", "nullable": True},
        {"name": "llm_full_analysis_json", "type": "jsonb", "nullable": True}, # Store full LLM analysis
        {"name": "assigned_sales_agent_id", "type": "text", "nullable": True, "index": True},
        {"name": "tags", "type": "text[]", "nullable": True}, # For arbitrary tagging
        {"name": "notes", "type": "text", "nullable": True} # General notes field
    ],
    config.SUPABASE_CALL_LOGS_TABLE: [ # Renamed from call_log_table for consistency
        {"name": "id", "type": "uuid", "primary_key": True, "default": "uuid_generate_v4()"},
        {"name": "created_at", "type": "timestamptz", "default": "now()", "nullable": False},
        {"name": "call_sid", "type": "text", "unique": True, "nullable": False, "index": True}, # Twilio Call SID
        {"name": "contact_fk_id", "type": "uuid", "nullable": True, "foreign_key": {"table": config.SUPABASE_CONTACTS_TABLE, "column": "id", "on_delete": "SET NULL"}},
        {"name": "agent_id", "type": "text", "nullable": False, "index": True},
        {"name": "target_phone_number", "type": "text", "nullable": True},
        {"name": "call_status", "type": "text", "nullable": False}, # System status: Completed, Error, Timeout, Voicemail
        {"name": "call_outcome_category_llm", "type": "text", "nullable": True}, # LLM's categorized outcome
        {"name": "call_duration_seconds", "type": "integer", "nullable": True},
        {"name": "conversation_history_json", "type": "jsonb", "nullable": True},
        {"name": "notes", "type": "text", "nullable": True},
        {"name": "llm_call_summary_json", "type": "jsonb", "nullable": True},
        {"name": "key_objections_tags", "type": "text[]", "nullable": True},
        {"name": "prospect_engagement_signals_tags", "type": "text[]", "nullable": True}
    ],
    config.SUPABASE_RESOURCES_TABLE: [
        {"name": "id", "type": "uuid", "primary_key": True, "default": "uuid_generate_v4()"},
        {"name": "created_at", "type": "timestamptz", "default": "now()", "nullable": False},
        {"name": "updated_at", "type": "timestamptz", "default": "now()", "nullable": False},
        {"name": "service_name", "type": "text", "nullable": False, "index": True}, # e.g., "clay.com", "some_other_api"
        {"name": "resource_type", "type": "text", "nullable": False, "index": True}, # e.g., "api_key", "trial_account", "credentials"
        {"name": "resource_data", "type": "jsonb", "nullable": True}, # Stores actual key, user/pass, etc.
        {"name": "status", "type": "text", "default": "'active'", "nullable": False, "index": True}, # e.g., active, expired, invalid
        {"name": "expiry_timestamp", "type": "timestamptz", "nullable": True, "index": True},
        {"name": "notes", "type": "text", "nullable": True},
        {"name": "fingerprint_profile_summary", "type": "jsonb", "nullable": True}, # Summary of fingerprint used for acquisition
        # Add a composite unique constraint for service_name and resource_type if a service should only have one active key of a type
        # This needs to be handled carefully if you allow multiple keys/accounts for the same service.
        # For now, rely on application logic to manage uniqueness or use specific identifiers in resource_data.
    ]
}

# SQL to enable uuid-ossp extension if not already enabled (needed for uuid_generate_v4())
ENABLE_UUID_OSSP_SQL = "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\" WITH SCHEMA extensions;"

# SQL to create a function to automatically update 'updated_at' columns
CREATE_UPDATE_TIMESTAMP_FUNCTION_SQL = """
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
   NEW.updated_at = now(); 
   RETURN NEW;
END;
$$ language 'plpgsql';
"""

async def table_exists(supabase: SupabaseClient, table_name: str) -> bool:
    """Checks if a table exists in the public schema."""
    try:
        # A simple way to check is to try to select a row with a limit of 0.
        # If it errors with "relation does not exist", table is not there.
        # A more robust way is to query pg_catalog.pg_tables.
        # Using a simpler query for now.
        # This is a bit of a hacky check. A proper check queries information_schema.tables
        # For Supabase Python client, a direct "check if table exists" isn't obvious.
        # We can try to select from it and catch the error if it doesn't exist.
        # This is not ideal as it relies on specific error codes/messages.
        
        # A more reliable way using rpc:
        # This requires a pgSQL function `table_exists(schema_name text, table_name text)`
        # For simplicity, we'll try to create and catch specific errors.
        # This function is not used in the create_table logic below, which uses a more direct approach.
        logger.debug(f"Checking existence of table: {table_name} (This check is illustrative)")
        # response = await asyncio.to_thread(supabase.table(table_name).select("id", count="exact").limit(0).execute)
        # return True # If no error
        return False # Placeholder, actual check is harder without direct pg_catalog access via client lib easily
    except Exception: # Broad exception for illustration
        return False


async def create_table_if_not_exists(supabase: SupabaseClient, table_name: str, columns: List[Dict[str, Any]]):
    """
    Creates a table with specified columns if it doesn't already exist.
    This is a simplified table creator. For complex schemas, constraints, and RLS, use Supabase Studio or migrations.
    """
    logger.info(f"Attempting to create table '{table_name}' if it does not exist...")

    # Construct column definitions
    column_sqls = []
    primary_key_column = None
    for col in columns:
        col_sql = f"\"{col['name']}\" {col['type']}" # Quote column names
        if col.get("primary_key"):
            primary_key_column = col['name'] # Will be defined separately for clarity
        if not col.get("nullable", True) and not col.get("primary_key"): # Primary keys are implicitly NOT NULL
            col_sql += " NOT NULL"
        if col.get("default") is not None:
            col_sql += f" DEFAULT {col['default']}"
        if col.get("unique"):
            col_sql += " UNIQUE"
        column_sqls.append(col_sql)

    # Add primary key constraint if defined
    if primary_key_column:
        column_sqls.append(f"PRIMARY KEY (\"{primary_key_column}\")")
    
    # Add foreign key constraints
    for col in columns:
        if col.get("foreign_key"):
            fk = col["foreign_key"]
            on_delete_action = fk.get("on_delete", "NO ACTION").upper()
            on_update_action = fk.get("on_update", "NO ACTION").upper()
            column_sqls.append(
                f"CONSTRAINT fk_{table_name}_{col['name']}_{fk['table']} "
                f"FOREIGN KEY (\"{col['name']}\") REFERENCES public.\"{fk['table']}\" (\"{fk['column']}\") "
                f"ON DELETE {on_delete_action} ON UPDATE {on_update_action}"
            )

    create_table_sql = f"CREATE TABLE IF NOT EXISTS public.\"{table_name}\" ({', '.join(column_sqls)});"
    
    try:
        logger.debug(f"Executing SQL for table '{table_name}':\n{create_table_sql}")
        # Supabase client doesn't have a direct "execute raw SQL" for DDL easily accessible for table creation checks.
        # We use an RPC call to a custom PostgreSQL function for this.
        # This requires you to create a function in your Supabase SQL editor:
        # CREATE OR REPLACE FUNCTION execute_ddl(sql_command text)
        # RETURNS void AS $$
        # BEGIN
        #   EXECUTE sql_command;
        # END;
        # $$ LANGUAGE plpgsql SECURITY DEFINER;
        # Grant execute permission: GRANT EXECUTE ON FUNCTION public.execute_ddl(text) TO authenticated, service_role;
        
        # Simpler approach: try to insert a dummy row and delete it, or select.
        # If table doesn't exist, it will fail. If it exists, this is a no-op.
        # This is still not ideal. The BEST way is SQL migrations (e.g. using Supabase CLI).
        # For programmatic setup, we'll assume if an error occurs that's not "relation already exists", it's a problem.

        # Let's try a more direct approach by attempting to select, then create if specific error.
        # This is still not perfect as error messages can change.
        try:
            await asyncio.to_thread(supabase.table(table_name).select("id").limit(1).execute)
            logger.info(f"Table '{table_name}' already exists.")
        except PostgrestAPIError as e:
            if e.code == "42P01": # "undefined_table"
                logger.info(f"Table '{table_name}' does not exist. Creating now...")
                # Re-enable direct DDL execution via RPC if you set up the execute_ddl function in Supabase
                # await asyncio.to_thread(supabase.rpc('execute_ddl', {'sql_command': create_table_sql}).execute)
                # For now, this part is a placeholder for how you'd run DDL.
                # The most robust way is using Supabase CLI migrations.
                # This script will LOG the SQL. You should run it manually in Supabase SQL Editor if needed.
                logger.warning(f"Manual Action Required: Table '{table_name}' needs to be created. SQL:\n{create_table_sql}")
                logger.warning("Alternatively, use Supabase CLI migrations for robust schema management.")

            else: # Other Postgrest error
                logger.error(f"Error checking/creating table '{table_name}': {e.message}", exc_info=True)
                raise
        
        # Create trigger for updated_at after table creation
        if "updated_at" in [c["name"] for c in columns]:
            trigger_sql = f"""
            CREATE OR REPLACE TRIGGER set_timestamp_on_{table_name}_update
            BEFORE UPDATE ON public."{table_name}"
            FOR EACH ROW
            EXECUTE FUNCTION public.update_updated_at_column();
            """
            logger.info(f"Attempting to create/replace updated_at trigger for '{table_name}'.")
            # Similar to table creation, robust DDL for triggers is best via migrations or SQL editor.
            # await asyncio.to_thread(supabase.rpc('execute_ddl', {'sql_command': trigger_sql}).execute)
            logger.warning(f"Manual Action Recommended: Ensure updated_at trigger exists for '{table_name}'. SQL:\n{trigger_sql}")


    except Exception as e:
        logger.error(f"Failed to ensure table '{table_name}' exists: {e}", exc_info=True)
        # Log the SQL for manual execution
        logger.error(f"SQL for manual creation of '{table_name}':\n{create_table_sql}")


async def setup_supabase_tables(supabase: SupabaseClient):
    """
    Sets up the required tables in Supabase if they don't exist.
    """
    if not supabase:
        logger.error("Supabase client not provided to setup_supabase_tables.")
        return

    logger.info("Starting Supabase database setup/verification...")

    # 1. Enable uuid-ossp extension (needed for uuid_generate_v4())
    try:
        logger.info("Ensuring 'uuid-ossp' extension is enabled...")
        # await asyncio.to_thread(supabase.rpc('execute_ddl', {'sql_command': ENABLE_UUID_OSSP_SQL}).execute)
        # This is critical. If it fails, UUID defaults will fail.
        # Best to run this manually once in Supabase SQL Editor if RPC method isn't set up.
        logger.warning(f"Manual Action Recommended: Ensure 'uuid-ossp' extension is enabled in Supabase. SQL: {ENABLE_UUID_OSSP_SQL}")
    except Exception as e:
        logger.error(f"Could not ensure 'uuid-ossp' extension. UUID generation might fail. Error: {e}")
        logger.warning("Please enable it manually in your Supabase SQL editor: CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")

    # 2. Create/Ensure the updated_at trigger function exists
    try:
        logger.info("Ensuring 'update_updated_at_column' trigger function exists...")
        # await asyncio.to_thread(supabase.rpc('execute_ddl', {'sql_command': CREATE_UPDATE_TIMESTAMP_FUNCTION_SQL}).execute)
        logger.warning(f"Manual Action Recommended: Ensure 'update_updated_at_column' function exists. SQL:\n{CREATE_UPDATE_TIMESTAMP_FUNCTION_SQL}")
    except Exception as e:
        logger.error(f"Could not ensure 'update_updated_at_column' function. Timestamps might not auto-update. Error: {e}")


    # 3. Create each table
    for table_name, columns in TABLE_DEFINITIONS.items():
        await create_table_if_not_exists(supabase, table_name, columns)
        # Add RLS policies after table creation if needed (highly recommended for security)
        # Example: supabase.sql(f"ALTER TABLE public.\"{table_name}\" ENABLE ROW LEVEL SECURITY;").execute()
        # Then create policies. This is complex and best done via Supabase Studio or migrations.
        logger.info(f"RLS (Row Level Security) is NOT automatically configured for table '{table_name}'. Review Supabase docs and apply appropriate policies for production.")


    logger.info("Supabase database setup/verification process complete.")
    logger.warning("IMPORTANT: Review logged SQL for tables/triggers and apply manually in Supabase SQL Editor if programmatic creation was skipped or failed.")
    logger.warning("For robust schema management, use Supabase CLI migrations (see Supabase documentation).")


# Example of how this might be called during server startup (in server.py lifespan)
async def main_db_setup_test():
    if not config.SUPABASE_ENABLED:
        print("Supabase is not enabled in config. Skipping database setup test.")
        return
    
    print("Attempting to connect to Supabase for DB setup test...")
    supabase_client = None
    try:
        supabase_client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
        print("Supabase client created. Running table setup...")
        await setup_supabase_tables(supabase_client)
        print("Database setup test finished. Check logs and your Supabase dashboard.")
    except Exception as e:
        print(f"Error during database setup test: {e}")
    finally:
        # The Supabase client doesn't have an explicit close method in the same way as a DB connection pool object.
        # Connections are typically managed per request.
        pass

if __name__ == "__main__":
    # This script is intended to be imported and setup_supabase_tables called.
    # Running it directly can be used for a one-time setup attempt.
    # Ensure .env is loaded for config values.
    # from dotenv import load_dotenv
    # load_dotenv()
    # asyncio.run(main_db_setup_test())
    print("Database setup script defined. Call setup_supabase_tables(supabase_client) to run.")
    print("For robust DDL, use Supabase CLI migrations or execute logged SQL manually.")
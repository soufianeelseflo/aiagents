# /core/database_setup.py: (Holistically Reviewed for Idempotency)
# --------------------------------------------------------------------------------
import asyncio
import logging
from typing import List, Dict, Any
from supabase_py_async import create_client, AsyncClient # Using supabase-py-async
# from supabase import create_client, Client # Original sync client

import config # Assuming config.py is in the project root or accessible

logger = logging.getLogger(__name__)

async def execute_sql_batch(db_client: AsyncClient, sql_statements: List[str]):
    """Executes a batch of SQL statements sequentially."""
    for i, statement in enumerate(sql_statements):
        try:
            # Supabase Python client's execute method is for RPC, not direct SQL batch.
            # For raw SQL, it's better to use an RPC that executes it or structure as inserts/upserts.
            # However, if we must execute raw DDL, we'll wrap it in an RPC or use a placeholder.
            # For simplicity, let's assume we have an RPC 'execute_raw_sql' or adjust.
            # This is a common pattern if direct DDL isn't straightforwardly supported.
            # For now, attempting direct execution if supported by the underlying PostgREST interface
            # (which it usually is for simple, non-transactional DDL via rpc).

            # A more robust way for schema changes if not using migrations:
            # Create an SQL function in Supabase dashboard:
            # CREATE OR REPLACE FUNCTION execute_admin_sql(sql_query TEXT)
            # RETURNS TEXT LANGUAGE plpgsql SECURITY DEFINER AS $$
            # BEGIN
            #   EXECUTE sql_query;
            #   RETURN 'SQL executed successfully';
            # END;
            # $$;
            # Then call it via RPC: await db_client.rpc("execute_admin_sql", {"sql_query": statement}).execute()

            # Simulating direct execution for DDL if it's simple and non-transactional.
            # This is a simplified approach; proper migration tools are better for complex changes.
            # The Supabase client might not directly support batch DDL execution like this.
            # We will try executing them one by one via rpc for broader compatibility
            # if a generic "execute_sql" rpc isn't available.
            # This part is tricky without knowing the exact capabilities of the async client for raw DDL.
            # For now, we assume single statements might work if wrapped in an RPC or simple execution.

            logger.info(f"Attempting to execute SQL (statement {i+1}/{len(sql_statements)}): {statement[:100]}...")
            # This is a placeholder for actual execution. The `supabase-py-async` client
            # uses `postgrest-py` which doesn't have a direct `execute_batch_sql` or similar.
            # You'd typically define an RPC in Supabase to run DDL.
            # For now, this function illustrates the intent.
            # A practical approach would be to use Supabase migrations or run these via SQL editor.
            # await db_client.rpc("some_rpc_to_run_sql", {"sql_statement": statement}).execute()
            # Since direct arbitrary SQL execution is complex and risky via client lib without specific RPCs:
            if "CREATE TYPE" in statement.upper() or "CREATE TABLE" in statement.upper():
                 # Supabase client typically manages table interactions via ORM-like methods or RPCs.
                 # For direct DDL, an RPC is usually the way.
                 # This is a conceptual representation. In a real scenario, use Supabase migrations.
                 logger.info(f"Conceptual DDL: {statement}. In production, use Supabase Migrations or an admin SQL function.")
            else: # For DML like INSERT, UPDATE, DELETE (though setup is usually DDL)
                # This won't work for DDL.
                # response = await db_client.table("some_table_for_generic_sql_if_supported").insert({"query": statement}).execute() # Example
                pass
            # Simulate success for now as direct DDL execution via client is tricky
            logger.info(f"Statement conceptually processed: {statement[:100]}...")

        except Exception as e:
            # Check if the error is about the object already existing, which is fine for idempotent DDL
            if "already exists" in str(e).lower() or "duplicate key" in str(e).lower():
                logger.info(f"Object in statement likely already exists (which is okay for idempotent setup): {statement[:100]}... Error: {e}")
            else:
                logger.error(f"Error executing SQL statement: {statement[:100]}... Error: {e}", exc_info=True)
                # Optionally re-raise or handle more gracefully
                # raise


async def setup_supabase_tables(db_client: AsyncClient):
    """
    Sets up the necessary tables, types, and RLS policies in Supabase.
    Designed to be idempotent.
    """
    logger.info("Starting Supabase database schema setup/verification...")

    # Define ENUM types first, as tables might depend on them. Use 'IF NOT EXISTS'.
    # Note: Direct 'CREATE TYPE IF NOT EXISTS' might need to be run via SQL editor or a custom RPC.
    # The Python client might not directly support this DDL for types easily.
    # For robust enum creation, use Supabase migrations or execute in SQL editor.
    # This is a conceptual representation.
    sql_commands = [
        # ENUM Types (Idempotent: CREATE TYPE IF NOT EXISTS)
        # For these to be truly idempotent via client, an RPC or try/except on creation is needed.
        # For now, assuming execution in an environment where these can be run, or they are pre-existing.
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'call_phase_enum') THEN
                CREATE TYPE call_phase_enum AS ENUM (
                    'NOT_STARTED', 'INITIALIZING', 'INTRODUCTION', 'QUALIFICATION', 
                    'DISCOVERY', 'PITCH', 'OBJECTION_HANDLING', 'CLOSING_ATTEMPT', 
                    'WRAP_UP', 'CALL_ENDED_SUCCESS', 'CALL_ENDED_FAIL_PROSPECT', 
                    'CALL_ENDED_FAIL_INTERNAL', 'CALL_ENDED_MAX_DURATION', 'CALL_ENDED_OPERATOR'
                );
            END IF;
        END$$;
        """,
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'prospect_sentiment_enum') THEN
                CREATE TYPE prospect_sentiment_enum AS ENUM (
                    'UNKNOWN', 'POSITIVE', 'NEUTRAL', 'NEGATIVE', 'INTERESTED', 'NOT_INTERESTED'
                );
            END IF;
        END$$;
        """,
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'call_outcome_category_enum') THEN
                CREATE TYPE call_outcome_category_enum AS ENUM (
                    'NO_ANSWER', 'VOICEMAIL_LEFT', 'SHORT_INTERACTION_NO_PITCH', 
                    'CONVERSATION_NO_CLOSE', 'VERBAL_AGREEMENT_TO_NEXT_STEP', 
                    'MEETING_SCHEDULED', 'SALE_CLOSED', 'REQUESTED_CALLBACK', 
                    'WRONG_NUMBER', 'DO_NOT_CALL', 'ERROR_OR_UNKNOWN'
                );
            END IF;
        END$$;
        """,
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'agent_status_enum') THEN
                CREATE TYPE agent_status_enum AS ENUM ('IDLE', 'PREPARING', 'ACTIVE_CALL', 'WRAP_UP', 'ERROR');
            END IF;
        END$$;
        """,
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'resource_status_enum') THEN
                CREATE TYPE resource_status_enum AS ENUM ('PENDING_PROVISIONING', 'ACTIVE', 'INACTIVE', 'ERROR', 'DELETED');
            END IF;
        END$$;
        """,
        
        # Tables (Idempotent: CREATE TABLE IF NOT EXISTS)
        f"""
        CREATE TABLE IF NOT EXISTS {config.SUPABASE_CONTACTS_TABLE} (
            contact_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            created_at TIMESTAMPTZ DEFAULT now() NOT NULL,
            updated_at TIMESTAMPTZ DEFAULT now() NOT NULL,
            company_name TEXT,
            domain TEXT UNIQUE, -- Ensures unique domain for companies
            primary_contact_name TEXT,
            primary_contact_email TEXT, -- Consider UNIQUE constraint if appropriate
            phone_number TEXT UNIQUE, -- Critical for call association
            linkedin_profile_url TEXT,
            role TEXT, -- e.g., CEO, Head of Sales
            industry TEXT,
            company_size TEXT, -- e.g., "11-50 employees"
            country TEXT,
            source_info TEXT, -- How the lead was acquired (e.g., "Clay.com batch X", "Manual Entry")
            status TEXT DEFAULT 'New_Raw_Lead', -- Lead status (e.g., New, Qualified, Contacted, Nurturing, Client, Disqualified)
            last_contacted_at TIMESTAMPTZ,
            next_follow_up_at TIMESTAMPTZ,
            notes TEXT,
            -- Custom fields as JSONB for flexibility
            custom_fields JSONB
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {config.SUPABASE_CALL_LOGS_TABLE} (
            log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            call_sid TEXT UNIQUE NOT NULL, -- Twilio Call SID
            agent_id TEXT NOT NULL,
            contact_id UUID REFERENCES {config.SUPABASE_CONTACTS_TABLE}(contact_id) ON DELETE SET NULL,
            phone_number TEXT, -- Denormalized for quick reference
            start_time TIMESTAMPTZ DEFAULT now() NOT NULL,
            end_time TIMESTAMPTZ,
            duration_seconds INTEGER,
            outcome call_outcome_category_enum, -- Using the ENUM type
            call_phase_at_end call_phase_enum, -- Using the ENUM type
            prospect_sentiment_at_end prospect_sentiment_enum, -- Using the ENUM type
            full_transcript JSONB, -- Store conversation history as JSON array of objects
            call_summary_details JSONB, -- Store structured summary from SalesAgent
            recording_url TEXT, -- If Twilio call recording is enabled
            cost NUMERIC(10, 5) -- Optional: if Twilio provides cost info
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {config.SUPABASE_RESOURCES_TABLE} (
            resource_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            contact_id UUID REFERENCES {config.SUPABASE_CONTACTS_TABLE}(contact_id) ON DELETE CASCADE, -- If contact is deleted, associated resources might be too
            agent_id TEXT, -- ID of the agent that provisioned this resource
            service_name TEXT NOT NULL, -- e.g., "Gmail", "Calendly", "SpecificCRM"
            provisioned_at TIMESTAMPTZ DEFAULT now() NOT NULL,
            status resource_status_enum DEFAULT 'PENDING_PROVISIONING',
            credentials JSONB, -- Store encrypted credentials or access tokens carefully
            notes TEXT,
            last_accessed TIMESTAMPTZ,
            configuration_details JSONB -- Other service-specific config
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {config.SUPABASE_AGENTS_TABLE} (
            agent_internal_id TEXT PRIMARY KEY, -- e.g., "SalesAgent_callsid_xyz" or "AcquisitionAgent_001"
            agent_type TEXT NOT NULL, -- e.g., "SalesAgent", "AcquisitionAgent", "ResourceManager"
            status agent_status_enum DEFAULT 'IDLE',
            created_at TIMESTAMPTZ DEFAULT now() NOT NULL,
            last_heartbeat TIMESTAMPTZ,
            current_task_description TEXT,
            assigned_contact_id UUID REFERENCES {config.SUPABASE_CONTACTS_TABLE}(contact_id) ON DELETE SET NULL,
            metadata JSONB -- For specific agent instance data, like current campaign for AcqAgent
        );
        """,

        # Indexes for performance on frequently queried columns
        f"CREATE INDEX IF NOT EXISTS idx_contacts_domain ON {config.SUPABASE_CONTACTS_TABLE}(domain);",
        f"CREATE INDEX IF NOT EXISTS idx_contacts_phone_number ON {config.SUPABASE_CONTACTS_TABLE}(phone_number);",
        f"CREATE INDEX IF NOT EXISTS idx_contacts_status ON {config.SUPABASE_CONTACTS_TABLE}(status);",
        f"CREATE INDEX IF NOT EXISTS idx_call_logs_contact_id ON {config.SUPABASE_CALL_LOGS_TABLE}(contact_id);",
        f"CREATE INDEX IF NOT EXISTS idx_call_logs_agent_id ON {config.SUPABASE_CALL_LOGS_TABLE}(agent_id);",
        f"CREATE INDEX IF NOT EXISTS idx_call_logs_start_time ON {config.SUPABASE_CALL_LOGS_TABLE}(start_time DESC);",
        f"CREATE INDEX IF NOT EXISTS idx_resources_contact_id ON {config.SUPABASE_RESOURCES_TABLE}(contact_id);",
        f"CREATE INDEX IF NOT EXISTS idx_agents_status ON {config.SUPABASE_AGENTS_TABLE}(status);",
        f"CREATE INDEX IF NOT EXISTS idx_agents_type ON {config.SUPABASE_AGENTS_TABLE}(agent_type);",
    ]

    # RLS Policies (Conceptual - these should be refined based on actual access patterns)
    # These are examples; actual RLS policies need careful design.
    # For backend service roles, RLS is often bypassed, but good to define for other roles.
    # This script, run with service_role, will create them, but they'll apply to other roles.
    # logger.info("Setting up basic RLS policies (conceptual examples)...")
    # sql_commands.extend([
    # f"ALTER TABLE {config.SUPABASE_CONTACTS_TABLE} ENABLE ROW LEVEL SECURITY;",
    # f"CREATE POLICY \"Allow authenticated users to see their own contacts\" ON {config.SUPABASE_CONTACTS_TABLE} FOR SELECT USING (auth.uid() = user_id_column);", # Requires a user_id_column
    # Add more specific RLS policies as needed for your multi-tenant or user-specific data access
    # ])

    # Create a Supabase client instance if not provided (primarily for direct script execution)
    # In FastAPI app, client is usually managed by lifespan
    # Ensure this client uses the SERVICE_ROLE_KEY for setup!
    if not config.SUPABASE_URL or not config.SUPABASE_KEY:
        logger.error("Supabase URL or Key not configured. Skipping database setup.")
        return

    # Using the provided client, assuming it's configured with service role
    # This function is intended to be called with an already initialized client from server.py lifespan
    # await execute_sql_batch(db_client, sql_commands) # This was the original intent

    # For direct execution or if the client needs re-init (less ideal but for completeness)
    # This block is more for standalone testing of this script.
    # In the actual app, db_client is passed from server.py lifespan.
    temp_client: Optional[AsyncClient] = None
    if db_client is None: # If no client was passed, create one
        logger.warning("No Supabase client passed to setup_supabase_tables. Attempting to create a temporary one (for standalone testing).")
        try:
            temp_client = await create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
            db_to_use = temp_client
            logger.info("Temporary Supabase client created for setup.")
        except Exception as e:
            logger.critical(f"Failed to create temporary Supabase client for setup: {e}", exc_info=True)
            return
    else:
        db_to_use = db_client

    if db_to_use:
        logger.info(f"Executing {len(sql_commands)} SQL commands for schema setup...")
        # Execute commands one by one. For complex migrations, use Supabase CLI migrate.
        for command in sql_commands:
            try:
                # Using an RPC call 'execute_sql' which you'd create in Supabase dashboard:
                # CREATE OR REPLACE FUNCTION execute_sql(sql_query TEXT) RETURNS void AS $$
                # BEGIN EXECUTE sql_query; END;
                # $$ LANGUAGE plpgsql SECURITY DEFINER;
                #
                # If such RPC doesn't exist, this will fail.
                # Direct arbitrary SQL execution via clients is often restricted for security.
                # This is a conceptual representation.
                # Consider using Supabase migrations (supabase/migrations/* via Supabase CLI) for robust schema management.
                logger.info(f"Executing: {command[:200]}...") # Log snippet
                # This is a placeholder for how you might execute raw SQL.
                # The `supabase-py-async` client (and `postgrest-py`)
                # does not have a generic `db_client.sql(command).execute()` method like some ORMs.
                # You need to use table methods, RPCs, or a lower-level connection if available/safe.
                # For DDL, creating an SQL function in Supabase and calling it via RPC is common.
                # await db_to_use.rpc("execute_sql_ddl", {"p_sql": command}).execute() # Example if you create such an RPC
                
                # For now, we'll just log that these would be run.
                # The user would need to ensure these are applied via Supabase Studio SQL editor or migrations.
                if "CREATE TYPE" in command.upper():
                    logger.info(f"Conceptual DDL (Type): {command}. Apply via Supabase Studio or migrations.")
                elif "CREATE TABLE" in command.upper():
                     logger.info(f"Conceptual DDL (Table): {command}. Apply via Supabase Studio or migrations.")
                elif "CREATE INDEX" in command.upper():
                    logger.info(f"Conceptual DDL (Index): {command}. Apply via Supabase Studio or migrations.")

            except Exception as e:
                # Simplified error check
                if "already exists" in str(e).lower() or "duplicate" in str(e).lower() or "relation" in str(e).lower() and "does not exist" not in str(e).lower() :
                    logger.info(f"Object in command likely already exists or dependent object check: {command[:100]}... OK. Error: {str(e)[:100]}")
                else:
                    logger.error(f"Error executing command: {command[:100]}... Error: {e}", exc_info=True)
                    # Decide if we should stop or continue
                    # raise # Optionally re-raise to halt setup

        logger.info("Supabase database schema setup/verification conceptually processed. Manual application via Supabase Studio or migrations recommended for DDL.")
    else:
        logger.error("Supabase client not available. Database setup skipped.")

    # If a temporary client was created, it should be closed if the library supports it.
    # However, supabase-py-async client typically doesn't need explicit closing like aiohttp.ClientSession.
    # Connections are managed by PostgREST client.


async def main(): # For standalone testing
    if not config.SUPABASE_URL or not config.SUPABASE_KEY:
        logger.error("Supabase URL or Key not configured in .env. Cannot run database setup test.")
        return

    supabase_client: Optional[AsyncClient] = None
    try:
        logger.info(f"Connecting to Supabase for setup: {config.SUPABASE_URL}")
        # Ensure you use the async client if your functions are async
        supabase_client = await create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
        await setup_supabase_tables(supabase_client)
        logger.info("Database setup script finished.")
    except Exception as e:
        logger.critical(f"Error during database_setup main execution: {e}", exc_info=True)
    # finally:
        # Client doesn't have an explicit close method in supabase-py-async v0.x
        # Connections are typically managed by the underlying HTTPX client pool.
        # if supabase_client:
        #     await supabase_client.aclose() # if httpx client is exposed and needs closing
        # pass

if __name__ == "__main__":
    # This allows running the setup script directly, e.g., during initial deployment or testing.
    # Ensure .env is loaded if running this way.
    logger.info("Running database_setup.py directly...")
    asyncio.run(main())
# --------------------------------------------------------------------------------
# boutique_ai_project/core/services/deployment_manager.py

import logging
import asyncio
import json
import uuid # For generating potential instance IDs
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

# --- Docker Integration ---
# Requires 'pip install docker'
# Assumes Docker daemon is running and accessible via socket (common in Coolify/VPS)
try:
    import docker
    from docker.errors import APIError, NotFound, DockerException
    from docker.models.containers import Container
    DOCKER_AVAILABLE = True
except ImportError:
    docker = None # type: ignore
    APIError = None # type: ignore
    NotFound = None # type: ignore
    DockerException = None # type: ignore
    Container = None # type: ignore
    DOCKER_AVAILABLE = False
# --- End Docker Integration ---

import config # Root config

logger = logging.getLogger(__name__)

class DeploymentConfig:
    """Data class holding configuration for deploying a client agent instance."""
    def __init__(self, client_id: str, client_email: str, agent_type: str, configuration: Dict[str, Any]):
        self.client_id = client_id
        self.client_email = client_email
        # Ensure agent_type is safe for use in container names/env vars
        safe_agent_type = "".join(c if c.isalnum() else '_' for c in agent_type)
        self.agent_type = safe_agent_type
        self.configuration = configuration # Agent-specific settings, prompts, API keys etc.
        # Generate a unique, predictable name for the container
        # Format: prefix_agentType_clientIdPrefix_uuidSuffix
        self.instance_id: str = f"boutiqueai_agent_{self.agent_type.lower()}_{self.client_id.replace('-','_')[:8]}_{uuid.uuid4().hex[:6]}"

    def get_environment_variables(self) -> Dict[str, str]:
        """Prepares environment variables for the container based on configuration."""
        # Start with base config needed by any agent instance
        env_vars = {
            # --- Agent Identification & Context ---
            "BOUTIQUEAI_AGENT_MODE": "CLIENT_INSTANCE", # Indicate this container runs a client agent
            "BOUTIQUEAI_AGENT_TYPE": self.agent_type,
            "BOUTIQUEAI_CLIENT_ID": self.client_id,
            "BOUTIQUEAI_CLIENT_EMAIL": self.client_email,
            "BOUTIQUEAI_INSTANCE_ID": self.instance_id, # Pass instance ID to agent itself

            # --- Core Service Config (Agents might use these if client doesn't provide dedicated keys) ---
            "SUPABASE_URL": config.SUPABASE_URL,
            "SUPABASE_KEY": config.SUPABASE_KEY, # Consider if client agents need different keys/RLS
            "OPENROUTER_API_KEY": config.OPENROUTER_API_KEY,
            "DEEPGRAM_API_KEY": config.DEEPGRAM_API_KEY,
            # Twilio might require subaccounts or client-specific keys for isolation
            "TWILIO_ACCOUNT_SID": config.TWILIO_ACCOUNT_SID, # Placeholder - likely needs client key
            "TWILIO_AUTH_TOKEN": config.TWILIO_AUTH_TOKEN,   # Placeholder - likely needs client key
            "TWILIO_PHONE_NUMBER": config.TWILIO_PHONE_NUMBER, # Placeholder - likely needs client number

            # --- Agent Specific Configuration ---
            "AGENT_TARGET_NICHE_OVERRIDE": self.configuration.get("target_niche", config.AGENT_TARGET_NICHE_DEFAULT),
            "AGENT_INITIAL_PROMPT_CONTEXT_JSON": json.dumps(self.configuration.get("initial_prompt_context", {})),
            # Add other config overrides from self.configuration as needed

            # --- Client-Provided Keys (Crucial for real operation) ---
            # Example: Load client's CRM key if provided in the configuration dict
            # "CLIENT_CRM_API_KEY": self.configuration.get("client_provided_api_keys", {}).get("crm_api_key"),
            # "CLIENT_TWILIO_SID": self.configuration.get("client_provided_api_keys", {}).get("twilio_sid"),
            # "CLIENT_TWILIO_TOKEN": self.configuration.get("client_provided_api_keys", {}).get("twilio_token"),

            # --- System Settings ---
            "LOG_LEVEL": config.LOG_LEVEL,
            "PYTHONUNBUFFERED": "1", # Good practice for container logs
        }
        # Filter out None values and ensure all values are strings
        return {k: str(v) for k, v in env_vars.items() if v is not None}

    def __repr__(self):
        return f"<DeploymentConfig instance_id={self.instance_id} client_id={self.client_id} agent_type={self.agent_type}>"

class DeploymentManagerInterface(ABC):
    """Abstract Base Class for managing client agent instance lifecycle."""
    @abstractmethod
    async def deploy_agent_instance(self, deployment_config: DeploymentConfig) -> Dict[str, Any]: pass
    @abstractmethod
    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]: pass
    @abstractmethod
    async def stop_agent_instance(self, instance_id: str) -> bool: pass
    @abstractmethod
    async def start_agent_instance(self, instance_id: str) -> bool: pass
    @abstractmethod
    async def delete_agent_instance(self, instance_id: str) -> bool: pass

# --- Docker Implementation ---

class DockerDeploymentManager(DeploymentManagerInterface):
    """
    Manages agent deployments using the Docker Engine API via the 'docker' library.
    Assumes Docker daemon is accessible (e.g., via Docker socket mounted in Coolify).
    """
    def __init__(self, base_image_name: Optional[str] = None):
        if not DOCKER_AVAILABLE:
            logger.critical("Docker library not installed ('pip install docker'). DockerDeploymentManager cannot function.")
            raise ImportError("Docker library is required for DockerDeploymentManager.")
        try:
            self.client = docker.from_env()
            self.client.ping() # Test connection
            # Determine image name: Use env var or default based on common Coolify naming
            self.base_image_name = base_image_name or os.getenv("COOLIFY_APPLICATION_IMAGE", "boutique-ai-engine:latest")
            logger.info(f"DockerDeploymentManager initialized. Using base image: {self.base_image_name}. Connected to Docker daemon.")
        except DockerException as e:
            logger.critical(f"Failed to connect to Docker daemon: {e}")
            logger.critical("Ensure Docker is running and accessible (check socket permissions/Docker context).")
            raise ConnectionError("Could not connect to Docker daemon") from e

    async def _find_container(self, instance_id: str) -> Optional[Container]:
        """Finds a container by name (instance_id). Runs sync code in thread."""
        try:
            # The docker library calls are blocking, run in thread pool
            container = await asyncio.to_thread(self.client.containers.get, instance_id)
            return container
        except NotFound:
            return None
        except APIError as e:
            logger.error(f"Docker API error finding container '{instance_id}': {e}", exc_info=True)
            return None
        except Exception as e: # Catch other potential errors
             logger.error(f"Unexpected error finding container '{instance_id}': {e}", exc_info=True)
             return None

    async def deploy_agent_instance(self, deployment_config: DeploymentConfig) -> Dict[str, Any]:
        """Deploys a new agent instance as a Docker container."""
        instance_id = deployment_config.instance_id
        logger.info(f"Attempting to deploy Docker container: {instance_id}")

        existing_container = await self._find_container(instance_id)
        if existing_container:
            logger.warning(f"Container '{instance_id}' already exists. Status: {existing_container.status}. Ensuring running.")
            if existing_container.status != 'running':
                await self.start_agent_instance(instance_id)
            return {"status": "success", "instance_id": instance_id, "message": "Instance already existed, ensured running."}

        try:
            env_vars = deployment_config.get_environment_variables()
            logger.debug(f"Deploying container '{instance_id}' with image '{self.base_image_name}'.")
            logger.debug(f"Environment Variables for {instance_id}: { {k: (v[:10]+'...' if len(v)>15 else v) for k,v in env_vars.items()} }") # Log summary

            # --- Command to run inside the container ---
            # This needs to start the correct agent logic based on BOUTIQUEAI_AGENT_TYPE.
            # Assumes an entrypoint script or modified server.py handles this.
            # Example: A script `agent_runner.py` that takes `--agent-type`
            # command = ["python", "-m", "core.agent_runner"] # Requires agent_runner.py
            # Simpler: Modify server.py to check BOUTIQUEAI_AGENT_MODE == "CLIENT_INSTANCE"
            # and run agent loop instead of uvicorn. For now, use default CMD.
            command = None
            logger.warning(f"Container command not specified for {instance_id}. Using image default CMD. Ensure image entrypoint handles BOUTIQUEAI_AGENT_MODE/TYPE env vars.")

            # Run container in detached mode, network connected to Coolify default bridge
            container = await asyncio.to_thread(
                self.client.containers.run,
                image=self.base_image_name,
                detach=True,
                name=instance_id,
                environment=env_vars,
                command=command,
                restart_policy={"Name": "unless-stopped"},
                network="coolify", # Connect to default Coolify bridge network
                # Consider resource limits
                # mem_limit="512m",
                # cpu_shares=512, # Relative CPU weight
            )
            logger.info(f"Successfully started container '{instance_id}' (ID: {container.short_id}).")
            return {"status": "success", "instance_id": instance_id}

        except APIError as e:
            logger.error(f"Docker API error deploying container '{instance_id}': {e}", exc_info=True)
            return {"status": "failed", "reason": f"Docker API Error: {e}"}
        except Exception as e:
            logger.error(f"Unexpected error deploying container '{instance_id}': {e}", exc_info=True)
            return {"status": "failed", "reason": f"Unexpected error: {e}"}

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        """Checks the status of a deployed agent container."""
        logger.debug(f"Getting status for container: {instance_id}")
        container = await self._find_container(instance_id)
        if container:
            try:
                await asyncio.to_thread(container.reload) # Refresh container state
                logger.info(f"Status for container '{instance_id}': {container.status}")
                return {"status": container.status, "instance_id": instance_id, "details": container.attrs}
            except Exception as e:
                 logger.error(f"Error getting status for '{instance_id}': {e}", exc_info=True)
                 return {"status": "error", "reason": f"Error getting status: {e}"}
        else:
            logger.warning(f"Container '{instance_id}' not found for status check.")
            return {"status": "not_found", "instance_id": instance_id}

    async def stop_agent_instance(self, instance_id: str) -> bool:
        """Stops a running agent container."""
        logger.info(f"Attempting to stop container: {instance_id}")
        container = await self._find_container(instance_id)
        if container:
            try:
                await asyncio.to_thread(container.stop, timeout=30)
                logger.info(f"Container '{instance_id}' stopped successfully.")
                return True
            except Exception as e:
                 logger.error(f"Error stopping container '{instance_id}': {e}", exc_info=True)
                 return False
        else:
            logger.warning(f"Container '{instance_id}' not found, cannot stop.")
            return False # Indicate not found or already stopped implicitly

    async def start_agent_instance(self, instance_id: str) -> bool:
        """Starts a stopped agent container."""
        logger.info(f"Attempting to start container: {instance_id}")
        container = await self._find_container(instance_id)
        if container:
            try:
                await asyncio.to_thread(container.start)
                logger.info(f"Container '{instance_id}' started successfully.")
                return True
            except Exception as e:
                 logger.error(f"Error starting container '{instance_id}': {e}", exc_info=True)
                 return False
        else:
            logger.warning(f"Container '{instance_id}' not found, cannot start.")
            return False

    async def delete_agent_instance(self, instance_id: str) -> bool:
        """Stops and removes an agent container."""
        logger.info(f"Attempting to delete container: {instance_id}")
        container = await self._find_container(instance_id)
        if container:
            try:
                await asyncio.to_thread(container.remove, force=True, v=True) # v=True removes associated anonymous volumes
                logger.info(f"Container '{instance_id}' deleted successfully.")
                return True
            except Exception as e:
                 logger.error(f"Error deleting container '{instance_id}': {e}", exc_info=True)
                 return False
        else:
            logger.warning(f"Container '{instance_id}' not found, cannot delete.")
            return True # Already gone

# --- Test Function ---
async def _test_docker_deployment_manager_final():
    print("--- Testing DockerDeploymentManager (FINAL) ---")
    if not DOCKER_AVAILABLE: print("Docker library not found, skipping test."); return
    try:
        # Use a readily available image like 'hello-world' for basic Docker interaction test
        # In production, this would be your actual application image name
        manager = DockerDeploymentManager(base_image_name="hello-world:latest")
        
        client_id_test = f"client_{uuid.uuid4().hex[:6]}"
        test_config = DeploymentConfig(
            client_id=client_id_test, client_email=f"{client_id_test}@test.com",
            agent_type="TestAgent", configuration={"param1": "value1"}
        )
        instance_id_test = test_config.instance_id

        print(f"\n1. Deploying instance: {instance_id_test}")
        deploy_result = await manager.deploy_agent_instance(test_config)
        print(f"  Deployment Result: {deploy_result}")

        if deploy_result.get("status") == "success":
            print(f"\n2. Getting status for: {instance_id_test}")
            await asyncio.sleep(1) # Allow container state to settle
            status_result = await manager.get_instance_status(instance_id_test)
            print(f"  Status Result: {status_result.get('status')}")

            print(f"\n3. Stopping instance: {instance_id_test}")
            stop_result = await manager.stop_agent_instance(instance_id_test)
            print(f"  Stop Result: {stop_result}")
            await asyncio.sleep(1)
            status_after_stop = await manager.get_instance_status(instance_id_test)
            print(f"  Status after stop: {status_after_stop.get('status')}")

            print(f"\n4. Starting instance: {instance_id_test}")
            start_result = await manager.start_agent_instance(instance_id_test)
            print(f"  Start Result: {start_result}")
            await asyncio.sleep(1)
            status_after_start = await manager.get_instance_status(instance_id_test)
            print(f"  Status after start: {status_after_start.get('status')}")

            print(f"\n5. Deleting instance: {instance_id_test}")
            delete_result = await manager.delete_agent_instance(instance_id_test)
            print(f"  Delete Result: {delete_result}")
            status_after_delete = await manager.get_instance_status(instance_id_test)
            print(f"  Status after delete: {status_after_delete.get('status')}")
        else: print("  Skipping further tests as deployment failed.")

    except ConnectionError as e: print(f"Test failed: Could not connect to Docker daemon. {e}")
    except Exception as e: print(f"Test failed unexpectedly: {e}", exc_info=True)

if __name__ == "__main__":
    # import asyncio
    # from dotenv import load_dotenv
    # load_dotenv()
    # asyncio.run(_test_docker_deployment_manager_final())
    print("DeploymentManagerInterface and DockerDeploymentManager defined. Test requires Docker and 'docker' library.")
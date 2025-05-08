# /core/services/deployment_manager.py:
# --------------------------------------------------------------------------------
# boutique_ai_project/core/services/deployment_manager.py

import logging
import asyncio
import json
import uuid # For generating potential instance IDs
import os # Added os import
from abc import ABC, abstractmethod
# *** ADDED typing imports ***
from typing import Dict, Any, Optional, List, Union

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
    logging.getLogger(__name__).warning("Docker library not found. DockerDeploymentManager will be unavailable.")
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
        # Ensure client_id is treated as string for slicing
        client_id_str = str(client_id)
        self.instance_id: str = f"boutiqueai_agent_{self.agent_type.lower()}_{client_id_str.replace('-','_')[:8]}_{uuid.uuid4().hex[:6]}"

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
            # Ensure config values are accessed correctly and exist
            "SUPABASE_URL": config.SUPABASE_URL if config.SUPABASE_URL else "",
            "SUPABASE_KEY": config.SUPABASE_KEY if config.SUPABASE_KEY else "", # Consider if client agents need different keys/RLS
            "OPENROUTER_API_KEY": config.OPENROUTER_API_KEY if config.OPENROUTER_API_KEY else "",
            "DEEPGRAM_API_KEY": config.DEEPGRAM_API_KEY if config.DEEPGRAM_API_KEY else "",
            # Twilio might require subaccounts or client-specific keys for isolation
            "TWILIO_ACCOUNT_SID": config.TWILIO_ACCOUNT_SID if config.TWILIO_ACCOUNT_SID else "", # Placeholder - likely needs client key
            "TWILIO_AUTH_TOKEN": config.TWILIO_AUTH_TOKEN if config.TWILIO_AUTH_TOKEN else "",   # Placeholder - likely needs client key
            "TWILIO_PHONE_NUMBER": config.TWILIO_PHONE_NUMBER if config.TWILIO_PHONE_NUMBER else "", # Placeholder - likely needs client number

            # --- Agent Specific Configuration ---
            # Ensure get() has a default or check existence
            "AGENT_TARGET_NICHE_OVERRIDE": self.configuration.get("target_niche", config.AGENT_TARGET_NICHE_DEFAULT),
            # Ensure JSON dump handles potential errors, default to empty dict string
            "AGENT_INITIAL_PROMPT_CONTEXT_JSON": json.dumps(self.configuration.get("initial_prompt_context", {}), default=str),

            # --- Client-Provided Keys (Crucial for real operation) ---
            # Example: Load client's CRM key if provided in the configuration dict
            # Ensure nested gets handle missing keys gracefully
            "CLIENT_CRM_API_KEY": self.configuration.get("client_provided_api_keys", {}).get("crm_api_key"),
            "CLIENT_TWILIO_SID": self.configuration.get("client_provided_api_keys", {}).get("twilio_sid"),
            "CLIENT_TWILIO_TOKEN": self.configuration.get("client_provided_api_keys", {}).get("twilio_token"),

            # --- System Settings ---
            "LOG_LEVEL": config.LOG_LEVEL,
            "PYTHONUNBUFFERED": "1", # Good practice for container logs
        }
        # Filter out None values and ensure all values are strings
        return {k: str(v) for k, v in env_vars.items() if v is not None and v != ""} # Filter empty strings too

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
            msg = "Docker library not installed ('pip install docker'). DockerDeploymentManager cannot function."
            logger.critical(msg)
            raise ImportError(msg) # Raise ImportError if Docker is essential
        try:
            self.client = docker.from_env()
            self.client.ping() # Test connection
            # Determine image name: Use env var or default based on common Coolify naming
            self.base_image_name = base_image_name or os.getenv("COOLIFY_APPLICATION_IMAGE", "boutique-ai-engine:latest") # Default to a sensible name
            logger.info(f"DockerDeploymentManager initialized. Using base image: {self.base_image_name}. Connected to Docker daemon.")
        except DockerException as e:
            logger.critical(f"Failed to connect to Docker daemon: {e}")
            logger.critical("Ensure Docker is running and accessible (check socket permissions/Docker context).")
            raise ConnectionError("Could not connect to Docker daemon") from e

    async def _find_container(self, instance_id: str) -> Optional[Container]:
        """Finds a container by name (instance_id). Runs sync code in thread."""
        if not DOCKER_AVAILABLE: return None
        try:
            # The docker library calls are blocking, run in thread pool
            container = await asyncio.to_thread(self.client.containers.get, instance_id)
            return container
        except NotFound:
            logger.debug(f"Container '{instance_id}' not found.")
            return None
        except APIError as e:
            # Log specific Docker API errors
            logger.error(f"Docker API error finding container '{instance_id}': Status {e.status_code} - {e.explanation}", exc_info=False)
            return None
        except Exception as e: # Catch other potential errors
             logger.error(f"Unexpected error finding container '{instance_id}': {e}", exc_info=True)
             return None

    async def deploy_agent_instance(self, deployment_config: DeploymentConfig) -> Dict[str, Any]:
        """Deploys a new agent instance as a Docker container."""
        if not DOCKER_AVAILABLE: return {"status": "failed", "reason": "Docker library not available"}
        instance_id = deployment_config.instance_id
        logger.info(f"Attempting to deploy Docker container: {instance_id}")

        existing_container = await self._find_container(instance_id)
        if existing_container:
            logger.warning(f"Container '{instance_id}' already exists. Status: {existing_container.status}. Ensuring running.")
            if existing_container.status != 'running':
                if await self.start_agent_instance(instance_id):
                     return {"status": "success", "instance_id": instance_id, "message": "Instance already existed, started successfully."}
                else:
                     return {"status": "failed", "instance_id": instance_id, "reason": "Instance already existed but failed to start."}
            return {"status": "success", "instance_id": instance_id, "message": "Instance already existed and running."}

        try:
            env_vars = deployment_config.get_environment_variables()
            logger.debug(f"Deploying container '{instance_id}' with image '{self.base_image_name}'.")
            # Log env vars carefully, masking sensitive ones might be needed in prod
            loggable_env = {k: (v[:4]+"..." if any(s in k for s in ['KEY', 'TOKEN', 'SECRET', 'PASSWORD']) and len(v)>4 else v) for k,v in env_vars.items()}
            logger.debug(f"Environment Variables for {instance_id} (Summary): {loggable_env}")

            # --- Command to run inside the container ---
            # Assumption: The main image's CMD/ENTRYPOINT handles the BOUTIQUEAI_AGENT_MODE.
            command = None # Use image default CMD/ENTRYPOINT
            logger.info(f"Container command not specified for {instance_id}. Using image default CMD/ENTRYPOINT.")

            # Ensure network exists or handle error gracefully
            try:
                 await asyncio.to_thread(self.client.networks.get, "coolify")
            except NotFound:
                 logger.error("Docker network 'coolify' not found. Cannot attach container. Please ensure Coolify network setup is correct.")
                 return {"status": "failed", "reason": "Docker network 'coolify' not found."}

            container = await asyncio.to_thread(
                self.client.containers.run,
                image=self.base_image_name,
                detach=True,
                name=instance_id,
                environment=env_vars,
                command=command,
                restart_policy={"Name": "unless-stopped"},
                network="coolify", # Attach to default Coolify bridge network
                # Consider resource limits
                # mem_limit="512m",
                # cpu_shares=512, # Relative CPU weight
                # labels={"created_by": "boutique-ai-provisioning-agent"} # Add labels for tracking
            )
            logger.info(f"Successfully started container '{instance_id}' (ID: {container.short_id}).")
            # Optionally, wait a moment and check container status
            await asyncio.sleep(1)
            status_check = await self.get_instance_status(instance_id)
            if "running" in status_check.get("status", "").lower():
                return {"status": "success", "instance_id": instance_id, "message": "Instance deployed and running."}
            else:
                logger.warning(f"Container {instance_id} started but status is '{status_check.get('status')}'. Check container logs.")
                return {"status": "success", "instance_id": instance_id, "message": f"Instance deployed, but current status is {status_check.get('status')}. Check logs."}


        except APIError as e:
            logger.error(f"Docker API error deploying container '{instance_id}': Status {e.status_code} - {e.explanation}", exc_info=False)
            return {"status": "failed", "reason": f"Docker API Error: {e.explanation}"}
        except Exception as e:
            logger.error(f"Unexpected error deploying container '{instance_id}': {e}", exc_info=True)
            return {"status": "failed", "reason": f"Unexpected error: {type(e).__name__}"}

    async def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        """Checks the status of a deployed agent container."""
        if not DOCKER_AVAILABLE: return {"status": "error", "reason": "Docker library not available"}
        logger.debug(f"Getting status for container: {instance_id}")
        container = await self._find_container(instance_id)
        if container:
            try:
                await asyncio.to_thread(container.reload) # Refresh container state
                status = container.status
                details = container.attrs # Get full attributes
                logger.info(f"Status for container '{instance_id}': {status}")
                return {"status": status, "instance_id": instance_id, "details": {"State": details.get("State", {}), "Created": details.get("Created")}}
            except Exception as e:
                 logger.error(f"Error reloading/getting status for '{instance_id}': {e}", exc_info=True)
                 # Try to return basic status even if reload fails
                 return {"status": getattr(container, 'status', 'error'), "reason": f"Error getting detailed status: {e}"}
        else:
            logger.info(f"Container '{instance_id}' not found for status check.") # Info level is fine
            return {"status": "not_found", "instance_id": instance_id}

    async def stop_agent_instance(self, instance_id: str) -> bool:
        """Stops a running agent container."""
        if not DOCKER_AVAILABLE: return False
        logger.info(f"Attempting to stop container: {instance_id}")
        container = await self._find_container(instance_id)
        if container:
            if container.status == 'running':
                try:
                    await asyncio.to_thread(container.stop, timeout=30) # Standard 30s timeout
                    logger.info(f"Container '{instance_id}' stopped successfully.")
                    return True
                except APIError as e:
                     logger.error(f"Docker API error stopping container '{instance_id}': Status {e.status_code} - {e.explanation}", exc_info=False)
                     return False
                except Exception as e:
                     logger.error(f"Error stopping container '{instance_id}': {e}", exc_info=True)
                     return False
            else:
                 logger.info(f"Container '{instance_id}' already stopped (status: {container.status}).")
                 return True # It's already stopped
        else:
            logger.warning(f"Container '{instance_id}' not found, cannot stop.")
            return False # Indicate not found

    async def start_agent_instance(self, instance_id: str) -> bool:
        """Starts a stopped agent container."""
        if not DOCKER_AVAILABLE: return False
        logger.info(f"Attempting to start container: {instance_id}")
        container = await self._find_container(instance_id)
        if container:
            if container.status != 'running':
                try:
                    await asyncio.to_thread(container.start)
                    logger.info(f"Container '{instance_id}' started successfully.")
                    return True
                except APIError as e:
                     logger.error(f"Docker API error starting container '{instance_id}': Status {e.status_code} - {e.explanation}", exc_info=False)
                     return False
                except Exception as e:
                     logger.error(f"Error starting container '{instance_id}': {e}", exc_info=True)
                     return False
            else:
                logger.info(f"Container '{instance_id}' already running.")
                return True
        else:
            logger.warning(f"Container '{instance_id}' not found, cannot start.")
            return False

    async def delete_agent_instance(self, instance_id: str) -> bool:
        """Stops and removes an agent container."""
        if not DOCKER_AVAILABLE: return False
        logger.info(f"Attempting to delete container: {instance_id}")
        container = await self._find_container(instance_id)
        if container:
            try:
                # Ensure stopped before removing if necessary
                if container.status == 'running':
                    logger.info(f"Stopping container {instance_id} before removal...")
                    await asyncio.to_thread(container.stop, timeout=10)
                await asyncio.to_thread(container.remove, force=True, v=False) # v=True removes anonymous volumes, maybe undesirable
                logger.info(f"Container '{instance_id}' deleted successfully.")
                return True
            except APIError as e:
                 # Handle "already removing" or "not found" errors gracefully during delete
                 if e.response.status_code == 404: # Not found
                     logger.warning(f"Container '{instance_id}' already removed (404).")
                     return True
                 elif e.response.status_code == 409: # Conflict (e.g., already being removed)
                      logger.warning(f"Container '{instance_id}' removal conflict (409), likely already in progress.")
                      return True # Assume it will be removed
                 else:
                     logger.error(f"Docker API error deleting container '{instance_id}': Status {e.status_code} - {e.explanation}", exc_info=False)
                     return False
            except Exception as e:
                 logger.error(f"Error deleting container '{instance_id}': {e}", exc_info=True)
                 return False
        else:
            logger.warning(f"Container '{instance_id}' not found, cannot delete (already gone).")
            return True # Already gone


# --- Test Function ---
async def _test_docker_deployment_manager_final():
    print("--- Testing DockerDeploymentManager (FINAL) ---")
    # ... (rest of test remains the same) ...
    # Test function requires Docker running and accessible.
    pass

if __name__ == "__main__":
    # import asyncio
    # from dotenv import load_dotenv
    # load_dotenv()
    # # Ensure Docker is installed: pip install docker
    # if DOCKER_AVAILABLE:
    #    asyncio.run(_test_docker_deployment_manager_final())
    # else:
    #    print("Skipping DockerDeploymentManager test: Docker library not installed.")
    print("DeploymentManagerInterface and DockerDeploymentManager defined.")
# --------------------------------------------------------------------------------
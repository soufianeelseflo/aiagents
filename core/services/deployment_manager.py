# boutique_ai_project/core/services/deployment_manager.py

import logging
import asyncio
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

import docker # Requires 'pip install docker'
from docker.errors import APIError, NotFound, DockerException
from docker.models.containers import Container

import config # Root config

logger = logging.getLogger(__name__)

class DeploymentConfig:
    """Data class for deploying a client agent instance."""
    def __init__(self, client_id: str, client_email: str, agent_type: str, configuration: Dict[str, Any]):
        self.client_id = client_id
        self.client_email = client_email
        self.agent_type = agent_type # e.g., "SalesAgent", "AcquisitionAgent" - determines which agent logic to run if image supports multiple
        self.configuration = configuration # Agent-specific settings, prompts, API keys etc.
        # Generate a unique, predictable name for the container
        self.instance_id: str = f"boutiqueai_agent_{self.agent_type.lower()}_{self.client_id.replace('-','_')[:8]}"

    def get_environment_variables(self) -> Dict[str, str]:
        """Prepares environment variables for the container based on configuration."""
        env_vars = {
            "BOUTIQUEAI_AGENT_TYPE": self.agent_type,
            "BOUTIQUEAI_CLIENT_ID": self.client_id,
            "BOUTIQUEAI_CLIENT_EMAIL": self.client_email,
            # Pass essential configurations as environment variables
            # Sensitive keys should ideally be mounted as secrets, but env vars are simpler for this example
            "AGENT_TARGET_NICHE_OVERRIDE": self.configuration.get("target_niche", config.AGENT_TARGET_NICHE_DEFAULT),
            # Add client-specific API keys if provided in the config
            # Example: If config contains {"client_api_keys": {"some_service": "key123"}}
            # "SOME_SERVICE_API_KEY": self.configuration.get("client_api_keys", {}).get("some_service"),
            # Pass other relevant config items as JSON string or individual vars
            "AGENT_INITIAL_PROMPT_CONTEXT_JSON": json.dumps(self.configuration.get("initial_prompt_context", {})),
            # Ensure core service keys are available if not using client-specific ones
            "TWILIO_ACCOUNT_SID": config.TWILIO_ACCOUNT_SID,
            "TWILIO_AUTH_TOKEN": config.TWILIO_AUTH_TOKEN,
            "TWILIO_PHONE_NUMBER": config.TWILIO_PHONE_NUMBER, # Might need client-specific number later
            "DEEPGRAM_API_KEY": config.DEEPGRAM_API_KEY,
            "OPENROUTER_API_KEY": config.OPENROUTER_API_KEY,
            "SUPABASE_URL": config.SUPABASE_URL,
            "SUPABASE_KEY": config.SUPABASE_KEY, # Use service role key for agents? Or client-specific keys? Needs thought.
            "LOG_LEVEL": config.LOG_LEVEL,
            # Add any other necessary env vars based on agent needs
        }
        # Filter out None values
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
    Assumes Docker daemon is accessible from where this code runs (e.g., via Docker socket).
    """
    def __init__(self, base_image_name: str = "boutique-ai-engine:latest"): # Use the image built by Dockerfile/Coolify
        try:
            self.client = docker.from_env()
            # Test connection
            self.client.ping()
            self.base_image_name = base_image_name
            logger.info(f"DockerDeploymentManager initialized. Using base image: {self.base_image_name}. Connected to Docker daemon.")
        except DockerException as e:
            logger.critical(f"Failed to connect to Docker daemon: {e}")
            logger.critical("Ensure Docker is running and accessible (e.g., check socket permissions or Docker context).")
            raise ConnectionError("Could not connect to Docker daemon") from e

    async def _find_container(self, instance_id: str) -> Optional[Container]:
        """Finds a container by name (instance_id)."""
        try:
            container = await asyncio.to_thread(self.client.containers.get, instance_id)
            return container
        except NotFound:
            return None
        except APIError as e:
            logger.error(f"Docker API error finding container '{instance_id}': {e}", exc_info=True)
            return None

    async def deploy_agent_instance(self, deployment_config: DeploymentConfig) -> Dict[str, Any]:
        """Deploys a new agent instance as a Docker container."""
        instance_id = deployment_config.instance_id # Use the generated name
        logger.info(f"Attempting to deploy Docker container: {instance_id}")

        # Check if container already exists
        existing_container = await self._find_container(instance_id)
        if existing_container:
            logger.warning(f"Container '{instance_id}' already exists. Status: {existing_container.status}. Attempting to ensure it's running.")
            if existing_container.status != 'running':
                await self.start_agent_instance(instance_id)
            return {"status": "success", "instance_id": instance_id, "message": "Instance already existed, ensured running."}

        try:
            env_vars = deployment_config.get_environment_variables()
            logger.debug(f"Deploying container '{instance_id}' with image '{self.base_image_name}' and env vars.")
            
            # Run the container in detached mode
            # We need a command that starts the specific agent type within the container.
            # This assumes server.py or another script can parse BOUTIQUEAI_AGENT_TYPE env var
            # and run the appropriate agent loop. This requires modification to server.py or a new entrypoint script.
            # For now, let's assume the container's default CMD runs the main server,
            # and we need a different command or entrypoint for dedicated agents.
            # --- Placeholder Command ---
            # This needs refinement based on how agents are started within the container image.
            # Option 1: Modify server.py to check env var and run agent loop instead of web server.
            # Option 2: Create a separate agent_runner.py script.
            # Option 3: Use docker exec after starting (more complex).
            # Assuming Option 2: python -m core.agent_runner --agent-type SalesAgent --client-id ...
            # This requires creating agent_runner.py
            logger.warning("Deployment command within container needs implementation (e.g., agent_runner.py). Using default CMD for now.")
            command = None # Use default CMD from Dockerfile for now

            container = await asyncio.to_thread(
                self.client.containers.run,
                image=self.base_image_name,
                detach=True,
                name=instance_id,
                environment=env_vars,
                command=command, # Specify command to run specific agent type if needed
                restart_policy={"Name": "unless-stopped"}, # Basic restart policy
                # Add network configuration if needed (e.g., connect to specific Docker network)
                # network="your_network_name",
                # Add volume mounts if agents need persistent storage separate from main app
                # volumes={'/path/on/host': {'bind': '/path/in/container', 'mode': 'rw'}}
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
                # Reload attributes to get latest status
                await asyncio.to_thread(container.reload)
                logger.info(f"Status for container '{instance_id}': {container.status}")
                return {"status": container.status, "instance_id": instance_id, "details": container.attrs}
            except APIError as e:
                 logger.error(f"Docker API error getting status for '{instance_id}': {e}")
                 return {"status": "error", "reason": f"Docker API Error: {e}"}
            except Exception as e:
                 logger.error(f"Unexpected error getting status for '{instance_id}': {e}", exc_info=True)
                 return {"status": "error", "reason": f"Unexpected error: {e}"}
        else:
            logger.warning(f"Container '{instance_id}' not found for status check.")
            return {"status": "not_found", "instance_id": instance_id}

    async def stop_agent_instance(self, instance_id: str) -> bool:
        """Stops a running agent container."""
        logger.info(f"Attempting to stop container: {instance_id}")
        container = await self._find_container(instance_id)
        if container:
            try:
                await asyncio.to_thread(container.stop, timeout=30) # 30 second timeout
                logger.info(f"Container '{instance_id}' stopped successfully.")
                return True
            except APIError as e:
                logger.error(f"Docker API error stopping container '{instance_id}': {e}")
                return False
            except Exception as e:
                 logger.error(f"Unexpected error stopping container '{instance_id}': {e}", exc_info=True)
                 return False
        else:
            logger.warning(f"Container '{instance_id}' not found, cannot stop.")
            return False # Or True if "already stopped" is considered success

    async def start_agent_instance(self, instance_id: str) -> bool:
        """Starts a stopped agent container."""
        logger.info(f"Attempting to start container: {instance_id}")
        container = await self._find_container(instance_id)
        if container:
            try:
                await asyncio.to_thread(container.start)
                logger.info(f"Container '{instance_id}' started successfully.")
                return True
            except APIError as e:
                logger.error(f"Docker API error starting container '{instance_id}': {e}")
                return False
            except Exception as e:
                 logger.error(f"Unexpected error starting container '{instance_id}': {e}", exc_info=True)
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
                # Force remove, stops if running
                await asyncio.to_thread(container.remove, force=True, v=True) # v=True removes associated anonymous volumes
                logger.info(f"Container '{instance_id}' deleted successfully.")
                return True
            except APIError as e:
                logger.error(f"Docker API error deleting container '{instance_id}': {e}")
                return False
            except Exception as e:
                 logger.error(f"Unexpected error deleting container '{instance_id}': {e}", exc_info=True)
                 return False
        else:
            logger.warning(f"Container '{instance_id}' not found, cannot delete.")
            return True # Already gone

# --- Test Function ---
async def _test_docker_deployment_manager():
    print("--- Testing DockerDeploymentManager ---")
    # This test requires Docker running and accessible, and an image named 'hello-world' (or change image name)
    try:
        # Use a readily available image for testing basic Docker interaction
        manager = DockerDeploymentManager(base_image_name="hello-world:latest")
        
        # Create dummy config
        client_id_test = f"client_{int(time.time())}"
        test_config = DeploymentConfig(
            client_id=client_id_test,
            client_email=f"{client_id_test}@test.com",
            agent_type="TestAgent",
            configuration={"test_param": "value"}
        )
        instance_id_test = test_config.instance_id

        print(f"\n1. Attempting to deploy instance: {instance_id_test}")
        deploy_result = await manager.deploy_agent_instance(test_config)
        print(f"  Deployment Result: {deploy_result}")

        if deploy_result.get("status") == "success":
            print(f"\n2. Getting status for instance: {instance_id_test}")
            await asyncio.sleep(2) # Give container time to potentially exit if it's hello-world
            status_result = await manager.get_instance_status(instance_id_test)
            print(f"  Status Result: {status_result}")

            print(f"\n3. Attempting to stop instance: {instance_id_test}")
            stop_result = await manager.stop_agent_instance(instance_id_test)
            print(f"  Stop Result: {stop_result}")
            await asyncio.sleep(1)
            status_after_stop = await manager.get_instance_status(instance_id_test)
            print(f"  Status after stop: {status_after_stop.get('status')}")


            print(f"\n4. Attempting to start instance: {instance_id_test}")
            start_result = await manager.start_agent_instance(instance_id_test)
            print(f"  Start Result: {start_result}")
            await asyncio.sleep(1)
            status_after_start = await manager.get_instance_status(instance_id_test)
            print(f"  Status after start: {status_after_start.get('status')}")

            print(f"\n5. Attempting to delete instance: {instance_id_test}")
            delete_result = await manager.delete_agent_instance(instance_id_test)
            print(f"  Delete Result: {delete_result}")
            status_after_delete = await manager.get_instance_status(instance_id_test)
            print(f"  Status after delete: {status_after_delete.get('status')}") # Should be not_found
        else:
            print("  Skipping further tests as deployment failed.")

    except ConnectionError as e:
        print(f"Test failed: Could not connect to Docker daemon. {e}")
    except Exception as e:
        print(f"Test failed unexpectedly: {e}", exc_info=True)

if __name__ == "__main__":
    # import asyncio
    # from dotenv import load_dotenv
    # load_dotenv() # Load .env if needed by config
    # asyncio.run(_test_docker_deployment_manager())
    print("DeploymentManagerInterface and DockerDeploymentManager defined. Test requires Docker.")
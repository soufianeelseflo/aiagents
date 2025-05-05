# core/services/proxy_manager_wrapper.py

import logging
import random
from typing import Optional, Tuple

# Import configuration centrally (though specific user/pass likely needed per agent/task)
# from config import SMARTPROXY_DEFAULT_USERNAME, SMARTPROXY_DEFAULT_PASSWORD # Example if defaults exist

logger = logging.getLogger(__name__)

# --- Smartproxy/Decodata Endpoint Information ---
# Based on Smartproxy documentation/common knowledge. Adjust if Decodata has different specifics.
# Mapping country codes (or regions) to Smartproxy endpoint domains.
# Add more as needed based on https://help.smartproxy.com/docs/list-of-all-residential-endpoints
SMARTPROXY_ENDPOINTS = {
    "random": "gate.smartproxy.com", # Random location from the pool
    "us": "us.smartproxy.com",
    "ca": "ca.smartproxy.com",
    "gb": "gb.smartproxy.com",
    "de": "de.smartproxy.com",
    "fr": "fr.smartproxy.com",
    "au": "au.smartproxy.com",
    # Add other specific countries or regions as needed
}

# Port ranges often determine session type (e.g., rotating vs. sticky)
# Consult Smartproxy documentation for exact port mapping for desired session behavior.
# Example: Rotating session ports might be 10000-19999, Sticky might be 30000+
ROTATING_PORT_RANGE = (10000, 19999) # Example range - VERIFY WITH SMARTPROXY DOCS
# STICKY_PORT_RANGE = (30000, 39999) # Example range - VERIFY WITH SMARTPROXY DOCS

class ProxyManagerWrapper:
    """
    Wrapper to construct connection strings for Decodata/Smartproxy residential proxies.
    Does NOT interact with the Smartproxy Management API, only formats connection details.
    Requires proxy username and password to be provided when getting a proxy string.
    """

    def __init__(self):
        """ Initializes the proxy manager wrapper. """
        logger.info("ProxyManagerWrapper (Decodata/Smartproxy) initialized.")
        # No API key needed for this approach, relies on user/pass credentials per request.

    def get_proxy_string(
        self,
        proxy_username: str,
        proxy_password: str,
        location: str = "random", # Default to random country from pool
        session_type: str = "rotating" # 'rotating' or 'sticky' (determines port)
        ) -> Optional[str]:
        """
        Constructs a proxy connection string for Decodata/Smartproxy.

        Args:
            proxy_username: The username for proxy authentication.
            proxy_password: The password for proxy authentication.
            location: Desired geographic location (e.g., 'us', 'ca', 'random').
                     Must be a key in SMARTPROXY_ENDPOINTS.
            session_type: 'rotating' or 'sticky' (determines port range).

        Returns:
            A formatted proxy connection string (http://user:pass@host:port) or None if invalid inputs.
        """
        if not proxy_username or not proxy_password:
            logger.error("Proxy username and password are required to generate connection string.")
            return None

        # Select endpoint based on location
        endpoint_host = SMARTPROXY_ENDPOINTS.get(location.lower())
        if not endpoint_host:
            logger.error(f"Invalid or unsupported location specified: '{location}'. Using 'random'.")
            endpoint_host = SMARTPROXY_ENDPOINTS["random"]

        # Select port based on session type (using example ranges - VERIFY)
        port: Optional[int] = None
        if session_type.lower() == "rotating":
            # Select a random port within the rotating range
            port = random.randint(ROTATING_PORT_RANGE[0], ROTATING_PORT_RANGE[1])
            logger.debug(f"Selected rotating session port: {port}")
        # elif session_type.lower() == "sticky":
        #     port = random.randint(STICKY_PORT_RANGE[0], STICKY_PORT_RANGE[1])
        #     logger.debug(f"Selected sticky session port: {port}")
        else:
            logger.error(f"Invalid session_type specified: '{session_type}'. Must be 'rotating' or 'sticky'.")
            # Fallback to rotating if invalid type given? Or return None? Let's fallback.
            port = random.randint(ROTATING_PORT_RANGE[0], ROTATING_PORT_RANGE[1])
            logger.warning(f"Invalid session_type, defaulting to rotating port: {port}")

        if port is None: # Should not happen with fallback, but safety check
             logger.error("Could not determine proxy port.")
             return None

        # Construct the connection string
        # Assumes HTTP protocol for the proxy connection itself
        proxy_string = f"http://{proxy_username}:{proxy_password}@{endpoint_host}:{port}"
        logger.info(f"Constructed Smartproxy connection string for location '{location}', session '{session_type}'.")
        logger.debug(f"Proxy String (masked pass): http://{proxy_username}:*****@{endpoint_host}:{port}")

        return proxy_string

    # Note: Release proxy function is removed as this wrapper doesn't manage sessions via API.
    # Session control is typically handled by using different port ranges or parameters
    # in the connection string itself, as per Smartproxy documentation.

# Example Usage (Conceptual)
def main():
     print("Testing ProxyManagerWrapper (Decodata/Smartproxy)...")
     # These credentials would typically come from ResourceManager or secure config
     TEST_USER = "proxy_user_123"
     TEST_PASS = "proxy_password_abc"

     wrapper = ProxyManagerWrapper()

     print("\nRequesting US rotating proxy string:")
     proxy1 = wrapper.get_proxy_string(TEST_USER, TEST_PASS, location="us", session_type="rotating")
     if proxy1:
         print(f"Got proxy string: http://{TEST_USER}:*****@{proxy1.split('@')[1]}") # Mask password
     else:
         print("Failed to get proxy string.")

     print("\nRequesting Random rotating proxy string:")
     proxy2 = wrapper.get_proxy_string(TEST_USER, TEST_PASS, location="random") # Default session is rotating
     if proxy2:
         print(f"Got proxy string: http://{TEST_USER}:*****@{proxy2.split('@')[1]}")
     else:
         print("Failed to get proxy string.")

     # print("\nRequesting CA sticky proxy string:")
     # proxy3 = wrapper.get_proxy_string(TEST_USER, TEST_PASS, location="ca", session_type="sticky")
     # if proxy3:
     #     print(f"Got proxy string: http://{TEST_USER}:*****@{proxy3.split('@')[1]}")
     # else:
     #     print("Failed to get proxy string.")

if __name__ == "__main__":
     # main() # Uncomment to run test
     print("ProxyManagerWrapper (Decodata/Smartproxy) structure defined. Run test manually.")


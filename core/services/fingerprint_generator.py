# /core/services/fingerprint_generator.py:
# --------------------------------------------------------------------------------
# boutique_ai_project/core/services/fingerprint_generator.py

import logging
import json
import hashlib # For deterministic parts if needed
from typing import Dict, Any, Optional, List
from faker import Faker # For generating synthetic but realistic data
from fake_useragent import UserAgent # For realistic user agent strings

# Local imports
import config
from core.services.llm_client import LLMClient, LLMClientSetupError # Import custom error

logger = logging.getLogger(__name__)

class FingerprintGeneratorError(Exception):
    """Custom exception for FingerprintGenerator errors."""
    pass

class FingerprintGenerator:
    """
    Generates sophisticated and dynamic browser/system fingerprints for agents,
    enhancing realism and reducing detectability. (Level 48)
    Uses a combination of deterministic, semi-random (Faker-based), and LLM-enhanced
    parameter generation.
    """

    # Pre-defined common values to select from or use as base
    COMMON_SCREEN_RESOLUTIONS = [
        "1920x1080", "1366x768", "1536x864", "2560x1440", "1440x900",
        "1280x720", "1600x900", "3840x2160", "1280x1024", "1920x1200"
    ]
    COMMON_COLOR_DEPTHS = [24, 32] # Typically 24-bit or 30/32-bit for HDR
    COMMON_LANGUAGES = ["en-US", "en-GB", "es-ES", "fr-FR", "de-DE", "en", "es", "fr", "de"] # Prioritize more specific
    COMMON_TIMEZONES = [ # Sample, not exhaustive
        "America/New_York", "America/Los_Angeles", "Europe/London",
        "Europe/Paris", "Europe/Berlin", "America/Chicago", "Asia/Tokyo"
    ]

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initializes the FingerprintGenerator.
        Args:
            llm_client: An optional instance of LLMClient. If not provided,
                        LLM-enhanced generation features will be disabled or limited.
        """
        self.llm_client = llm_client
        self.faker = Faker() # Initialize Faker for generating diverse data
        try:
            self.user_agent_rotator = UserAgent(fallback="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36") # Updated fallback
        except Exception as e: # Catch potential errors during UserAgent initialization (e.g., network issues fetching data)
            logger.error(f"Failed to initialize UserAgent: {e}. Using a fixed fallback UA.", exc_info=True)
            self.user_agent_rotator = None # Mark as unavailable, methods will use fixed fallback

        if not self.llm_client:
            logger.warning(
                "LLMClient not provided to FingerprintGenerator. "
                "LLM-enhanced fingerprint generation will be disabled. "
                "Consider providing an LLMClient for more dynamic fingerprints."
            )
        logger.info("FingerprintGenerator initialized.")

    def _get_random_user_agent(self, os_type: Optional[str] = None, navigator_type: Optional[str] = None) -> str:
        """Gets a random User-Agent string, optionally filtered by OS or browser type."""
        if not self.user_agent_rotator:
            logger.debug("UserAgent rotator not available, using hardcoded fallback UA.")
            return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36" # Keep a recent general UA

        try:
            if os_type and navigator_type:
                # This is an example; fake-useragent might not support combined filtering directly
                # It often filters by browser name which implies OS, or specific OS.
                # We'll try browser first, then OS if that fails.
                try: return self.user_agent_rotator.get(navigator=navigator_type.lower(), os=os_type.lower())
                except: pass # Try next
            if navigator_type:
                return self.user_agent_rotator.get(navigator=navigator_type.lower())
            if os_type:
                return self.user_agent_rotator.get(os=os_type.lower())
            return self.user_agent_rotator.random # Get any random UA
        except Exception as e:
            logger.warning(f"Error getting random User-Agent from fake-useragent: {e}. Using fallback.", exc_info=True)
            return self.user_agent_rotator.fallback if self.user_agent_rotator and self.user_agent_rotator.fallback else "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"

    def _get_semi_random_platform(self, user_agent_str: str) -> str:
        """Derives a plausible platform string from the User-Agent."""
        ua_lower = user_agent_str.lower()
        if "windows" in ua_lower: return self.faker.random_element(elements=("Win32", "Win64")) # Win64 more common now
        if "linux" in ua_lower: return self.faker.random_element(elements=("Linux armv8l", "Linux x86_64", "Linux i686"))
        if "mac os x" in ua_lower or "macintosh" in ua_lower: return "MacIntel" # Common for modern Macs
        if "android" in ua_lower: return self.faker.random_element(elements=("Linux armv8l", "Linux aarch64")) # Common for Android
        if "iphone" in ua_lower or "ipad" in ua_lower: return "iPhone" # Or "iPad", but "iPhone" often reported
        return "Unknown" # Fallback

    async def _llm_enhance_parameter(self, parameter_name: str, base_value: Any, context: Dict[str, Any]) -> Any:
        """Uses LLM to generate a more realistic or contextually relevant parameter value."""
        if not self.llm_client:
            logger.debug(f"LLM enhancement skipped for '{parameter_name}' as LLMClient is not available.")
            return base_value

        prompt_template = (
            f"Given the following browser context and a base value for a parameter, "
            f"suggest a more realistic or contextually appropriate value for '{parameter_name}'. "
            f"If the base value is already good or you cannot improve it, return the base value. "
            f"Focus on realism and common patterns. Do not explain, just provide the value.\n\n"
            f"Context:\n"
            f"User-Agent: {context.get('userAgent', 'N/A')}\n"
            f"Platform: {context.get('platform', 'N/A')}\n"
            f"Language: {context.get('language', 'N/A')}\n"
            f"Screen Resolution: {context.get('screenResolution', 'N/A')}\n\n"
            f"Parameter to enhance: '{parameter_name}'\n"
            f"Base value: '{base_value}'\n\n"
            f"Suggested realistic value (return base value if unsure or good enough):"
        )
        messages = [{"role": "user", "content": prompt_template}]

        try:
            # Use a model good for quick, creative suggestions if available and configured
            enhancement_model = config.OPENROUTER_DEFAULT_STRATEGY_MODEL # Or a specific smaller/faster model
            response_data = await self.llm_client.get_chat_completion(
                messages,
                model=enhancement_model,
                temperature=0.6, # Moderate temperature for some variability
                max_tokens=50 # Values are usually short
            )
            if response_data and response_data.get("choices"):
                suggested_value = response_data["choices"][0]["message"]["content"].strip().strip('"').strip("'")
                logger.debug(f"LLM suggested value for '{parameter_name}': '{suggested_value}' (Base was: '{base_value}')")
                # Basic validation: if LLM returns empty or something too different, stick to base.
                # This needs to be carefully tuned. For now, if it's non-empty, use it.
                return suggested_value if suggested_value else base_value
        except LLMClientSetupError: # Handle if LLM client had a setup issue later
            logger.error("LLMClient encountered a setup error during enhancement. LLM enhancement disabled for this call.")
            self.llm_client = None # Disable further LLM calls if setup issue
        except Exception as e:
            logger.error(f"Error during LLM enhancement for '{parameter_name}': {e}", exc_info=True)
        return base_value # Fallback to base value on error or no suggestion

    async def generate_fingerprint(
        self,
        base_os: Optional[str] = None, # e.g., "windows", "macos", "linux", "android"
        base_navigator: Optional[str] = None, # e.g., "chrome", "firefox", "safari"
        use_llm_enhancement: bool = True
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive browser/system fingerprint.
        Args:
            base_os: Hint for the base operating system.
            base_navigator: Hint for the base browser type.
            use_llm_enhancement: Whether to use LLM for refining parameters.
                                 Requires LLMClient to be initialized.
        Returns:
            A dictionary representing the fingerprint.
        """
        logger.debug(f"Generating fingerprint. Base OS: {base_os}, Base Nav: {base_navigator}, LLM Enhance: {use_llm_enhancement}")
        faker_instance = self.faker # Use the instance member

        # 1. User Agent (foundation)
        user_agent = self._get_random_user_agent(os_type=base_os, navigator_type=base_navigator)

        # 2. Platform (derived from UA, semi-random)
        platform = self._get_semi_random_platform(user_agent)

        # 3. Language (semi-random, Faker provides some, or pick from common)
        language = faker_instance.random_element(elements=self.COMMON_LANGUAGES)

        # 4. Screen Resolution & Color Depth (semi-random)
        screen_resolution = faker_instance.random_element(elements=self.COMMON_SCREEN_RESOLUTIONS)
        color_depth = faker_instance.random_element(elements=self.COMMON_COLOR_DEPTHS)
        device_memory = faker_instance.random_element(elements=(2, 4, 8, 16, 32, 64)) # Common RAM gigs

        # 5. Timezone (semi-random)
        timezone = faker_instance.random_element(elements=self.COMMON_TIMEZONES)

        # 6. WebGL Renderer & Vendor (can be very specific, use Faker for plausibility)
        # These are often very specific. Faker can generate plausible prefixes.
        # A more advanced approach would involve an LLM or a database of common WebGL strings.
        common_vendors = ["Google Inc. (NVIDIA)", "Google Inc. (Intel)", "Apple Inc.", "Mozilla", "Intel Inc.", "NVIDIA Corporation", "ATI Technologies Inc."]
        webgl_vendor = faker_instance.random_element(elements=common_vendors)
        if "intel" in webgl_vendor.lower():
            renderer_prefix = "ANGLE (Intel, Intel(R) Iris(R) Xe Graphics Direct3D11 vs_5_0 ps_5_0"
        elif "nvidia" in webgl_vendor.lower():
            renderer_prefix = "ANGLE (NVIDIA, NVIDIA GeForce RTX 3080 Direct3D11 vs_5_0 ps_5_0"
        elif "apple" in webgl_vendor.lower():
            renderer_prefix = "Apple M1" # Or M2, M3 etc.
        else:
            renderer_prefix = "ANGLE (Unknown)"
        webgl_renderer = f"{renderer_prefix}, {faker_instance.word()})" # Add a random word part

        # 7. Canvas Fingerprint (simple hash for now, true canvas requires rendering)
        # This is a placeholder. Real canvas fingerprinting is complex.
        # We use a hash of some other properties to make it vary consistently.
        canvas_data_string = f"{user_agent}-{screen_resolution}-{color_depth}-{language}"
        canvas_fingerprint = hashlib.md5(canvas_data_string.encode()).hexdigest()

        # 8. Other common parameters
        # Using boolean values directly, not strings 'true'/'false' unless JS expects strings
        do_not_track = faker_instance.random_element(elements=(True, False, None)) # DNT can be 1, 0, or not set (null)
        hardware_concurrency = faker_instance.random_element(elements=(2, 4, 8, 12, 16, 20, 24, 32)) # CPU cores
        # BuildID should be somewhat related to browser version/date
        build_id = faker_instance.date_time_this_decade(before_now=True, after_now=False).strftime('%Y%m%d%H%M%S')

        # --- Initial Fingerprint Dictionary ---
        fingerprint = {
            "userAgent": user_agent,
            "platform": platform,
            "language": language, # navigator.language
            "languages": [language, language.split('-')[0], "en"], # navigator.languages (ordered by preference)
            "screenResolution": screen_resolution, # e.g., screen.width x screen.height
            "colorDepth": color_depth, # screen.colorDepth
            "deviceMemory": device_memory, # navigator.deviceMemory (approx gigs)
            "timezone": timezone, # Intl.DateTimeFormat().resolvedOptions().timeZone
            "webglVendor": webgl_vendor,
            "webglRenderer": webgl_renderer,
            "canvasFingerprint": canvas_fingerprint, # Highly variable, site-specific generation logic
            "doNotTrack": do_not_track, # navigator.doNotTrack
            "hardwareConcurrency": hardware_concurrency, # navigator.hardwareConcurrency
            "buildID": build_id, # navigator.buildID
            "plugins": [], # navigator.plugins (typically empty or specific list for modern browsers)
            "mimeTypes": [], # navigator.mimeTypes (similar to plugins)
            "cookiesEnabled": True, # navigator.cookieEnabled
            "javaEnabled": False, # navigator.javaEnabled() (almost always false now)
            "productSub": "20030107", # Common value for Gecko-based browsers
            "vendor": self.faker.random_element(elements=("Google Inc.", "Apple Computer, Inc.", "")), # navigator.vendor
            "vendorSub": "", # navigator.vendorSub
            # Additional properties often checked
            "touchSupport": {"maxTouchPoints": faker_instance.random_element(elements=(0, 1, 5)), "touchEvent": faker_instance.boolean(), "ontouchend": faker_instance.boolean()},
            "fonts": self.faker.random_elements(elements_ordered=( # A small, plausible set of common fonts
                "Arial", "Times New Roman", "Courier New", "Verdana", "Georgia", "Helvetica", "Calibri", "Segoe UI"
            ), length=self.faker.random_int(min=3, max=8), unique=True)
        }

        # --- LLM Enhancement (Optional) ---
        if use_llm_enhancement and self.llm_client:
            logger.debug("Attempting LLM enhancement for selected fingerprint parameters...")
            # Select a few parameters that might benefit most from contextual LLM refinement
            # For example, 'webglRenderer' or 'platform' if the base is too generic.
            # Here, we'll try to enhance a couple of complex ones.
            # This is illustrative; the choice of parameters and prompts needs careful design.

            # Create a context dictionary for the LLM
            current_context = {
                "userAgent": fingerprint["userAgent"],
                "platform": fingerprint["platform"],
                "language": fingerprint["language"],
                "screenResolution": fingerprint["screenResolution"]
            }

            # Example: Enhance WebGL Renderer
            # This specific parameter can be very diverse and identifying.
            # An LLM might generate a more plausible string based on context.
            fingerprint["webglRenderer"] = await self._llm_enhance_parameter(
                "webglRenderer", fingerprint["webglRenderer"], current_context
            )
            # Example: Enhance platform string if it's too generic or needs subtlety
            fingerprint["platform"] = await self._llm_enhance_parameter(
                "platform", fingerprint["platform"], current_context
            )
            # Example: Potentially refine userAgent itself if LLM can make it more consistent with other params
            # fingerprint["userAgent"] = await self._llm_enhance_parameter(
            # "userAgent", fingerprint["userAgent"], current_context
            # )
        elif use_llm_enhancement and not self.llm_client:
            logger.warning("LLM enhancement requested but LLMClient is not available.")


        # --- Post-processing and Final Checks ---
        # Ensure 'languages' is consistent with 'language'
        if fingerprint["language"] not in fingerprint["languages"]:
            fingerprint["languages"].insert(0, fingerprint["language"])
        fingerprint["languages"] = list(dict.fromkeys(fingerprint["languages"])) # Remove duplicates, preserve order


        logger.info(f"Generated fingerprint. User-Agent: {fingerprint['userAgent']}")
        # logger.debug(f"Full fingerprint details: {json.dumps(fingerprint, indent=2)}")
        return fingerprint

    async def generate_multiple_fingerprints(self, count: int, **kwargs) -> List[Dict[str, Any]]:
        """Generates a list of unique fingerprints."""
        fingerprints = []
        attempts = 0
        max_attempts = count * 3 # Allow some leeway for collisions if generation isn't perfectly unique

        # Using a set to quickly check for uniqueness based on a key part (e.g., User-Agent)
        # For true uniqueness, one might hash the whole dict, but UA is a good proxy.
        # However, full dict comparison is safer for true uniqueness.
        # For now, we'll rely on the inherent randomness of generation for uniqueness over small counts.
        # If stricter uniqueness is needed, a set of frozenset(fp.items()) could be used.

        while len(fingerprints) < count and attempts < max_attempts:
            fp = await self.generate_fingerprint(**kwargs)
            # Simple uniqueness check (can be made more robust)
            # For now, assume generate_fingerprint is random enough for small counts
            # If more robust check is needed:
            # if not any(existing_fp == fp for existing_fp in fingerprints):
            # fingerprints.append(fp)
            fingerprints.append(fp) # Adding without strict uniqueness check for now
            attempts += 1

        if len(fingerprints) < count:
            logger.warning(f"Could only generate {len(fingerprints)} unique fingerprints out of {count} requested after {max_attempts} attempts.")

        return fingerprints

# --- Example Usage (for testing) ---
async def main_test():
    # Load .env for local testing
    from dotenv import load_dotenv, find_dotenv
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path: load_dotenv(dotenv_path)
    else: logger.warning("No .env file found for FingerprintGenerator test. LLM features may be limited.")

    # Initialize LLMClient if API key is available
    test_llm_client = None
    if config.OPENROUTER_API_KEY:
        try:
            test_llm_client = LLMClient()
        except LLMClientSetupError as e:
            logger.error(f"LLMClient setup failed for test: {e}")
        except Exception as e:
            logger.error(f"Unexpected error setting up LLMClient for test: {e}", exc_info=True)

    generator = FingerprintGenerator(llm_client=test_llm_client)

    logger.info("\n--- Generating Fingerprint (No LLM Enhancement, default hints) ---")
    fp1 = await generator.generate_fingerprint(use_llm_enhancement=False)
    logger.info(f"FP1 User-Agent: {fp1.get('userAgent')}")
    logger.info(f"FP1 Platform: {fp1.get('platform')}")
    # logger.info(f"FP1 Full: {json.dumps(fp1, indent=2)}")


    if test_llm_client:
        logger.info("\n--- Generating Fingerprint (With LLM Enhancement, Windows/Chrome hints) ---")
        fp2 = await generator.generate_fingerprint(base_os="windows", base_navigator="chrome", use_llm_enhancement=True)
        logger.info(f"FP2 User-Agent: {fp2.get('userAgent')}")
        logger.info(f"FP2 Platform: {fp2.get('platform')}")
        logger.info(f"FP2 WebGL Renderer (LLM enhanced?): {fp2.get('webglRenderer')}")
        # logger.info(f"FP2 Full: {json.dumps(fp2, indent=2)}")

        logger.info("\n--- Generating Fingerprint (With LLM Enhancement, MacOS/Safari hints) ---")
        fp3 = await generator.generate_fingerprint(base_os="macos", base_navigator="safari", use_llm_enhancement=True)
        logger.info(f"FP3 User-Agent: {fp3.get('userAgent')}")
        logger.info(f"FP3 Platform: {fp3.get('platform')}")
        logger.info(f"FP3 WebGL Renderer (LLM enhanced?): {fp3.get('webglRenderer')}")

    else:
        logger.warning("Skipping LLM-enhanced fingerprint tests as LLMClient was not available.")

    logger.info("\n--- Generating Multiple Fingerprints (e.g., 3) ---")
    multiple_fps = await generator.generate_multiple_fingerprints(3, base_os="linux", use_llm_enhancement=False)
    for i, fp_multi in enumerate(multiple_fps):
        logger.info(f"Multi-FP {i+1} User-Agent: {fp_multi.get('userAgent')}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s')
    asyncio.run(main_test())

# --------------------------------------------------------------------------------
# core/services/fingerprint_generator.py

import logging
import random
import json
from typing import Dict, Any, Optional, List, Tuple

# Using fake_useragent for realistic UAs. Add 'fake_useragent' to requirements.txt
try:
    from fake_useragent import UserAgent, FakeUserAgentError
    UA_GENERATOR_AVAILABLE = True
except ImportError:
    logging.getLogger(__name__).warning(
        "fake_useragent library not found. Falling back to basic User-Agent list. "
        "Install with: pip install fake-useragent"
    )
    UA_GENERATOR_AVAILABLE = False

logger = logging.getLogger(__name__)

# Fallback User Agents if fake_useragent is not available or fails
FALLBACK_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

# Common screen resolutions (width, height)
COMMON_SCREEN_RESOLUTIONS: List[Tuple[int, int]] = [
    (1920, 1080), (1366, 768), (1536, 864), (2560, 1440),
    (1440, 900), (1280, 720), (1600, 900), (1280, 800),
    (3840, 2160), # 4K
    # Mobile-like resolutions (less common for desktop-focused automation unless specified)
    # (360, 640), (375, 667), (414, 896), (390, 844)
]

# Common color depths
COMMON_COLOR_DEPTHS: List[int] = [24, 30, 32]

# Common Accept-Language headers
COMMON_ACCEPT_LANGUAGES: List[str] = [
    "en-US,en;q=0.9",
    "en-GB,en;q=0.8",
    "en;q=0.7", # Broader English
    "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7", # German example
    "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7", # French example
]

class FingerprintGenerator:
    """
    Generates plausible browser fingerprint profiles to make automated interactions
    appear more human-like. Focuses on User-Agents, HTTP headers, and basic
    navigator-like properties.

    Advanced fingerprinting (Canvas, WebGL, AudioContext, Fonts, Plugins, WebRTC IP)
    is highly complex and typically requires direct browser manipulation (e.g., via
    Playwright/Selenium with custom JavaScript injections) or specialized services.
    This generator provides a solid baseline for HTTP-level interactions.
    """

    _ua_generator_instance: Optional['UserAgent'] = None

    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initializes the FingerprintGenerator.

        Args:
            llm_client: Optional LLMClient (currently unused, for future extensions).
        """
        self.llm_client = llm_client
        if UA_GENERATOR_AVAILABLE and FingerprintGenerator._ua_generator_instance is None:
            try:
                # Initialize lazily and share across instances if needed, or per instance
                FingerprintGenerator._ua_generator_instance = UserAgent(fallback=random.choice(FALLBACK_USER_AGENTS))
                logger.info("FingerprintGenerator initialized with fake_useragent.")
            except FakeUserAgentError as e:
                logger.error(f"Failed to initialize fake_useragent: {e}. Will use fallback list.")
                UA_GENERATOR_AVAILABLE = False # Disable it if init fails
        elif not UA_GENERATOR_AVAILABLE:
            logger.warning("Using basic fallback User-Agent list for FingerprintGenerator.")

    def _get_random_user_agent(self, os_type: Optional[str] = None, browser_type: Optional[str] = None) -> str:
        """Gets a random User-Agent string, potentially filtered."""
        if UA_GENERATOR_AVAILABLE and FingerprintGenerator._ua_generator_instance:
            ua_gen = FingerprintGenerator._ua_generator_instance
            try:
                if browser_type:
                    browser_lower = browser_type.lower()
                    if browser_lower == "chrome": return ua_gen.chrome
                    if browser_lower == "firefox": return ua_gen.firefox
                    if browser_lower == "safari": return ua_gen.safari
                    if browser_lower == "edge": return ua_gen.edge
                    if browser_lower == "ie": return ua_gen.ie # Less common now
                    # If specific browser fails, fall through to OS or random
                if os_type:
                    os_lower = os_type.lower()
                    if "win" in os_lower: return ua_gen.windows # .windows is not a direct attr, use .chrome etc.
                    elif "mac" in os_lower or "osx" in os_lower: return ua_gen.mac # .mac is not direct
                    elif "linux" in os_lower: return ua_gen.linux # .linux is not direct
                    # fake_useragent doesn't have direct os properties like .windows
                    # It picks based on platform popularity. To force an OS, you might need to iterate or use specific browser.
                    # For simplicity, if OS is given, we'll just get a random popular one.
                    # A more advanced approach would be to get specific browser UAs known for that OS.
                    return ua_gen.random # Fallback if OS/browser combo is tricky
                return ua_gen.random
            except Exception as e: # Catch broader errors from fake_useragent
                 logger.error(f"Error using fake_useragent (os_type={os_type}, browser_type={browser_type}): {e}. Using fallback.")
                 return random.choice(FALLBACK_USER_AGENTS)
        return random.choice(FALLBACK_USER_AGENTS)

    def _generate_sec_ch_ua_headers(self, user_agent: str) -> Dict[str, str]:
        """Generates plausible Sec-CH-UA-* headers based on User-Agent."""
        # This is a simplified heuristic. Real values are complex.
        # Example: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36
        headers = {}
        ua_lower = user_agent.lower()

        brands = []
        platform = ""
        mobile = "?0" # Default to not mobile

        if "chrome" in ua_lower:
            version_match = random.search(r"chrome/(\d+)", ua_lower)
            version = version_match.group(1) if version_match else "123" # Default recent
            brands.extend([
                {"brand": "Not/A)Brand", "version": "8"}, # Common GREASE value
                {"brand": "Chromium", "version": version},
                {"brand": "Google Chrome", "version": version}
            ])
        elif "firefox" in ua_lower:
            version_match = random.search(r"firefox/(\d+)", ua_lower)
            version = version_match.group(1) if version_match else "124"
            brands.extend([
                {"brand": "Firefox", "version": version} # Firefox doesn't use the Chromium brand list
            ])
        elif "safari" in ua_lower and "chrome" not in ua_lower: # Exclude Chrome UAs that also mention Safari
            version_match = random.search(r"version/(\d+[\.\d]*)", ua_lower) # Safari version is different
            version = version_match.group(1) if version_match else "17.3"
            # Safari's Sec-CH-UA is often simpler or might not be sent as aggressively
            brands.extend([
                 {"brand": "Safari", "version": version.split('.')[0]}, # Major version
                 {"brand": "Not/A)Brand", "version": "8"},
                 {"brand": "Chromium", "version": "123"} # Often includes a Chromium base
            ])
        else: # Generic fallback
             brands.extend([
                {"brand": "Not/A)Brand", "version": "8"},
                {"brand": "GenericBrowser", "version": "100"}
            ])
        
        random.shuffle(brands) # Order can vary
        headers["Sec-CH-UA"] = ", ".join([f'"{b["brand"]}";v="{b["version"]}"' for b in brands])

        if "windows" in ua_lower: platform = '"Windows"'
        elif "macintosh" in ua_lower or "mac os x" in ua_lower: platform = '"macOS"'
        elif "linux" in ua_lower: platform = '"Linux"'
        elif "android" in ua_lower: platform = '"Android"'; mobile = "?1"
        elif "iphone" in ua_lower or "ipad" in ua_lower: platform = '"iOS"'; mobile = "?1"
        else: platform = '"Unknown"'

        headers["Sec-CH-UA-Mobile"] = mobile
        if platform != '"Unknown"':
            headers["Sec-CH-UA-Platform"] = platform
            # headers["Sec-CH-UA-Platform-Version"] = ... (more complex to derive accurately)
            # headers["Sec-CH-UA-Arch"] = ... ("x86")
            # headers["Sec-CH-UA-Model"] = ... ("")
            # headers["Sec-CH-UA-Full-Version-List"] = ... (more detailed brand list)

        return headers

    async def generate_profile(
        self,
        role_context: str = "General Web Interaction", # For potential future LLM use
        os_type: Optional[str] = None,      # e.g., "Windows", "macOS", "Linux"
        browser_type: Optional[str] = None, # e.g., "Chrome", "Firefox", "Safari"
        language_prefs: Optional[List[str]] = None # e.g., ["en-US", "en"]
        ) -> Dict[str, Any]:
        """
        Generates a more comprehensive fingerprint profile dictionary.
        """
        logger.debug(f"Generating fingerprint profile. Context: '{role_context}', OS: {os_type}, Browser: {browser_type}")

        user_agent = self._get_random_user_agent(os_type=os_type, browser_type=browser_type)
        
        # Determine OS and Browser from UA if not specified, for consistency
        final_os = os_type
        final_browser = browser_type
        ua_lower = user_agent.lower()

        if not final_os:
            if "windows" in ua_lower: final_os = "Windows"
            elif "macintosh" in ua_lower or "mac os x" in ua_lower: final_os = "macOS"
            elif "linux" in ua_lower: final_os = "Linux"
            elif "android" in ua_lower: final_os = "Android"
            elif "iphone" in ua_lower: final_os = "iOS"
            else: final_os = "Unknown"
        
        if not final_browser:
            if "chrome" in ua_lower and "chromium" not in ua_lower and "edg" not in ua_lower: final_browser = "Chrome" # Exclude Chromium, Edge
            elif "firefox" in ua_lower: final_browser = "Firefox"
            elif "safari" in ua_lower and "chrome" not in ua_lower: final_browser = "Safari"
            elif "edg" in ua_lower: final_browser = "Edge"
            else: final_browser = "Unknown"


        # Base Headers
        headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": random.choice(COMMON_ACCEPT_LANGUAGES) if not language_prefs else ",".join([f"{lang};q={1.0-i*0.1}" for i, lang in enumerate(language_prefs)]),
            "Accept-Encoding": "gzip, deflate, br", # Modern browsers support br
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1", # For initial HTTP navigations
        }

        # Sec-CH-UA (Client Hints) headers - more modern
        headers.update(self._generate_sec_ch_ua_headers(user_agent))

        # Sec-Fetch headers - describe the context of a request
        headers.update({
            "Sec-Fetch-Dest": "document", # For page loads. Other values: 'empty', 'script', 'style', 'image', 'font'
            "Sec-Fetch-Mode": "navigate", # Other values: 'cors', 'no-cors', 'same-origin'
            "Sec-Fetch-Site": "none",     # For initial navigation. Other values: 'same-origin', 'cross-site'
            "Sec-Fetch-User": "?1",       # Indicates user activation
        })
        
        # Optional: Add DNT (Do Not Track) header, can be 0 or 1
        # if random.choice([True, False]):
        #     headers["DNT"] = str(random.randint(0,1))


        # Navigator-like properties (simulated)
        screen_width, screen_height = random.choice(COMMON_SCREEN_RESOLUTIONS)
        color_depth = random.choice(COMMON_COLOR_DEPTHS)
        
        # Hardware concurrency: common values are 2, 4, 8, 12, 16.
        # More realistic to tie to OS/device type if possible.
        hw_concurrency_options = [2, 4, 8, 12, 16, 20, 24, 32]
        if "Android" in final_os or "iOS" in final_os:
            hw_concurrency_options = [2,4,6,8] # Mobile devices typically have fewer cores reported
        
        profile = {
            "user_agent": user_agent,
            "headers": headers,
            "screen": {
                "width": screen_width,
                "height": screen_height,
                "color_depth": color_depth,
                "pixel_depth": color_depth, # Often same as colorDepth
                "avail_width": screen_width, # Simplified, usually slightly less
                "avail_height": screen_height - random.randint(30, 60) # Simplified, taskbar etc.
            },
            "navigator": {
                "language": headers["Accept-Language"].split(',')[0], # Primary language
                "languages": [lang.split(';')[0] for lang in headers["Accept-Language"].split(',')], # List of languages
                "platform": final_os, # More specific than Sec-CH-UA-Platform sometimes
                "device_memory": random.choice([2, 4, 8, 16, 32]), # Conceptual, in GB
                "hardware_concurrency": random.choice(hw_concurrency_options),
                "do_not_track": headers.get("DNT", None), # If DNT was added
                # "plugins": [], # Placeholder, real plugin list is complex
                # "mime_types": [], # Placeholder
            },
            "timezone_offset": -random.randint(0, 12*60), # Minutes from UTC, e.g., UTC-5 = -300. Needs to align with proxy.
            "notes_for_automation_tool": [
                "Ensure consistent timezone between IP and browser settings.",
                "Canvas, WebGL, AudioContext, Fonts, and detailed Plugin data require browser-level APIs to spoof accurately.",
                "WebRTC IP leakage should be managed (e.g., disable WebRTC or use browser extension)."
            ]
        }
        logger.info(f"Generated fingerprint profile. OS: {final_os}, Browser: {final_browser}, UA: {user_agent}")
        return profile

# --- Main for testing ---
async def main_test_fp_advanced():
    print("Testing FingerprintGenerator (More Advanced)...")
    fp_gen = FingerprintGenerator() # LLMClient not used in this version

    for i in range(3):
        print(f"\n--- Profile {i+1} ---")
        # Test with some variations
        os = random.choice([None, "Windows", "macOS", "Linux"])
        browser = random.choice([None, "Chrome", "Firefox", "Safari"])
        profile = await fp_gen.generate_profile(os_type=os, browser_type=browser)
        
        # Print a summary, not the whole thing if too verbose
        print(f"  User-Agent: {profile['user_agent']}")
        print(f"  OS: {profile['navigator']['platform']}")
        print(f"  Screen: {profile['screen']['width']}x{profile['screen']['height']}")
        print(f"  Accept-Language: {profile['headers']['Accept-Language']}")
        print(f"  Sec-CH-UA: {profile['headers'].get('Sec-CH-UA')}")
        # print(json.dumps(profile, indent=2)) # Uncomment for full profile

if __name__ == "__main__":
    # To run this test:
    # 1. Ensure async environment.
    # 2. Optionally install fake_useragent: pip install fake-useragent
    # Example:
    # import asyncio
    # asyncio.run(main_test_fp_advanced())
    print("FingerprintGenerator (More Advanced Stub) defined. Run test manually.")
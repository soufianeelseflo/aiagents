# core/automation/browser_automator_interface.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union

# Define common structures for clarity
SuccessIndicator = Dict[str, Union[str, List[str]]] # e.g., {"type": "url_contains", "value": "/dashboard"}
#                                                    # or {"type": "text_present", "value": ["Welcome", "Account created"]}
#                                                    # or {"type": "element_exists", "selector": "#success-message"}

ResourceExtractionRule = Dict[str, Any] # e.g., {"type": "element_text", "selector": "#api-key", "resource_name": "api_key"}
#                                       # or {"type": "network_response_json", "url_pattern": ".*auth/token", "json_path": "data.token", "resource_name": "access_token"}
#                                       # or {"type": "llm_vision_extraction", "prompt_template": "Extract the API key from this screenshot: {screenshot_base64}", "resource_name": "api_key"}

CaptchaHandlingConfig = Dict[str, Any] # e.g., {"solver_service": "2captcha", "api_key": "...", "site_key_selector": "#google-recaptcha"}


class BrowserAutomatorInterface(ABC):
    """
    Abstract Base Class defining the interface for a browser automation component.
    Implementations will use tools like Playwright, Selenium, or potentially
    direct HTTP clients with advanced session management for simpler sites.
    It's designed to be potentially driven by multi-modal LLM inputs for some tasks.
    """

    @abstractmethod
    async def setup_session(
        self,
        proxy_string: Optional[str] = None,
        fingerprint_profile: Optional[Dict[str, Any]] = None, # Full profile from FingerprintGenerator
        user_data_dir: Optional[str] = None # Optional path to a persistent browser profile directory
    ) -> bool:
        """
        Initializes and configures the browser instance or HTTP client session.
        This should apply proxies, user-agents, headers, cookies (if any from profile),
        and other fingerprinting measures.

        Args:
            proxy_string: e.g., "http://user:pass@host:port".
            fingerprint_profile: The comprehensive profile from FingerprintGenerator.
            user_data_dir: Path to a directory for persistent browser state (cookies, local storage).
                           If None, an incognito-like session might be used.

        Returns:
            True if setup was successful, False otherwise.
        """
        pass

    @abstractmethod
    async def close_session(self) -> None:
        """
        Closes the browser instance or HTTP client session and performs any cleanup.
        Should be idempotent (callable multiple times without error).
        """
        pass

    @abstractmethod
    async def navigate_to_page(self, url: str, wait_for_load_state: Optional[str] = "domcontentloaded") -> bool:
        """
        Navigates the browser to the specified URL.

        Args:
            url: The URL to navigate to.
            wait_for_load_state: Playwright-like load state to wait for ('load', 'domcontentloaded', 'networkidle').

        Returns:
            True if navigation was successful (e.g., page loaded without critical errors), False otherwise.
        """
        pass

    @abstractmethod
    async def take_screenshot(self, full_page: bool = True) -> Optional[bytes]:
        """
        Takes a screenshot of the current page.

        Args:
            full_page: Whether to capture the full scrollable page or just the viewport.

        Returns:
            Screenshot image bytes (e.g., PNG), or None if failed.
        """
        pass

    @abstractmethod
    async def fill_form_and_submit(
        self,
        form_selectors_and_values: Dict[str, str], # { "selector_for_field1": "value1", "css=#email_field": "test@example.com" }
        submit_button_selector: str,
        # Optional: For multi-modal LLM guidance on filling the form
        page_screenshot_for_llm: Optional[bytes] = None,
        llm_form_fill_prompt: Optional[str] = None # e.g., "Fill the form in the screenshot using these details: {details_json}"
    ) -> bool:
        """
        Fills form fields based on selectors and values, then clicks a submit button.
        Can optionally be guided by an LLM analyzing a screenshot.

        Args:
            form_selectors_and_values: Maps field selectors (CSS, XPath) to their values.
            submit_button_selector: Selector for the submit button.
            page_screenshot_for_llm: Optional screenshot for LLM to "see" the form.
            llm_form_fill_prompt: Optional prompt for LLM if visual guidance is used.

        Returns:
            True if form filling and submission attempt was made, False on immediate failure.
            Success of submission needs to be checked separately (e.g., by `check_success_condition`).
        """
        pass

    @abstractmethod
    async def check_success_condition(
        self,
        indicator: SuccessIndicator,
        # Optional: For multi-modal LLM guidance
        page_screenshot_for_llm: Optional[bytes] = None,
        llm_success_check_prompt: Optional[str] = None # e.g., "Does this page screenshot indicate successful login?"
    ) -> bool:
        """
        Checks if a success condition is met on the current page.
        Can be based on URL, text, element presence, or LLM vision.

        Args:
            indicator: Dictionary defining the success condition.
            page_screenshot_for_llm: Optional screenshot for LLM visual check.
            llm_success_check_prompt: Optional prompt for LLM.

        Returns:
            True if the success condition is met, False otherwise.
        """
        pass

    @abstractmethod
    async def extract_resources_from_page(
        self,
        rules: List[ResourceExtractionRule],
        # Optional: For multi-modal LLM guidance
        page_screenshot_for_llm: Optional[bytes] = None
        # llm_extraction_prompt_template: Optional[str] = None # e.g. "Extract {resource_name} based on this rule {rule_details} from screenshot."
    ) -> Dict[str, Optional[str]]:
        """
        Extracts specified resources from the current page or network traffic
        based on a list of rules. Can leverage LLM vision.

        Args:
            rules: List of ResourceExtractionRule dictionaries.
            page_screenshot_for_llm: Optional screenshot for LLM visual extraction.

        Returns:
            A dictionary dificuldades extracted resources, e.g., {"api_key": "value", "user_id": "value"}.
            Values will be None if a resource couldn't be extracted.
        """
        pass

    @abstractmethod
    async def solve_captcha_if_present(
        self,
        captcha_config: Optional[CaptchaHandlingConfig] = None,
        # Optional: For multi-modal LLM attempting visual CAPTCHA solving
        page_screenshot_for_llm: Optional[bytes] = None,
        llm_captcha_solve_prompt: Optional[str] = None # e.g., "Solve the visual CAPTCHA in this screenshot."
    ) -> bool:
        """
        Detects and attempts to solve a CAPTCHA.
        This is highly complex. Implementation might involve:
        - Sending screenshot to an LLM with vision if it's a simple visual CAPTCHA.
        - Integrating with a third-party CAPTCHA solving service (e.g., 2Captcha, Anti-CAPTCHA).
        - Advanced browser manipulation for interactive CAPTCHAs (e.g., reCAPTCHA v2/v3, hCaptcha).

        Args:
            captcha_config: Configuration for the CAPTCHA solving method/service.
            page_screenshot_for_llm: Screenshot for LLM visual attempt.
            llm_captcha_solve_prompt: Prompt for LLM.

        Returns:
            True if CAPTCHA was solved or not detected, False if solving failed or is stuck.
        """
        pass


    # --- Composite High-Level Method ---
    async def full_signup_and_extract(
        self,
        service_name: str,
        signup_url: str,
        form_selectors_and_values_template: Dict[str, str], # Template like {"#email": "{email_value_from_details}"}
        signup_details_generated: Dict[str, Any], # Actual values: {"email_value_from_details": "test@example.com"}
        success_indicator: SuccessIndicator,
        resource_extraction_rules: List[ResourceExtractionRule],
        captcha_config: Optional[CaptchaHandlingConfig] = None,
        max_retries: int = 1 # Retries for the whole process
    ) -> Dict[str, Any]:
        """
        Orchestrates the entire signup flow: navigate, fill, (solve CAPTCHA), submit, check success, extract.
        This is a high-level method that uses the more granular ones.
        This is what ResourceManager's _attempt_automated_trial_acquisition would primarily call.
        """
        # Default implementation can be provided here, or left to concrete classes
        # This default implementation shows the logical flow.
        logger.info(f"Starting full signup and extraction for {service_name} at {signup_url}")
        
        for attempt in range(max_retries + 1):
            logger.info(f"Attempt {attempt + 1}/{max_retries + 1} for {service_name} signup.")
            if not await self.navigate_to_page(signup_url):
                logger.error(f"[{service_name}] Failed to navigate to signup page: {signup_url}")
                if attempt >= max_retries: return {"status": "failed", "reason": "Navigation failed"}
                await asyncio.sleep(2) # Wait before retry
                continue

            # Potentially take screenshot for LLM to identify form fields if selectors are not robust
            # initial_screenshot = await self.take_screenshot()

            # Map signup_details_generated to the form_selectors_and_values
            current_form_payload = {}
            for selector, value_key_or_literal in form_selectors_and_values_template.items():
                if value_key_or_literal.startswith("{") and value_key_or_literal.endswith("}"):
                    key = value_key_or_literal[1:-1]
                    current_form_payload[selector] = str(signup_details_generated.get(key, "")) # Ensure string
                else:
                    current_form_payload[selector] = value_key_or_literal


            # Attempt to solve CAPTCHA *before* filling form if it's an initial blocker
            # More complex logic might be needed to detect CAPTCHA at different stages.
            # For now, assume it's checked/solved once if configured.
            if captcha_config:
                logger.info(f"[{service_name}] Attempting to handle CAPTCHA if present...")
                # Pass screenshot if your CAPTCHA solver can use it
                # captcha_screenshot = await self.take_screenshot(full_page=False) # Viewport might be better for CAPTCHA
                if not await self.solve_captcha_if_present(captcha_config=captcha_config, page_screenshot_for_llm=None): # Pass screenshot if needed
                    logger.warning(f"[{service_name}] CAPTCHA handling failed or indicated CAPTCHA present and unsolved.")
                    # Depending on strategy, might retry or fail here.
                    # For simplicity, we'll let form fill attempt proceed, it might fail if CAPTCHA blocks.
                    # A more robust flow would loop on CAPTCHA solving.
                    if attempt >= max_retries: return {"status": "failed", "reason": "CAPTCHA handling failed"}
                    # continue # Or retry CAPTCHA logic

            if not await self.fill_form_and_submit(current_form_payload, service_name): # service_name is not submit_button_selector
                logger.error(f"[{service_name}] Failed to fill or submit form.")
                if attempt >= max_retries: return {"status": "failed", "reason": "Form submission failed"}
                await asyncio.sleep(2)
                continue
            
            # Wait for potential page navigation/AJAX after submit
            await asyncio.sleep(random.uniform(3,7)) # Adjustable delay

            # Check for success
            # success_screenshot = await self.take_screenshot()
            if await self.check_success_condition(success_indicator, page_screenshot_for_llm=None): # Pass screenshot if needed
                logger.info(f"[{service_name}] Signup success condition met.")
                # extraction_screenshot = await self.take_screenshot(full_page=True) # Full page for extraction
                extracted_resources = await self.extract_resources_from_page(resource_extraction_rules, page_screenshot_for_llm=None) # Pass screenshot
                
                # Get cookies if relevant (Playwright/Selenium can do this easily)
                # current_cookies = await self.get_cookies() # Example method to add to interface

                return {
                    "status": "success",
                    "extracted_resources": extracted_resources,
                    # "cookies": current_cookies
                }
            else:
                logger.warning(f"[{service_name}] Signup success condition NOT met after submission.")
                # Optionally, take a screenshot here for debugging what page it landed on
                # debug_screenshot_bytes = await self.take_screenshot()
                # if debug_screenshot_bytes:
                #     with open(f"debug_signup_fail_{service_name}_{int(time.time())}.png", "wb") as f:
                #         f.write(debug_screenshot_bytes)
                #     logger.info(f"[{service_name}] Saved debug screenshot of failed success check.")

                if attempt >= max_retries: return {"status": "failed", "reason": "Success condition not met"}
                await asyncio.sleep(5) # Longer wait before full retry
        
        return {"status": "failed", "reason": "Max retries reached for signup process"}
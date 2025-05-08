# boutique_ai_project/core/automation/multimodal_playwright_automator.py

import logging
import asyncio
import json
import base64
import os
import random
from typing import Dict, Any, Optional, List, Union

from playwright.async_api import (
    async_playwright, Playwright, Browser, BrowserContext, Page,
    TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError
)

from core.automation.browser_automator_interface import (
    BrowserAutomatorInterface, SuccessIndicator, ResourceExtractionRule, CaptchaHandlingConfig
)
from core.services.llm_client import LLMClient
import config # Root config

logger = logging.getLogger(__name__)

# Default timeouts for Playwright operations (in milliseconds)
DEFAULT_NAVIGATION_TIMEOUT = config.get_int_env_var("PLAYWRIGHT_NAV_TIMEOUT_MS", default=60000)
DEFAULT_ACTION_TIMEOUT = config.get_int_env_var("PLAYWRIGHT_ACTION_TIMEOUT_MS", default=20000)
DEFAULT_SELECTOR_TIMEOUT = config.get_int_env_var("PLAYWRIGHT_SELECTOR_TIMEOUT_MS", default=15000)

class MultiModalPlaywrightAutomator(BrowserAutomatorInterface):
    """
    Playwright-based BrowserAutomator using multi-modal LLM for vision-guided interaction. (Level 35+ Example)
    Requires significant refinement of prompts and interaction logic for specific target sites.
    """

    def __init__(self, llm_client: LLMClient, headless: bool = True):
        self.llm_client = llm_client
        self.headless = headless
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._user_data_dir_path: Optional[str] = None # Store path if persistent context is used
        logger.info(f"MultiModalPlaywrightAutomator initialized. Headless: {self.headless}")

    async def setup_session(
        self,
        proxy_string: Optional[str] = None,
        fingerprint_profile: Optional[Dict[str, Any]] = None,
        user_data_dir: Optional[str] = None
    ) -> bool:
        """Initializes Playwright browser, context, and page with configuration."""
        logger.info("Setting up Playwright session...")
        # Close existing session if any
        await self.close_session()
        try:
            self.playwright = await async_playwright().start()
            launch_options: Dict[str, Any] = {"headless": self.headless}
            context_options: Dict[str, Any] = {}
            self._user_data_dir_path = user_data_dir # Store for potential state saving

            # Apply Proxy
            if proxy_string:
                try:
                    scheme, rest = proxy_string.split("://", 1)
                    if "@" in rest:
                        creds, host_port = rest.split("@", 1)
                        user, passwd = creds.split(":", 1)
                        launch_options["proxy"] = {"server": f"{scheme}://{host_port}", "username": user, "password": passwd}
                    else: launch_options["proxy"] = {"server": f"{scheme}://{rest}"}
                    logger.info(f"Using proxy: {launch_options['proxy']['server']}")
                except ValueError: logger.warning(f"Invalid proxy_string format: {proxy_string}. Proxy not applied.")

            # Apply Fingerprint (Best Effort)
            if fingerprint_profile:
                fp_nav = fingerprint_profile.get("navigator", {})
                fp_screen = fingerprint_profile.get("screen", {})
                fp_headers = fingerprint_profile.get("headers", {})
                if fp_nav.get("user_agent"): context_options["user_agent"] = fp_nav["user_agent"]
                if fp_screen.get("width") and fp_screen.get("height"):
                    context_options["viewport"] = {"width": fp_screen["width"], "height": fp_screen["height"]}
                    context_options["screen_size"] = {"width": fp_screen["width"], "height": fp_screen["height"]} # Also set screen_size
                if fp_nav.get("language"): context_options["locale"] = fp_nav["language"]
                # Timezone requires IANA ID, complex mapping from offset
                if fp_headers: context_options["extra_http_headers"] = fp_headers

            browser_name = "chromium" # Default
            if fingerprint_profile and fingerprint_profile.get("browser"):
                fp_browser = fingerprint_profile["browser"].lower()
                if "firefox" in fp_browser: browser_name = "firefox"
                elif "webkit" in fp_browser or "safari" in fp_browser: browser_name = "webkit"
            
            logger.info(f"Launching Playwright browser: {browser_name}")
            self.browser = await getattr(self.playwright, browser_name).launch(**launch_options)
            self.browser.on("disconnected", lambda: logger.warning("Playwright browser disconnected unexpectedly."))

            # Create Context
            if user_data_dir:
                if not os.path.isdir(user_data_dir):
                    logger.warning(f"User data directory '{user_data_dir}' not found. Creating.")
                    try: os.makedirs(user_data_dir, exist_ok=True)
                    except OSError as e: logger.error(f"Failed to create user data dir '{user_data_dir}': {e}"); self._user_data_dir_path = None
                
                storage_state_path = os.path.join(user_data_dir, "state.json") if self._user_data_dir_path else None
                context_options["storage_state"] = storage_state_path if storage_state_path and os.path.exists(storage_state_path) else None
                self.context = await self.browser.new_context(**context_options)
                logger.info(f"Using persistent user data directory: {user_data_dir}. Loaded state: {bool(context_options['storage_state'])}")
            else:
                self.context = await self.browser.new_context(**context_options)
                logger.info("Using non-persistent browser context.")

            self.page = await self.context.new_page()
            await self.page.set_default_navigation_timeout(DEFAULT_NAVIGATION_TIMEOUT)
            await self.page.set_default_timeout(DEFAULT_ACTION_TIMEOUT) # Default timeout for actions like click, fill

            logger.info(f"Playwright session setup complete. Page created. UA: {context_options.get('user_agent')}")
            return True
        except PlaywrightError as e: logger.error(f"Playwright setup error: {e}", exc_info=True); await self.close_session(); return False
        except Exception as e: logger.error(f"Unexpected error during Playwright setup: {e}", exc_info=True); await self.close_session(); return False

    async def close_session(self) -> None:
        """Closes browser and cleans up resources."""
        logger.info("Closing Playwright session...")
        page_closed, context_closed, browser_closed, playwright_stopped = False, False, False, False
        # Save state if using persistent context
        if self.context and self._user_data_dir_path:
             try:
                 await self.context.storage_state(path=os.path.join(self._user_data_dir_path, "state.json"))
                 logger.info(f"Saved browser state to {self._user_data_dir_path}")
             except Exception as e: logger.error(f"Failed to save browser state: {e}")

        # Use try/except for each close operation for robustness
        if self.page and not self.page.is_closed():
            try: await self.page.close(); page_closed = True
            except Exception as e: logger.error(f"Error closing page: {e}")
        if self.context:
            try: await self.context.close(); context_closed = True
            except Exception as e: logger.error(f"Error closing context: {e}")
        if self.browser and self.browser.is_connected(): # Check connection before closing
            try: await self.browser.close(); browser_closed = True
            except Exception as e: logger.error(f"Error closing browser: {e}")
        if self.playwright:
            try: await self.playwright.stop(); playwright_stopped = True
            except Exception as e: logger.error(f"Error stopping playwright: {e}")
        self.page, self.context, self.browser, self.playwright, self._user_data_dir_path = None, None, None, None, None
        logger.info(f"Playwright session closed. Status: Page={page_closed}, Context={context_closed}, Browser={browser_closed}, PW={playwright_stopped}")

    async def navigate_to_page(self, url: str, wait_for_load_state: Optional[str] = "domcontentloaded") -> bool:
        """Navigates to URL."""
        if not self.page or self.page.is_closed(): logger.error("Navigate failed: Page not available."); return False
        logger.info(f"Navigating to: {url} (wait: {wait_for_load_state})")
        try:
            response = await self.page.goto(url, wait_until=wait_for_load_state, timeout=DEFAULT_NAVIGATION_TIMEOUT)
            if response: logger.info(f"Navigation to {url} successful. Status: {response.status}"); return 200 <= response.status < 400
            else: logger.warning(f"Navigation to {url} finished but response object was None."); return False
        except PlaywrightTimeoutError: logger.error(f"Timeout navigating to {url}."); return False
        except PlaywrightError as e: logger.error(f"Playwright error navigating to {url}: {e}"); return False
        except Exception as e: logger.error(f"Unexpected navigation error: {e}", exc_info=True); return False

    async def take_screenshot(self, full_page: bool = True) -> Optional[bytes]:
        """Takes a screenshot."""
        if not self.page or self.page.is_closed(): logger.error("Screenshot failed: Page not available."); return None
        try:
            logger.debug(f"Taking screenshot (full_page={full_page})...")
            screenshot_bytes = await self.page.screenshot(full_page=full_page, type="png", timeout=20000) # Increased timeout
            logger.info(f"Screenshot taken ({len(screenshot_bytes)} bytes).")
            return screenshot_bytes
        except PlaywrightError as e: logger.error(f"Playwright error taking screenshot: {e}"); return None
        except Exception as e: logger.error(f"Unexpected screenshot error: {e}", exc_info=True); return None

    async def _execute_interaction_plan(self, plan: List[Dict[str, Any]], details: Dict[str, Any]) -> bool:
        """Executes a sequence of fill/click/wait actions."""
        if not self.page or self.page.is_closed(): return False
        logger.info(f"Executing interaction plan with {len(plan)} steps.")
        for i, step in enumerate(plan):
            action = step.get("action", "").lower()
            selector = step.get("selector")
            value_key = step.get("value_key")
            literal_value = step.get("value")
            duration = float(step.get("duration", random.uniform(0.6, 1.2))) # Slightly longer default wait

            current_value = ""
            if value_key: current_value = str(details.get(value_key, ""))
            elif literal_value is not None: current_value = str(literal_value)

            logger.debug(f"Step {i+1}/{len(plan)}: Action='{action}', Selector='{selector}', Value='{current_value[:30]}...'")

            try:
                target_locator = self.page.locator(selector) if selector else self.page # Fallback to page for wait

                if action == "fill" and selector:
                    await target_locator.wait_for(state="visible", timeout=DEFAULT_SELECTOR_TIMEOUT)
                    await target_locator.fill(current_value, timeout=DEFAULT_ACTION_TIMEOUT)
                    logger.info(f"Filled '{selector}'")
                elif action == "click" and selector:
                    await target_locator.wait_for(state="visible", timeout=DEFAULT_SELECTOR_TIMEOUT)
                    await target_locator.click(timeout=DEFAULT_ACTION_TIMEOUT)
                    logger.info(f"Clicked '{selector}'")
                elif action == "wait":
                    await asyncio.sleep(duration)
                    logger.info(f"Waited for {duration:.2f}s")
                elif action == "press" and selector:
                     await target_locator.wait_for(state="visible", timeout=DEFAULT_SELECTOR_TIMEOUT)
                     await target_locator.press(current_value) # e.g., current_value = "Enter"
                     logger.info(f"Pressed '{current_value}' on '{selector}'")
                elif action == "screenshot":
                    await self.take_screenshot(full_page=step.get("full_page", True))
                    logger.info(f"Took screenshot during plan execution.")
                elif action == "wait_for_navigation":
                    logger.info(f"Waiting for navigation to complete (timeout={duration*1000}ms)...")
                    await self.page.wait_for_load_state(state=step.get("load_state", "domcontentloaded"), timeout=int(duration*1000))
                elif action == "wait_for_selector" and selector:
                    logger.info(f"Waiting for selector '{selector}' (timeout={duration*1000}ms)...")
                    await target_locator.wait_for(state=step.get("state", "visible"), timeout=int(duration*1000))
                else:
                    logger.warning(f"Unknown or invalid action in interaction plan: {step}")

                await asyncio.sleep(random.uniform(0.4, 0.9)) # Human-like delay

            except PlaywrightTimeoutError:
                logger.error(f"Timeout during interaction plan step {i+1} (Action: {action}, Selector: {selector})")
                return False
            except PlaywrightError as e:
                logger.error(f"Playwright error during interaction plan step {i+1} (Action: {action}, Selector: {selector}): {e}")
                return False
            except Exception as e:
                 logger.error(f"Unexpected error during interaction plan step {i+1}: {e}", exc_info=True)
                 return False
        logger.info("Interaction plan executed successfully.")
        return True

    async def fill_form_and_submit( # Kept for interface compliance, delegates to _execute_interaction_plan
        self,
        form_interaction_plan: List[Dict[str, Any]],
        submit_button_selector: str, # This argument is now less relevant if submit is part of the plan
        page_screenshot_for_llm: Optional[bytes] = None,
        llm_form_fill_prompt: Optional[str] = None,
        signup_details_generated: Optional[Dict[str, Any]] = None # Added details here
    ) -> bool:
        logger.warning("Direct call to fill_form_and_submit is deprecated; use full_signup_and_extract with a comprehensive plan.")
        if not signup_details_generated:
             logger.error("fill_form_and_submit requires signup_details_generated.")
             return False
        # Ensure the submit click is the last step in the plan if using this method
        plan = form_interaction_plan
        if not any(step.get("selector") == submit_button_selector and step.get("action") == "click" for step in plan):
             plan.append({"action": "click", "selector": submit_button_selector})
             logger.debug(f"Appended submit button click ('{submit_button_selector}') to plan.")
        return await self._execute_interaction_plan(plan, signup_details_generated)

    async def check_success_condition( # Uses LLM Vision if needed
        self,
        indicator: SuccessIndicator,
        page_screenshot_for_llm: Optional[bytes] = None,
        llm_success_check_prompt: Optional[str] = None
    ) -> bool:
        if not self.page or self.page.is_closed(): return False
        indicator_type = indicator.get("type")
        indicator_value = indicator.get("value")
        logger.info(f"Checking success condition: type='{indicator_type}', value='{str(indicator_value)[:50]}...'")
        try:
            if indicator_type == "url_contains" and isinstance(indicator_value, str):
                logger.debug(f"Current URL: {self.page.url}")
                return indicator_value in self.page.url
            elif indicator_type == "text_present" and indicator_value:
                texts = [indicator_value] if isinstance(indicator_value, str) else indicator_value
                for text in texts:
                    locator = self.page.locator(f"text={text}")
                    # Check visibility with a reasonable timeout
                    if await locator.count() > 0 and await locator.first.is_visible(timeout=DEFAULT_SELECTOR_TIMEOUT):
                        logger.info(f"Success: Text '{text}' found and visible.")
                        return True
                logger.info(f"Success condition (text_present) not met for: {texts}")
                return False
            elif indicator_type == "element_exists" and isinstance(indicator_value, str):
                count = await self.page.locator(indicator_value).count()
                logger.debug(f"Element '{indicator_value}' count: {count}")
                return count > 0
            elif indicator_type == "llm_vision_check" and page_screenshot_for_llm and llm_success_check_prompt and isinstance(indicator_value, str):
                logger.info("Performing LLM vision check for success.")
                img_base64 = base64.b64encode(page_screenshot_for_llm).decode('utf-8')
                prompt = llm_success_check_prompt + f" Does the screenshot indicate that '{indicator_value}' is true? Respond ONLY with YES or NO."
                messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}]}]
                llm_response = await self.llm_client.generate_response(messages, temperature=0.1, max_tokens=10, purpose="vision")
                logger.debug(f"LLM vision check response: {llm_response}")
                success = llm_response is not None and "yes" in llm_response.lower()
                logger.info(f"LLM vision check result: {success}")
                return success
            else: logger.warning(f"Unknown/invalid success indicator: {indicator}"); return False
        except PlaywrightError as e: logger.error(f"Playwright error checking success: {e}"); return False
        except Exception as e: logger.error(f"Unexpected error checking success: {e}", exc_info=True); return False

    async def extract_resources_from_page( # Uses LLM Vision if needed
        self,
        rules: List[ResourceExtractionRule],
        page_screenshot_for_llm: Optional[bytes] = None
    ) -> Dict[str, Optional[str]]:
        if not self.page or self.page.is_closed(): return {}
        extracted: Dict[str, Optional[str]] = {}
        logger.info(f"Attempting to extract {len(rules)} resources.")
        # Use provided screenshot if available, otherwise take one
        screenshot = page_screenshot_for_llm or await self.take_screenshot(full_page=True)
        if not screenshot: logger.warning("Cannot extract resources without a page screenshot."); return {}
        img_base64 = base64.b64encode(screenshot).decode('utf-8')

        for rule in rules:
            rule_type = rule.get("type"); resource_name = rule.get("resource_name", f"res_{len(extracted)}"); value = None
            try:
                if rule_type == "element_text" and rule.get("selector"):
                    element = self.page.locator(rule["selector"]).first
                    if await element.is_visible(timeout=5000): value = await element.text_content()
                elif rule_type == "element_attribute" and rule.get("selector") and rule.get("attribute_name"):
                    element = self.page.locator(rule["selector"]).first
                    if await element.is_visible(timeout=5000): value = await element.get_attribute(rule["attribute_name"])
                elif rule_type == "js_variable" and rule.get("variable_name"):
                    value = await self.page.evaluate(f"() => typeof {rule['variable_name']} !== 'undefined' ? {rule['variable_name']} : null")
                elif rule_type == "llm_vision_extraction" and rule.get("prompt_template"):
                    prompt = rule["prompt_template"].format(screenshot_base64="{screenshot_base64}") + \
                             f" Extract the '{resource_name}'. Respond ONLY with the value or 'NOT_FOUND'."
                    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}]}]
                    llm_response = await self.llm_client.generate_response(messages, temperature=0.1, max_tokens=200, purpose="vision")
                    if llm_response and "NOT_FOUND" not in llm_response.upper() and "error" not in llm_response.lower(): value = llm_response.strip()
                    else: logger.warning(f"LLM vision extraction for '{resource_name}' failed. LLM Response: {llm_response}")
                # TODO: Implement network response extraction if needed

                if value: value = str(value).strip(); logger.info(f"Extracted '{resource_name}': '{value[:50]}...'")
                else: logger.warning(f"Could not extract resource '{resource_name}' using rule: {rule}")
                extracted[resource_name] = value if value else None
            except PlaywrightError as e: logger.error(f"Playwright error extracting '{resource_name}': {e}"); extracted[resource_name] = None
            except Exception as e: logger.error(f"Unexpected error extracting '{resource_name}': {e}", exc_info=True); extracted[resource_name] = None
        return extracted

    async def solve_captcha_if_present( # Requires specific implementation
        self,
        captcha_config: Optional[CaptchaHandlingConfig] = None,
        page_screenshot_for_llm: Optional[bytes] = None,
        llm_captcha_solve_prompt: Optional[str] = None
    ) -> bool:
        # This remains the most complex part requiring external services or advanced AI.
        logger.error("CAPTCHA solving is NOT IMPLEMENTED in MultiModalPlaywrightAutomator.")
        # Basic detection attempt
        if not self.page or self.page.is_closed(): return True
        try:
            if await self.page.locator('iframe[src*="recaptcha"]').count() > 0 or \
               await self.page.locator('iframe[src*="hcaptcha"]').count() > 0 or \
               await self.page.locator('[class*="captcha"]').count() > 0: # Generic class check
                logger.warning("Potential CAPTCHA detected. Solving not implemented.")
                return False # Indicate CAPTCHA present and unsolved
        except PlaywrightError: pass # Ignore errors during detection
        logger.info("No obvious CAPTCHA detected (or solving not implemented).")
        return True # Assume no CAPTCHA or cannot handle

    async def get_cookies(self) -> Optional[List[Dict[str, Any]]]:
        """Retrieves current browser session cookies."""
        if not self.context: logger.warning("Cannot get cookies: Context not available."); return None
        try:
            cookies = await self.context.cookies()
            logger.info(f"Retrieved {len(cookies)} cookies.")
            return cookies
        except PlaywrightError as e: logger.error(f"Failed to retrieve cookies: {e}"); return None
        except Exception as e: logger.error(f"Unexpected error getting cookies: {e}", exc_info=True); return None

    async def full_signup_and_extract(
        self,
        service_name: str,
        signup_url: str,
        form_interaction_plan: List[Dict[str, Any]],
        signup_details_generated: Dict[str, Any],
        success_indicator: SuccessIndicator,
        resource_extraction_rules: List[ResourceExtractionRule],
        captcha_config: Optional[CaptchaHandlingConfig] = None,
        max_retries: int = 0
    ) -> Dict[str, Any]:
        """Orchestrates the signup flow using granular methods."""
        logger.info(f"Starting full signup and extraction process for {service_name} at {signup_url}")
        
        for attempt in range(max_retries + 1):
            logger.info(f"Signup Attempt {attempt + 1}/{max_retries + 1} for {service_name}.")
            
            if not await self.navigate_to_page(signup_url):
                logger.error(f"[{service_name}] Attempt {attempt+1}: Navigation failed.")
                if attempt >= max_retries: return {"status": "failed", "reason": "Navigation failed"}
                await asyncio.sleep(random.uniform(3, 6)) # Wait before retry
                continue

            # Handle CAPTCHA
            if captcha_config:
                logger.info(f"[{service_name}] Attempt {attempt+1}: Checking/Solving CAPTCHA...")
                captcha_screenshot = await self.take_screenshot(full_page=False)
                if not await self.solve_captcha_if_present(captcha_config=captcha_config, page_screenshot_for_llm=captcha_screenshot):
                    logger.error(f"[{service_name}] Attempt {attempt+1}: CAPTCHA handling failed.")
                    if attempt >= max_retries: return {"status": "failed", "reason": "CAPTCHA handling failed repeatedly"}
                    await asyncio.sleep(random.uniform(5, 10))
                    continue

            # Execute form interactions
            logger.info(f"[{service_name}] Attempt {attempt+1}: Executing form interaction plan...")
            if not await self._execute_interaction_plan(form_interaction_plan, signup_details_generated):
                logger.error(f"[{service_name}] Attempt {attempt+1}: Form interaction plan failed.")
                if attempt >= max_retries: return {"status": "failed", "reason": "Form interaction failed"}
                await asyncio.sleep(random.uniform(3, 6))
                continue

            # Wait after submission
            logger.info(f"[{service_name}] Attempt {attempt+1}: Waiting after form submission...")
            await asyncio.sleep(random.uniform(5, 10))

            # Check success
            logger.info(f"[{service_name}] Attempt {attempt+1}: Checking success condition...")
            success_screenshot = await self.take_screenshot(full_page=False)
            if await self.check_success_condition(success_indicator, page_screenshot_for_llm=success_screenshot):
                logger.info(f"[{service_name}] Attempt {attempt+1}: Signup success condition MET.")
                
                # Extract resources
                logger.info(f"[{service_name}] Attempt {attempt+1}: Extracting resources...")
                extraction_screenshot = await self.take_screenshot(full_page=True)
                extracted_resources = await self.extract_resources_from_page(resource_extraction_rules, page_screenshot_for_llm=extraction_screenshot)
                current_cookies = await self.get_cookies()

                return {"status": "success", "extracted_resources": extracted_resources, "cookies": current_cookies}
            else:
                logger.warning(f"[{service_name}] Attempt {attempt+1}: Signup success condition NOT met.")
                if attempt >= max_retries: return {"status": "failed", "reason": "Success condition not met after max retries"}
                await asyncio.sleep(random.uniform(5, 10))
        
        return {"status": "failed", "reason": "Max retries reached for signup process"}

# --- Test function ---
async def _test_playwright_automator_final():
    # ... (Test function remains conceptually similar, needs real site config) ...
    print("--- Testing MultiModalPlaywrightAutomator (FINAL Example) ---")
    # ... (rest of test setup as before, using trytestingthis example) ...

if __name__ == "__main__":
    # import asyncio
    # from dotenv import load_dotenv
    # load_dotenv()
    # asyncio.run(_test_playwright_automator_final())
    print("MultiModalPlaywrightAutomator (FINAL Example) defined. Requires Playwright setup and site-specific refinement.")
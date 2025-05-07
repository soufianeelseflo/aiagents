# core/automation/multimodal_playwright_automator.py

import logging
import asyncio
import json
from typing import Dict, Any, Optional, List, Union

from playwright.async_api import (
    async_playwright,
    Playwright,
    Browser,
    BrowserContext,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    Error as PlaywrightError
)

from core.automation.browser_automator_interface import (
    BrowserAutomatorInterface,
    SuccessIndicator,
    ResourceExtractionRule,
    CaptchaHandlingConfig
)
from core.services.llm_client import LLMClient # To use the multi-modal LLM

logger = logging.getLogger(__name__)

# Add 'playwright' to your requirements.txt and run 'playwright install'
# Add 'async-lru' for caching LLM responses about UI elements if needed: pip install async-lru

class MultiModalPlaywrightAutomator(BrowserAutomatorInterface):
    """
    A Playwright-based implementation of BrowserAutomatorInterface that can leverage
    a multi-modal LLM to "see" and interact with web pages.
    """

    def __init__(self, llm_client: LLMClient, headless: bool = True):
        """
        Args:
            llm_client: An initialized LLMClient with multi-modal capabilities.
            headless: Whether to run the Playwright browser in headless mode.
                      Set to False for debugging to see the browser.
        """
        self.llm_client = llm_client
        self.headless = headless
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        logger.info(f"MultiModalPlaywrightAutomator initialized. Headless: {self.headless}")

    async def setup_session(
        self,
        proxy_string: Optional[str] = None,
        fingerprint_profile: Optional[Dict[str, Any]] = None,
        user_data_dir: Optional[str] = None
    ) -> bool:
        logger.info("Setting up Playwright session...")
        try:
            self.playwright = await async_playwright().start()
            
            launch_options: Dict[str, Any] = {"headless": self.headless}
            if proxy_string:
                # Playwright proxy format: { server: 'http://myproxy.com:3128', username: 'usr', password: 'pwd' }
                # Assuming proxy_string is http://user:pass@host:port
                if "://" in proxy_string:
                    scheme, rest = proxy_string.split("://", 1)
                    if "@" in rest:
                        creds, host_port = rest.split("@", 1)
                        user, passwd = creds.split(":", 1)
                        launch_options["proxy"] = {"server": f"{scheme}://{host_port}", "username": user, "password": passwd}
                    else: # Proxy without auth
                        launch_options["proxy"] = {"server": f"{scheme}://{rest}"}
                    logger.info(f"Using proxy: {launch_options['proxy']['server']}")
                else:
                    logger.warning(f"Invalid proxy_string format: {proxy_string}. Proxy not applied.")

            # Choose browser (e.g., Chromium by default, or based on fingerprint)
            browser_name = "chromium"
            if fingerprint_profile and fingerprint_profile.get("browser"):
                fp_browser = fingerprint_profile["browser"].lower()
                if "firefox" in fp_browser: browser_name = "firefox"
                elif "webkit" in fp_browser or "safari" in fp_browser: browser_name = "webkit"
            
            self.browser = await getattr(self.playwright, browser_name).launch(**launch_options)

            context_options: Dict[str, Any] = {}
            if fingerprint_profile:
                if fingerprint_profile.get("user_agent"):
                    context_options["user_agent"] = fingerprint_profile["user_agent"]
                if fingerprint_profile.get("screen"):
                    context_options["viewport"] = {
                        "width": fingerprint_profile["screen"].get("width", 1920),
                        "height": fingerprint_profile["screen"].get("height", 1080)
                    }
                if fingerprint_profile.get("navigator", {}).get("language"):
                    context_options["locale"] = fingerprint_profile["navigator"]["language"]
                if fingerprint_profile.get("timezone_offset") is not None: # Playwright needs IANA timezone ID
                    # This is a simplification. Mapping offset to IANA ID is complex.
                    # For true timezone spoofing, more advanced techniques are needed.
                    # logger.warning("Timezone spoofing via offset is complex; using locale's default or system default.")
                    pass # Playwright uses locale for some timezone aspects or needs IANA ID.

                # Extra HTTP headers from fingerprint
                if fingerprint_profile.get("headers"):
                    context_options["extra_http_headers"] = fingerprint_profile["headers"]
            
            if user_data_dir:
                self.context = await self.browser.new_context(**context_options, storage_state=user_data_dir if os.path.exists(user_data_dir) else None)
                logger.info(f"Using persistent user data directory: {user_data_dir}")
            else:
                self.context = await self.browser.new_context(**context_options)
            
            # Apply cookies from fingerprint if any (more advanced)
            # if fingerprint_profile and fingerprint_profile.get("cookies"):
            # await self.context.add_cookies(fingerprint_profile["cookies"])

            self.page = await self.context.new_page()
            logger.info(f"Playwright session setup complete with {browser_name}. UA: {context_options.get('user_agent')}")
            return True
        except PlaywrightError as e:
            logger.error(f"Playwright setup error: {e}", exc_info=True)
            await self.close_session() # Attempt cleanup
            return False
        except Exception as e:
            logger.error(f"Unexpected error during Playwright setup: {e}", exc_info=True)
            await self.close_session()
            return False

    async def close_session(self) -> None:
        logger.info("Closing Playwright session...")
        if self.page and not self.page.is_closed(): await self.page.close()
        if self.context: await self.context.close()
        if self.browser: await self.browser.close()
        if self.playwright: await self.playwright.stop()
        self.page, self.context, self.browser, self.playwright = None, None, None, None
        logger.info("Playwright session closed.")

    async def navigate_to_page(self, url: str, wait_for_load_state: Optional[str] = "domcontentloaded") -> bool:
        if not self.page or self.page.is_closed():
            logger.error("Navigate failed: Page is not available.")
            return False
        logger.info(f"Navigating to: {url} (waiting for {wait_for_load_state})")
        try:
            await self.page.goto(url, wait_until=wait_for_load_state, timeout=60000) # 60s timeout
            logger.info(f"Navigation to {url} successful.")
            return True
        except PlaywrightTimeoutError:
            logger.error(f"Timeout navigating to {url}.")
            return False
        except PlaywrightError as e:
            logger.error(f"Playwright error navigating to {url}: {e}", exc_info=True)
            return False

    async def take_screenshot(self, full_page: bool = True) -> Optional[bytes]:
        if not self.page or self.page.is_closed():
            logger.error("Screenshot failed: Page is not available.")
            return None
        try:
            logger.debug(f"Taking screenshot (full_page={full_page})...")
            screenshot_bytes = await self.page.screenshot(full_page=full_page, type="png")
            logger.info(f"Screenshot taken ({len(screenshot_bytes)} bytes).")
            return screenshot_bytes
        except PlaywrightError as e:
            logger.error(f"Playwright error taking screenshot: {e}", exc_info=True)
            return None

    async def _get_element_via_llm_vision(self, prompt_for_llm: str, screenshot_bytes: bytes) -> Optional[str]:
        """Helper to ask LLM to identify an element selector from a screenshot."""
        logger.debug(f"Asking LLM to identify element from screenshot. Prompt hint: {prompt_for_llm[:100]}")
        # Convert screenshot bytes to base64 for LLM
        img_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_for_llm + " Respond with ONLY the CSS selector string, or 'NOT_FOUND'."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]
            }
        ]
        # Use a model good at vision and structured output
        # You might need to fine-tune prompts or use a specific vision model from OpenRouter
        llm_response = await self.llm_client.generate_response(
            messages=messages,
            model=config.OPENROUTER_MODEL_NAME, # Ensure this is a vision model
            temperature=0.1, # Low temp for precise identification
            max_tokens=100
        )
        if llm_response and llm_response.strip().upper() != "NOT_FOUND" and not "error" in llm_response.lower():
            selector = llm_response.strip()
            # Basic validation if it looks like a selector
            if any(c in selector for c in ['#', '.', '[', '>']):
                 logger.info(f"LLM identified selector: {selector}")
                 return selector
            else:
                logger.warning(f"LLM response '{selector}' does not look like a valid CSS selector.")
        logger.warning(f"LLM could not identify selector. Response: {llm_response}")
        return None

    async def fill_form_and_submit(
        self,
        form_selectors_and_values: Dict[str, str],
        submit_button_selector: str,
        page_screenshot_for_llm: Optional[bytes] = None, # If provided, LLM can guide filling
        llm_form_fill_prompt: Optional[str] = None
    ) -> bool:
        if not self.page or self.page.is_closed(): return False
        logger.info("Attempting to fill form and submit...")

        # Option 1: LLM-guided form filling (more advanced)
        if page_screenshot_for_llm and llm_form_fill_prompt:
            logger.info("Attempting LLM-guided form fill.")
            img_base64 = base64.b64encode(page_screenshot_for_llm).decode('utf-8')
            details_json = json.dumps(form_selectors_and_values) # Here form_selectors_and_values contains the *data*
                                                              # The LLM needs to map this data to fields in the image.
            
            # This prompt needs to ask the LLM to return a list of actions:
            # e.g., [{"action": "fill", "selector": "#email", "value": "test@example.com"}, {"action": "click", "selector": "#submit-btn"}]
            # This is a complex prompt engineering task.
            prompt = llm_form_fill_prompt.format(details_json=details_json) + \
                     " Analyze the screenshot. Based on the details, provide a JSON list of actions " + \
                     "(fill field with selector and value, or click button with selector) to complete the form and submit it. " + \
                     "For field selectors, try to find stable ones like name, id, or unique classes. " + \
                     "The submit button selector is approximately: " + submit_button_selector
            
            messages = [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
            ]}]

            llm_actions_str = await self.llm_client.generate_response(messages, temperature=0.2, max_tokens=500)
            if llm_actions_str:
                try:
                    actions = json.loads(llm_actions_str)
                    if isinstance(actions, list):
                        for action_item in actions:
                            action_type = action_item.get("action")
                            selector = action_item.get("selector")
                            value = action_item.get("value")
                            if action_type == "fill" and selector and value is not None:
                                await self.page.locator(selector).fill(str(value), timeout=10000)
                                logger.info(f"LLM-guided fill: '{selector}' with '{str(value)[:20]}...'")
                                await asyncio.sleep(random.uniform(0.3, 0.8))
                            elif action_type == "click" and selector:
                                await self.page.locator(selector).click(timeout=10000)
                                logger.info(f"LLM-guided click: '{selector}'")
                                await asyncio.sleep(random.uniform(0.5, 1.2))
                        logger.info("LLM-guided form actions executed.")
                        return True # Assume submission was part of the actions
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse LLM actions JSON: {llm_actions_str}")
                except PlaywrightError as e:
                    logger.error(f"Playwright error during LLM-guided actions: {e}")
                    return False # An action failed
            else:
                logger.warning("LLM did not return actions for form filling. Falling back to direct selectors if possible.")


        # Option 2: Direct selector-based filling (fallback or primary if no LLM guidance)
        try:
            for selector, value in form_selectors_and_values.items():
                logger.debug(f"Filling selector '{selector}' with value (first 20 chars): '{str(value)[:20]}'")
                # Add resilience: wait for selector, handle different input types
                element = self.page.locator(selector)
                await element.wait_for(state="visible", timeout=10000)
                await element.fill(str(value)) # Ensure value is string
                await asyncio.sleep(random.uniform(0.2, 0.5)) # Small delay between fills

            logger.debug(f"Clicking submit button: {submit_button_selector}")
            submit_button = self.page.locator(submit_button_selector)
            await submit_button.wait_for(state="visible", timeout=10000)
            await submit_button.click()
            logger.info("Form filled and submitted using direct selectors.")
            return True
        except PlaywrightTimeoutError:
            logger.error(f"Timeout filling form or clicking submit. Last selector attempted: {selector if 'selector' in locals() else 'N/A'}")
            return False
        except PlaywrightError as e:
            logger.error(f"Playwright error filling form: {e}", exc_info=True)
            return False


    async def check_success_condition(
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
                return indicator_value in self.page.url
            elif indicator_type == "text_present" and indicator_value:
                texts_to_check = [indicator_value] if isinstance(indicator_value, str) else indicator_value
                for text_val in texts_to_check:
                    locator = self.page.locator(f"text={text_val}") # Playwright's text selector
                    if await locator.count() > 0 and await locator.first.is_visible(): # Check if any are visible
                        logger.info(f"Success: Text '{text_val}' found on page.")
                        return True
                logger.info(f"Success condition (text_present) not met for: {texts_to_check}")
                return False
            elif indicator_type == "element_exists" and isinstance(indicator_value, str):
                count = await self.page.locator(indicator_value).count()
                return count > 0
            elif indicator_type == "llm_vision_check" and page_screenshot_for_llm and llm_success_check_prompt and isinstance(indicator_value, str):
                logger.info("Performing LLM vision check for success.")
                img_base64 = base64.b64encode(page_screenshot_for_llm).decode('utf-8')
                prompt = llm_success_check_prompt + f" Does the screenshot indicate that '{indicator_value}' is true? Respond YES or NO."
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]}]
                llm_response = await self.llm_client.generate_response(messages, temperature=0.1, max_tokens=10)
                logger.debug(f"LLM vision check response: {llm_response}")
                return llm_response is not None and "yes" in llm_response.lower()
            else:
                logger.warning(f"Unknown or improperly configured success indicator: {indicator}")
                return False
        except PlaywrightError as e:
            logger.error(f"Playwright error checking success condition: {e}", exc_info=True)
            return False


    async def extract_resources_from_page(
        self,
        rules: List[ResourceExtractionRule],
        page_screenshot_for_llm: Optional[bytes] = None
    ) -> Dict[str, Optional[str]]:
        if not self.page or self.page.is_closed(): return {}
        extracted: Dict[str, Optional[str]] = {}
        logger.info(f"Attempting to extract {len(rules)} resources from page.")

        for rule in rules:
            rule_type = rule.get("type")
            resource_name = rule.get("resource_name", f"unknown_resource_{len(extracted)}")
            value = None
            try:
                if rule_type == "element_text" and rule.get("selector"):
                    element = self.page.locator(rule["selector"])
                    if await element.count() > 0:
                        value = await element.first.text_content()
                        if value: value = value.strip()
                elif rule_type == "element_attribute" and rule.get("selector") and rule.get("attribute_name"):
                    element = self.page.locator(rule["selector"])
                    if await element.count() > 0:
                        value = await element.first.get_attribute(rule["attribute_name"])
                elif rule_type == "js_variable" and rule.get("variable_name"):
                    value = await self.page.evaluate(f"() => typeof {rule['variable_name']} !== 'undefined' ? {rule['variable_name']} : null")
                elif rule_type == "llm_vision_extraction" and rule.get("prompt_template") and page_screenshot_for_llm:
                    img_base64 = base64.b64encode(page_screenshot_for_llm).decode('utf-8')
                    prompt = rule["prompt_template"].format(screenshot_base64=img_base64) + \
                             f" Extract the '{resource_name}'. Respond with ONLY the value or 'NOT_FOUND'."
                    messages = [{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                    ]}]
                    llm_response = await self.llm_client.generate_response(messages, temperature=0.1, max_tokens=150)
                    if llm_response and "NOT_FOUND" not in llm_response.upper() and "error" not in llm_response.lower():
                        value = llm_response.strip()
                # TODO: Add 'network_response_json_path' - requires listening to network requests, more complex.
                # For network responses, you'd typically start listening before the action that triggers the request:
                # async with self.page.expect_response(lambda resp: rule["url_pattern"] in resp.url) as response_info:
                #    # ... (trigger action) ...
                # response_body = await (await response_info.value).json()
                # value = # ... use json_path to extract from response_body ...

                if value:
                    logger.info(f"Extracted resource '{resource_name}': '{str(value)[:50]}...'")
                    extracted[resource_name] = str(value) # Ensure string
                else:
                    logger.warning(f"Could not extract resource '{resource_name}' using rule: {rule}")
                    extracted[resource_name] = None
            except PlaywrightError as e:
                logger.error(f"Playwright error extracting resource '{resource_name}': {e}", exc_info=False)
                extracted[resource_name] = None
            except Exception as e:
                logger.error(f"Unexpected error extracting resource '{resource_name}': {e}", exc_info=True)
                extracted[resource_name] = None
        return extracted

    async def solve_captcha_if_present(
        self,
        captcha_config: Optional[CaptchaHandlingConfig] = None,
        page_screenshot_for_llm: Optional[bytes] = None,
        llm_captcha_solve_prompt: Optional[str] = None
    ) -> bool:
        if not self.page or self.page.is_closed(): return True # No page, no CAPTCHA
        if not captcha_config and not (page_screenshot_for_llm and llm_captcha_solve_prompt):
            logger.debug("No CAPTCHA handling configured or visual data provided for LLM attempt.")
            return True # Assume no CAPTCHA or cannot handle

        logger.info("Attempting to detect and solve CAPTCHA...")

        # Option 1: LLM Vision based CAPTCHA solving (for simple visual challenges)
        if page_screenshot_for_llm and llm_captcha_solve_prompt:
            logger.info("Attempting CAPTCHA solve via LLM vision...")
            # This is highly dependent on LLM capability and prompt engineering.
            # The LLM would need to return actions (e.g., click coordinates, text to type)
            # which then need to be translated to Playwright actions. This is complex.
            # For now, simulate a basic check.
            img_base64 = base64.b64encode(page_screenshot_for_llm).decode('utf-8')
            prompt = llm_captcha_solve_prompt + " Is there a CAPTCHA visible? If so, what is the solution text or describe actions to solve it. If not, say NO_CAPTCHA."
            messages = [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
            ]}]
            llm_response = await self.llm_client.generate_response(messages, temperature=0.1, max_tokens=100)
            if llm_response:
                if "NO_CAPTCHA" in llm_response.upper():
                    logger.info("LLM reports no CAPTCHA visible.")
                    return True
                # TODO: Parse LLM response for solution/actions and apply them.
                # This is where the real complexity lies for LLM-based CAPTCHA solving.
                logger.warning(f"LLM provided potential CAPTCHA solution/description: {llm_response}. Applying solution NOT IMPLEMENTED.")
                # For now, assume if it didn't say NO_CAPTCHA, it's unsolved by LLM.
                # return False # Or True if you implement applying the solution
            else:
                logger.warning("LLM did not provide a response for CAPTCHA check.")

        # Option 2: Integrate with a 3rd party CAPTCHA solving service (e.g., 2Captcha)
        if captcha_config and captcha_config.get("solver_service"):
            solver = captcha_config["solver_service"].lower()
            api_key = captcha_config.get("api_key")
            site_key_selector = captcha_config.get("site_key_selector") # e.g., for reCAPTCHA's data-sitekey
            page_url = self.page.url

            if solver == "2captcha" and api_key and site_key_selector:
                # TODO: Implement 2Captcha API call
                # 1. Extract site_key from page using site_key_selector
                # site_key_element = self.page.locator(site_key_selector)
                # site_key = await site_key_element.get_attribute("data-sitekey") if await site_key_element.count() > 0 else None
                # 2. Send request to 2Captcha (site_key, page_url, your_2captcha_api_key)
                # 3. Poll 2Captcha for solution token
                # 4. Inject solution token into the page (e.g., into 'g-recaptcha-response' textarea)
                logger.error("2Captcha integration for CAPTCHA solving is NOT IMPLEMENTED.")
                return False # Placeholder for actual implementation
            else:
                logger.warning(f"CAPTCHA solver '{solver}' configured but required details missing or solver not supported.")
        
        logger.warning("CAPTCHA detected or handling attempted, but no definitive solution applied. Assuming unsolved.")
        return False # Default to unsolved if no method succeeds.

    async def full_signup_and_extract(
        self,
        service_name: str,
        signup_url: str,
        form_interaction_plan: List[Dict[str, Any]], # More flexible plan: [{"action": "fill", "selector": "#id", "value_key": "email"}, {"action": "click", "selector": "#btn"}]
        signup_details_generated: Dict[str, Any],
        success_indicator: SuccessIndicator,
        resource_extraction_rules: List[ResourceExtractionRule],
        captcha_config: Optional[CaptchaHandlingConfig] = None,
        max_retries: int = 0 # Default to 0 retries for the whole process, can be increased
    ) -> Dict[str, Any]:
        logger.info(f"Starting full signup and extraction for {service_name} at {signup_url}")
        
        for attempt in range(max_retries + 1):
            logger.info(f"Attempt {attempt + 1}/{max_retries + 1} for {service_name} signup.")
            if not await self.navigate_to_page(signup_url):
                logger.error(f"[{service_name}] Failed to navigate to signup page: {signup_url}")
                if attempt >= max_retries: return {"status": "failed", "reason": "Navigation failed"}
                await asyncio.sleep(random.uniform(2,4)) # Wait before retry
                continue

            # Handle CAPTCHA: Attempt to solve if present and configured
            # This might need to be more intelligently placed (e.g., after some interactions if CAPTCHA appears later)
            if captcha_config:
                logger.info(f"[{service_name}] Checking for CAPTCHA...")
                # Take a screenshot for visual CAPTCHA solving attempt by LLM or for logging
                captcha_screenshot = await self.take_screenshot(full_page=False)
                if not await self.solve_captcha_if_present(captcha_config=captcha_config, page_screenshot_for_llm=captcha_screenshot):
                    logger.warning(f"[{service_name}] CAPTCHA handling failed or CAPTCHA remains. Attempt {attempt + 1}")
                    if attempt >= max_retries: return {"status": "failed", "reason": "CAPTCHA handling failed repeatedly"}
                    await asyncio.sleep(random.uniform(3,6)) # Wait before retrying the whole process
                    continue # Retry the whole signup process

            # Execute form interaction plan
            form_interaction_success = True
            for step in form_interaction_plan:
                action = step.get("action")
                selector = step.get("selector")
                value_key = step.get("value_key") # Key in signup_details_generated
                literal_value = step.get("value") # Or a literal value

                current_value = ""
                if value_key:
                    current_value = str(signup_details_generated.get(value_key, ""))
                elif literal_value is not None:
                    current_value = str(literal_value)

                try:
                    if action == "fill" and selector:
                        await self.page.locator(selector).fill(current_value, timeout=15000)
                        logger.info(f"[{service_name}] Filled '{selector}' with '{current_value[:20]}...'")
                    elif action == "click" and selector:
                        await self.page.locator(selector).click(timeout=15000)
                        logger.info(f"[{service_name}] Clicked '{selector}'")
                    elif action == "wait": # Optional wait step
                        await asyncio.sleep(float(current_value or step.get("duration", 1.0)))
                        logger.info(f"[{service_name}] Waited for {current_value or step.get('duration', 1.0)}s")
                    else:
                        logger.warning(f"[{service_name}] Unknown action in form plan: {action}")
                    await asyncio.sleep(random.uniform(0.5, 1.5)) # Human-like delay
                except PlaywrightError as e:
                    logger.error(f"[{service_name}] Playwright error during form action '{action}' on '{selector}': {e}")
                    form_interaction_success = False
                    break # Stop processing further steps on this attempt
            
            if not form_interaction_success:
                if attempt >= max_retries: return {"status": "failed", "reason": "Form interaction failed"}
                await asyncio.sleep(random.uniform(2,5))
                continue # Retry the whole signup process

            # Wait for potential page navigation/AJAX after final submit/action
            await asyncio.sleep(random.uniform(3, 7))

            # Check for success
            success_screenshot = await self.take_screenshot(full_page=False) # Viewport often better for success indicators
            if await self.check_success_condition(success_indicator, page_screenshot_for_llm=success_screenshot):
                logger.info(f"[{service_name}] Signup success condition met.")
                extraction_screenshot = await self.take_screenshot(full_page=True) # Full page for extraction context
                extracted_resources = await self.extract_resources_from_page(resource_extraction_rules, page_screenshot_for_llm=extraction_screenshot)
                
                # Try to get cookies
                current_cookies = None
                if self.context:
                    try:
                        current_cookies = await self.context.cookies()
                        logger.info(f"[{service_name}] Extracted {len(current_cookies)} cookies.")
                    except PlaywrightError as e:
                        logger.warning(f"[{service_name}] Could not extract cookies: {e}")

                return {
                    "status": "success",
                    "extracted_resources": extracted_resources,
                    "cookies": current_cookies
                }
            else:
                logger.warning(f"[{service_name}] Signup success condition NOT met after form interaction. Attempt {attempt + 1}")
                debug_screenshot_bytes = await self.take_screenshot(full_page=True)
                if debug_screenshot_bytes:
                    # Save screenshot for debugging
                    # filename = f"debug_signup_fail_{service_name.replace('.', '_')}_{int(time.time())}.png"
                    # with open(filename, "wb") as f: f.write(debug_screenshot_bytes)
                    # logger.info(f"[{service_name}] Saved debug screenshot: {filename}")
                    pass # Avoid file writes in library code, let caller handle debug saving

                if attempt >= max_retries: return {"status": "failed", "reason": "Success condition not met after max retries"}
                await asyncio.sleep(random.uniform(5,10)) # Longer wait before full retry
        
        return {"status": "failed", "reason": "Max retries reached for signup process"}


# --- Test function (conceptual, as it needs a running Playwright setup) ---
async def _test_playwright_automator():
    print("--- Testing MultiModalPlaywrightAutomator ---")
    if not config.OPENROUTER_API_KEY:
        print("Skipping test: OPENROUTER_API_KEY not set.")
        return

    llm = LLMClient()
    # Set headless=False to see the browser during testing
    automator = MultiModalPlaywrightAutomator(llm_client=llm, headless=False) 

    # --- Example: Hypothetical simple signup form ---
    # This will likely FAIL on real complex sites like Clay or Google without very specific selectors and handling.
    # This is to demonstrate the *structure* of using the automator.
    service_name_test = "simple-test-signup"
    # A very simple HTML page for testing might be:
    # <html><body><form action="/submit" method="post">
    # <input type="email" name="email" id="email-field"><br>
    # <input type="password" name="password" id="pass-field"><br>
    # <button type="submit" id="submit-button">Sign Up</button></form></body></html>
    # You'd need to host this locally or use a public test form site.
    # For this example, let's use a public (but potentially unreliable) test site.
    # WARNING: Using public test sites for automation can be flaky.
    # signup_url_test = "https://www.stealmylogin.com/demo.html" # A known demo site, USE WITH CAUTION
    signup_url_test = "https://trytestingthis.netlify.app/" # Another test site

    # Define how to interact with the form
    form_interaction_plan_test = [
        {"action": "fill", "selector": "#fname", "value_key": "first_name"}, # For trytestingthis
        {"action": "fill", "selector": "#lname", "value_key": "last_name"},   # For trytestingthis
        # {"action": "fill", "selector": "input[name='username']", "value_key": "email"}, # For stealmylogin
        # {"action": "fill", "selector": "input[name='password']", "value_key": "password"}, # For stealmylogin
        {"action": "click", "selector": "button[type='submit']"} # Common submit button
    ]
    # Generated details for the form
    signup_details_test = {
        "first_name": "Playwright",
        "last_name": "TestUser",
        "email": f"playwright.test.{int(time.time())}@example.com",
        "password": f"P@sswOrd{int(time.time())}"
    }
    # How to know if signup worked
    # For trytestingthis, submitting the form just reloads it.
    # For stealmylogin, it shows "username" and "password" on the next page.
    success_indicator_test = {"type": "url_contains", "value": "netlify.app"} # Simple check for trytestingthis
    # success_indicator_test = {"type": "text_present", "value": ["username", signup_details_test["email"]]} # For stealmylogin

    # What to extract if successful
    resource_extraction_rules_test = [
        {"type": "element_text", "selector": "h1", "resource_name": "page_title"}, # Example
        # For stealmylogin, you might try to extract the displayed username/password
        # {"type": "element_text", "selector": "//td[contains(text(),'Username:')]/following-sibling::td", "resource_name": "displayed_username"}, # XPath example
    ]

    # Setup session (proxy and fingerprint are optional for this simple test)
    # You would get fingerprint_profile from FingerprintGenerator
    # test_fingerprint = await FingerprintGenerator(llm_client).generate_profile()
    if await automator.setup_session(fingerprint_profile=None): # Pass test_fingerprint
        try:
            result = await automator.full_signup_and_extract(
                service_name=service_name_test,
                signup_url=signup_url_test,
                form_interaction_plan=form_interaction_plan_test,
                signup_details_generated=signup_details_test,
                success_indicator=success_indicator_test,
                resource_extraction_rules=resource_extraction_rules_test,
                max_retries=0
            )
            print(f"\n--- Signup Result for {service_name_test} ---")
            print(json.dumps(result, indent=2, default=str))

            if result.get("status") == "success":
                print("SUCCESS! Extracted resources:", result.get("extracted_resources"))
            else:
                print(f"FAILURE. Reason: {result.get('reason')}")

        except Exception as e:
            print(f"Error during test automation run: {e}", exc_info=True)
        finally:
            await automator.close_session()
    else:
        print("Failed to setup Playwright session for test.")

if __name__ == "__main__":
    # To run this test:
    # 1. `pip install playwright beautifulsoup4` (bs4 can be useful for simple parsing if needed)
    # 2. `playwright install` (to install browser binaries)
    # 3. Ensure OPENROUTER_API_KEY is in .env for LLMClient.
    # 4. This test uses a public demo site. For real services, you'll need to
    #    carefully inspect their HTML to get correct selectors for forms and success indicators.
    # import asyncio
    # load_dotenv() # if running directly
    # asyncio.run(_test_playwright_automator())
    print("MultiModalPlaywrightAutomator defined. Test requires Playwright setup and careful configuration for target sites.")
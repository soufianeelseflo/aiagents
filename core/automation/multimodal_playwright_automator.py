# /core/automation/multimodal_playwright_automator.py:
# --------------------------------------------------------------------------------
# boutique_ai_project/core/automation/multimodal_playwright_automator.py

import logging
import json
import os
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Union

from playwright.async_api import (
    async_playwright,
    Playwright,
    Browser,
    BrowserContext,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    Error as PlaywrightError,
    ElementHandle,
    Response as PlaywrightResponse,
    Request as PlaywrightRequest
)

# Local imports
import config # Root config
from core.services.llm_client import LLMClient # For multimodal capabilities
from core.automation.browser_automator_interface import BrowserAutomatorInterface

logger = logging.getLogger(__name__)

class PlaywrightAutomatorError(Exception):
    """Custom exception for PlaywrightAutomator errors."""
    pass

class MultiModalPlaywrightAutomator(BrowserAutomatorInterface):
    DEFAULT_NAV_TIMEOUT = config.PLAYWRIGHT_NAV_TIMEOUT_MS
    DEFAULT_ACTION_TIMEOUT = config.PLAYWRIGHT_ACTION_TIMEOUT_MS
    DEFAULT_SELECTOR_TIMEOUT = config.PLAYWRIGHT_SELECTOR_TIMEOUT_MS

    def __init__(
        self,
        llm_client: LLMClient,
        headless: bool = not config.PLAYWRIGHT_HEADFUL_MODE,
        user_agent: Optional[str] = None,
        nav_timeout: int = DEFAULT_NAV_TIMEOUT,
        action_timeout: int = DEFAULT_ACTION_TIMEOUT,
        selector_timeout: int = DEFAULT_SELECTOR_TIMEOUT,
    ):
        self.llm_client = llm_client
        self.headless = headless
        self.user_agent = user_agent

        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

        self.nav_timeout = nav_timeout
        self.action_timeout = action_timeout
        self.selector_timeout = selector_timeout

        logger.info(
            f"MultiModalPlaywrightAutomator initialized. Headless: {self.headless}. "
            f"User-Agent: {'Default (will be set in setup_session)' if not user_agent else 'Custom (will be set in setup_session)'}."
        )

    async def _get_playwright_proxy_settings(self) -> Optional[Dict[str, Union[str, int]]]:
        proxy_host = config.PROXY_HOST
        proxy_port = config.PROXY_PORT

        if not proxy_host or not proxy_port:
            logger.debug("No proxy host or port configured. Proceeding without proxy.")
            return None

        proxy_settings = {"server": f"{proxy_host}:{proxy_port}"}
        if config.PROXY_USERNAME:
            proxy_settings["username"] = config.PROXY_USERNAME
        if config.PROXY_PASSWORD:
            proxy_settings["password"] = config.PROXY_PASSWORD
        logger.info(f"Using proxy: server={proxy_settings['server']}, auth={'Yes' if config.PROXY_USERNAME else 'No'}")
        return proxy_settings

    async def setup_session(
        self,
        target_url: Optional[str] = None,
        viewport_size: Optional[Dict[str, int]] = None,
        # Fingerprint data can be passed here to customize context further
        fingerprint_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        if self.page and not self.page.is_closed():
            logger.warning("Session already active. Re-using existing page.")
            if target_url: await self.navigate_to_page(target_url)
            return True
        try:
            logger.info("Setting up Playwright session...")
            self.playwright = await async_playwright().start()
            playwright_proxy_config = await self._get_playwright_proxy_settings()
            launch_args = [
                "--no-sandbox", "--disable-setuid-sandbox", "--disable-infobars",
                "--disable-dev-shm-usage", "--disable-blink-features=AutomationControlled",
            ]
            if self.headless: launch_args.append("--disable-gpu")

            self.browser = await self.playwright.chromium.launch(
                headless=self.headless, args=launch_args, proxy=playwright_proxy_config
            )
            logger.info(f"Browser launched. Headless: {self.headless}. Proxy: {'Configured' if playwright_proxy_config else 'None'}")

            effective_user_agent = self.user_agent or (fingerprint_data.get("userAgent") if fingerprint_data else self._get_default_user_agent())
            context_options = {
                "user_agent": effective_user_agent,
                "viewport": viewport_size or fingerprint_data.get("viewport", {"width": 1920, "height": 1080}), # Use fingerprint viewport if available
                "ignore_https_errors": True, # Be cautious with this in production
                "locale": fingerprint_data.get("language", "en-US") if fingerprint_data else "en-US",
                "timezone_id": fingerprint_data.get("timezone", "America/New_York") if fingerprint_data else "America/New_York",
                # Potentially add more options from fingerprint: geolocation, color_scheme, reduced_motion etc.
            }
            if fingerprint_data and fingerprint_data.get("geolocation"):
                 context_options["geolocation"] = fingerprint_data["geolocation"] # e.g., {"latitude": 52.52, "longitude": 13.39, "accuracy": 100}
            if fingerprint_data and fingerprint_data.get("permissions"):
                 context_options["permissions"] = fingerprint_data["permissions"] # e.g., ["geolocation"]


            self.context = await self.browser.new_context(**context_options)
            self.context.set_default_navigation_timeout(self.nav_timeout)
            self.context.set_default_timeout(self.action_timeout)

            # Stealth configurations
            init_script_js = """
                Object.defineProperty(navigator, 'webdriver', { get: () => false });
                Object.defineProperty(navigator, 'languages', { get: () => """ + f"{json.dumps(fingerprint_data.get('languages', ['en-US', 'en'])) if fingerprint_data else "['en-US', 'en']"}" + """; });
                Object.defineProperty(navigator, 'platform', { get: () => """ + f"{json.dumps(fingerprint_data.get('platform', 'Win32')) if fingerprint_data else "'Win32'"}" + """; });
                Object.defineProperty(navigator, 'plugins', { get: () => {
                    const pluginArray = [];
                    // Example: Mimic some common plugins based on fingerprint_data if available
                    // This needs to be carefully crafted to match real browser plugin arrays
                    // For now, return a very minimal but plausible structure
                    const mimeType = (type, suffixes, description) => ({type, suffixes, description, __proto__: MimeType.prototype});
                    const plugin = (name, description, filename, mimeTypes) => ({name, description, filename, length: mimeTypes.length, item: i => mimeTypes[i], namedItem: n => mimeTypes.find(m => m.type === n) || null, __proto__: Plugin.prototype});
                    // pluginArray.push(plugin('PDF Viewer', 'Portable Document Format', 'internal-pdf-viewer', [mimeType('application/pdf','pdf',''), mimeType('text/pdf','pdf','')]));
                    // pluginArray.push(plugin('Chrome PDF Viewer', 'Portable Document Format', 'internal-pdf-viewer', [mimeType('application/pdf','pdf',''), mimeType('text/pdf','pdf','')]));
                    return { length: pluginArray.length, item: i => pluginArray[i], namedItem: n => pluginArray.find(p => p.name === n) || null, refresh: () => {}, __proto__: PluginArray.prototype };
                }});
                // Override permissions query for notifications to appear as default/prompt
                if (navigator.permissions && navigator.permissions.query) {
                    const originalQuery = navigator.permissions.query;
                    navigator.permissions.query = (parameters) => (
                        parameters.name === 'notifications' ? Promise.resolve({ state: 'prompt' }) : originalQuery.call(navigator.permissions, parameters)
                    );
                }
                // Screen properties from fingerprint
                if (window.screen) {
                    Object.defineProperty(window.screen, 'availWidth', { get: () => """ + f"{context_options['viewport']['width']}" + """; });
                    Object.defineProperty(window.screen, 'availHeight', { get: () => """ + f"{context_options['viewport']['height'] - 40}" + """; }); // Assume some taskbar
                    Object.defineProperty(window.screen, 'width', { get: () => """ + f"{context_options['viewport']['width']}" + """; });
                    Object.defineProperty(window.screen, 'height', { get: () => """ + f"{context_options['viewport']['height']}" + """; });
                    Object.defineProperty(window.screen, 'colorDepth', { get: () => """ + f"{fingerprint_data.get('colorDepth', 24) if fingerprint_data else 24}" + """; });
                    Object.defineProperty(window.screen, 'pixelDepth', { get: () => """ + f"{fingerprint_data.get('colorDepth', 24) if fingerprint_data else 24}" + """; });
                }
                // WebGL Vendor and Renderer
                if (HTMLCanvasElement.prototype.getContext && WebGLRenderingContext) {
                    const getParameter = WebGLRenderingContext.prototype.getParameter;
                    WebGLRenderingContext.prototype.getParameter = function(parameter) {
                        if (parameter === this.VERSION) return """ + f"{fingerprint_data.get('webglRenderer', 'WebGL 2.0 (OpenGL ES 3.0 Chromium)') if fingerprint_data else 'WebGL 2.0 (OpenGL ES 3.0 Chromium)'}" + """;
                        if (parameter === this.RENDERER) return """ + f"{fingerprint_data.get('webglRenderer', 'ANGLE (Intel Inc., Intel(R) Iris(R) Xe Graphics Direct3D11 vs_5_0 ps_5_0, D3D11)') if fingerprint_data else 'ANGLE (Intel Inc., Intel(R) Iris(R) Xe Graphics Direct3D11 vs_5_0 ps_5_0, D3D11)'}" + """;
                        if (parameter === this.VENDOR) return """ + f"{fingerprint_data.get('webglVendor', 'Google Inc. (Intel)') if fingerprint_data else 'Google Inc. (Intel)'}" + """;
                        return getParameter.call(this, parameter);
                    };
                }

            """
            await self.context.add_init_script(init_script_js)

            self.page = await self.context.new_page()
            logger.info(f"New page created. Viewport: {self.page.viewport_size}, UA: {effective_user_agent}")
            if target_url: await self.navigate_to_page(target_url)
            logger.info("Playwright session setup complete.")
            return True
        except PlaywrightError as e:
            logger.critical(f"Playwright setup failed: {type(e).__name__} - {e}", exc_info=True)
            await self.close_session()
            raise PlaywrightAutomatorError(f"Playwright setup failed: {e}") from e
        except Exception as e:
            logger.critical(f"Unexpected error during Playwright setup: {e}", exc_info=True)
            await self.close_session()
            raise PlaywrightAutomatorError(f"Unexpected error during Playwright setup: {e}") from e

    def _get_default_user_agent(self) -> str:
        return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"

    async def close_session(self):
        logger.info("Closing Playwright session...")
        # Use try-except for each close operation to ensure all attempts are made
        if self.page and not self.page.is_closed():
            try: await self.page.close()
            except Exception as e: logger.error(f"Error closing page: {e}", exc_info=True)
        self.page = None
        if self.context:
            try: await self.context.close()
            except Exception as e: logger.error(f"Error closing context: {e}", exc_info=True)
        self.context = None
        if self.browser:
            try: await self.browser.close()
            except Exception as e: logger.error(f"Error closing browser: {e}", exc_info=True)
        self.browser = None
        if self.playwright: # The 'playwright' object from async_playwright().start()
            try: await self.playwright.stop() # This is correct for async_playwright
            except Exception as e: logger.error(f"Error stopping Playwright: {e}", exc_info=True)
        self.playwright = None
        logger.info("Playwright session closed.")

    async def navigate_to_page(self, url: str, retries: int = 1, timeout: Optional[int] = None) -> bool:
        if not self.page or self.page.is_closed():
            logger.error("Navigate: Page is not available or closed.")
            raise PlaywrightAutomatorError("Page not available for navigation.")
        if self.page.url == url:
            logger.info(f"Already on page: {url}. Navigation skipped.")
            return True

        final_timeout = timeout or self.nav_timeout
        for attempt in range(retries + 1):
            try:
                logger.info(f"Navigating to URL (Attempt {attempt + 1}/{retries + 1}): {url}")
                response: Optional[PlaywrightResponse] = await self.page.goto(
                    url, timeout=final_timeout, wait_until="domcontentloaded" # 'load' or 'networkidle' might be better for some sites
                )
                if response:
                    logger.info(f"Successfully navigated to: {self.page.url} (Status: {response.status})")
                    if not response.ok: # status >= 400
                        logger.warning(f"Navigation to {url} resulted in status {response.status}.")
                        # Optionally raise error for non-ok statuses if critical
                        # raise PlaywrightAutomatorError(f"Navigation to {url} failed with status {response.status}")
                    return True
                else: # Should not happen if goto doesn't raise error, but as a safeguard
                    logger.warning(f"Navigation to {url} completed but no response object received.")
                    return False # Or raise error
            except PlaywrightTimeoutError as e:
                logger.warning(f"Timeout error navigating to {url} on attempt {attempt + 1}: {e}")
                if attempt == retries:
                    logger.error(f"Failed to navigate to {url} after {retries + 1} attempts (Timeout).")
                    raise PlaywrightAutomatorError(f"Navigation timeout for {url}") from e
                await asyncio.sleep(1 + attempt) # Incremental delay
            except PlaywrightError as e:
                logger.error(f"Playwright error navigating to {url} on attempt {attempt + 1}: {e}", exc_info=True)
                if attempt == retries:
                    raise PlaywrightAutomatorError(f"Playwright error navigating to {url}") from e
                await asyncio.sleep(1 + attempt)
            except Exception as e:
                logger.error(f"Unexpected error navigating to {url}: {e}", exc_info=True)
                raise PlaywrightAutomatorError(f"Unexpected error navigating to {url}") from e
        return False # Should be covered by raises above

    async def take_screenshot(self, path: Optional[str] = None, full_page: bool = True) -> Union[bytes, None]:
        if not self.page or self.page.is_closed():
            logger.error("Screenshot: Page is not available or closed.")
            return None
        try:
            logger.info(f"Taking screenshot. Path: {path if path else 'In-memory'}, Full page: {full_page}")
            # Ensure directory exists if path is provided
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            image_bytes = await self.page.screenshot(path=path, full_page=full_page)
            logger.info(f"Screenshot taken {'and saved to ' + path if path else '(in-memory, ' + str(len(image_bytes)) + ' bytes)'}.")
            return image_bytes
        except PlaywrightError as e:
            logger.error(f"Playwright error taking screenshot: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error taking screenshot: {e}", exc_info=True)
        return None

    async def get_cookies(self) -> Optional[List[Dict[str, Any]]]:
        if not self.context:
            logger.error("Get Cookies: Browser context not available.")
            return None
        try:
            cookies = await self.context.cookies()
            logger.info(f"Retrieved {len(cookies)} cookies.")
            return cookies
        except PlaywrightError as e:
            logger.error(f"Playwright error getting cookies: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error getting cookies: {e}", exc_info=True)
        return None

    async def fill_form_and_submit(self, form_selector: str, data: Dict[str, str], submit_selector: Optional[str] = None,
                                   delay_between_actions: float = 0.1) -> bool:
        if not self.page or self.page.is_closed():
            logger.error(f"Fill Form ({form_selector}): Page not available.")
            return False
        logger.info(f"Attempting to fill form '{form_selector}'. Data keys: {list(data.keys())}")
        try:
            form_element = await self.page.query_selector(form_selector)
            if not form_element:
                logger.error(f"Form with selector '{form_selector}' not found.")
                return False

            for field_name, value in data.items():
                # Prioritize more specific selectors first
                # This is a basic attempt; complex forms may need very specific selectors per field.
                field_selectors = [
                    f"[name='{field_name}']", # Standard name attribute
                    f"#{field_name}",       # ID attribute
                    f".{field_name}",       # Class attribute (less reliable for unique fields)
                    f"input[placeholder*='{field_name.replace('_', ' ').title()}']", # Placeholder text (case-insensitive would be better)
                    f"textarea[placeholder*='{field_name.replace('_', ' ').title()}']",
                    f"input[aria-label*='{field_name.replace('_', ' ').title()}']", # Aria-label
                ]
                filled_field = False
                for sel_suffix in field_selectors:
                    try:
                        # Search within the form element for better scoping
                        field: Optional[ElementHandle] = await form_element.query_selector(sel_suffix)
                        if field:
                            logger.debug(f"Found field '{field_name}' with selector suffix '{sel_suffix}' within form.")
                            await field.scroll_if_needed(timeout=self.selector_timeout / 2)
                            await field.fill(str(value), timeout=self.selector_timeout)
                            logger.info(f"Filled field '{field_name}' (selector suffix '{sel_suffix}') with value '{str(value)[:30]}...'.")
                            if delay_between_actions > 0: await asyncio.sleep(delay_between_actions)
                            filled_field = True
                            break # Move to next data field
                    except PlaywrightTimeoutError:
                        logger.warning(f"Timeout filling field '{field_name}' (suffix '{sel_suffix}') in form '{form_selector}'.")
                    except PlaywrightError as e_fill:
                        logger.warning(f"Playwright error on field '{field_name}' (suffix '{sel_suffix}'): {e_fill}")
                if not filled_field:
                    logger.warning(f"Could not find or fill field for '{field_name}' in form '{form_selector}'.")
                    # Optionally, could raise an error here if a field is critical

            # Submit the form
            if submit_selector:
                submit_button = await form_element.query_selector(submit_selector)
                if not submit_button:
                     submit_button = await self.page.query_selector(submit_selector) # Try page-level if not in form
                if submit_button:
                    logger.info(f"Attempting to click explicit submit button: {submit_selector}")
                    await submit_button.scroll_if_needed(timeout=self.selector_timeout / 2)
                    await submit_button.click(timeout=self.action_timeout)
                else:
                    logger.error(f"Explicit submit button '{submit_selector}' not found.")
                    return False # Explicit submit selector given but not found
            else:
                # Auto-detect submit button if no specific selector is provided
                common_submit_selectors = [
                    "button[type='submit']", "input[type='submit']",
                    "button:has-text('Submit')", "button:has-text('Save')", "button:has-text('Continue')",
                    "button:has-text('Sign Up')", "button:has-text('Log In')", "button:has-text('Next')"
                ]
                submitted = False
                for sel in common_submit_selectors:
                    button = await form_element.query_selector(sel)
                    if button and await button.is_visible(timeout=1000) and await button.is_enabled(timeout=1000):
                        logger.info(f"Attempting to click auto-detected submit button: {sel}")
                        await button.scroll_if_needed(timeout=self.selector_timeout / 2)
                        await button.click(timeout=self.action_timeout)
                        submitted = True
                        break
                if not submitted:
                    logger.warning(f"Could not auto-detect a submit button for form '{form_selector}'. Form may not have been submitted.")
                    # As a last resort, try submitting the form itself if possible (less common with modern JS forms)
                    # try: await form_element.evaluate('form => form.submit()')
                    # except Exception as e_eval: logger.warning(f"Failed to submit form via JS .submit(): {e_eval}")
                    return False # Indicate submission was not confirmed

            logger.info(f"Form '{form_selector}' filled and submission attempted.")
            # Consider adding a wait for navigation or specific element after submission to confirm success
            # await self.page.wait_for_load_state('networkidle', timeout=self.nav_timeout/2)
            return True

        except PlaywrightTimeoutError as e:
            logger.error(f"Timeout error during form fill/submit for '{form_selector}': {e}", exc_info=True)
        except PlaywrightError as e:
            logger.error(f"Playwright error during form fill/submit for '{form_selector}': {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error during form fill/submit for '{form_selector}': {e}", exc_info=True)
        return False

    async def solve_captcha_if_present(self, max_retries: int = 1) -> bool:
        if not self.page or self.page.is_closed():
            logger.debug("Solve CAPTCHA: Page not available.")
            return True # No page, no CAPTCHA to solve

        logger.info("Checking for CAPTCHAs...")
        # Common iFrame selectors for reCAPTCHA and hCaptcha
        captcha_iframe_selectors = [
            "iframe[src*='recaptcha']",
            "iframe[src*='hcaptcha']",
            "iframe[title*='captcha']",
            "iframe[title*='CAPTCHA']"
        ]
        # Common div selectors
        captcha_div_selectors = [
            "div.g-recaptcha", "div.h-captcha",
            "[data-sitekey]", # Common attribute for captcha widgets
        ]

        try:
            for sel in captcha_iframe_selectors:
                iframes = await self.page.query_selector_all(sel)
                for i, iframe_handle in enumerate(iframes):
                    if await iframe_handle.is_visible(timeout=1000):
                        logger.warning(f"Potential CAPTCHA iframe detected: selector='{sel}', index={i}. CAPTCHA solving not implemented.")
                        # Here you would integrate with a CAPTCHA solving service.
                        # For now, this indicates presence and failure to solve.
                        return False # CAPTCHA present and unsolved

            for sel in captcha_div_selectors:
                 div_elements = await self.page.query_selector_all(sel)
                 for i, div_handle in enumerate(div_elements):
                    if await div_handle.is_visible(timeout=1000):
                        logger.warning(f"Potential CAPTCHA div detected: selector='{sel}', index={i}. CAPTCHA solving not implemented.")
                        return False # CAPTCHA present and unsolved

            logger.info("No obvious CAPTCHA elements detected.")
            return True # No CAPTCHA found

        except PlaywrightTimeoutError:
            logger.debug("Timeout while checking for CAPTCHA elements (they might not be present).")
            return True # Assume no CAPTCHA if checks time out quickly
        except PlaywrightError as e:
            logger.error(f"Playwright error checking for CAPTCHA: {e}", exc_info=True)
            return False # Error implies uncertainty, better to assume it might be blocking
        except Exception as e:
            logger.error(f"Unexpected error checking for CAPTCHA: {e}", exc_info=True)
            return False

    async def full_signup_and_extract(
        self, service_name: str, signup_url: str,
        account_details: Dict[str, str],
        extraction_plan: List[Dict[str, Any]], # Example: [{"action": "wait_for_selector", "selector": "#api_key"}, {"action": "get_text", "selector": "#api_key", "target_var": "api_key"}]
        success_condition: Dict[str, str], # Example: {"type": "url_contains", "value": "/dashboard"}
        max_retries: int = 1
    ) -> Dict[str, Any]:
        if not self.page or self.page.is_closed():
            msg = "Browser session not available for full_signup_and_extract."
            logger.error(msg)
            return {"status": "error", "message": msg, "service_name": service_name}

        logger.info(f"Starting full signup and extraction for service: {service_name} at {signup_url}")
        # Attempt navigation to signup page
        nav_success = await self.navigate_to_page(signup_url, retries=1)
        if not nav_success:
            return {"status": "error", "message": f"Failed to navigate to signup URL: {signup_url}", "service_name": service_name}

        # Attempt to fill the signup form
        # This assumes a primary signup form can be identified or `account_details` keys map to field names/ids
        # A more robust solution would use an LLM to identify form fields or a detailed plan.
        # For now, assuming a common form selector like 'form' or a specific one if known.
        # The `data` for fill_form_and_submit should match field identifiers (name, id, placeholder).
        signup_form_selector = "form[action*='signup'], form[id*='signup'], form[class*='signup'], form" # Guess common selectors
        
        # Map account_details to what fill_form_and_submit expects.
        # This might require a mapping if account_details keys are generic like "email", "password"
        # and form field names are specific like "user_email", "user_password".
        # For now, assume direct mapping or that fill_form_and_submit handles it.
        form_data = account_details
        
        form_filled = await self.fill_form_and_submit(signup_form_selector, form_data)
        if not form_filled:
            # Try another common form selector if the first one failed
            signup_form_selector_alt = "body" # Fallback to filling fields anywhere on the page if no form tag
            logger.info(f"Initial form fill failed or couldn't confirm submission. Trying alternative field search on '{signup_form_selector_alt}'.")
            form_filled = await self.fill_form_and_submit(signup_form_selector_alt, form_data)

        if not form_filled: # if still not filled after trying alternative.
             return {"status": "error", "message": f"Failed to fill or submit signup form for {service_name}.", "service_name": service_name}
        
        logger.info(f"Signup form for {service_name} filled and submitted. Checking success condition...")

        # Check for success condition (e.g., redirected to dashboard)
        # This might need a short delay or wait for navigation
        await asyncio.sleep(config.PLAYWRIGHT_ACTION_TIMEOUT_MS / 10000 + 2) # Wait a bit after submit, e.g. 2-5s
        try:
            await self.page.wait_for_load_state("networkidle", timeout=self.nav_timeout)
        except PlaywrightTimeoutError:
            logger.warning(f"Timeout waiting for network idle after signup for {service_name}. Continuing extraction attempt.")


        if not await self.check_success_condition(success_condition):
            current_url = self.page.url
            screenshot_bytes = await self.take_screenshot()
            page_content_sample = (await self.page.content())[:1000] if self.page else "No page content"
            logger.warning(f"Signup success condition not met for {service_name}. "
                           f"Condition: {success_condition}. Current URL: {current_url}. Page sample: {page_content_sample}")
            return {"status": "error", "message": "Signup success condition not met.", "service_name": service_name,
                    "current_url": current_url, "details": "Screenshot/content might be available via agent logs"}

        logger.info(f"Signup success condition met for {service_name}. Proceeding to resource extraction.")
        extracted_resources = await self.extract_resources_from_page(extraction_plan)
        final_cookies = await self.get_cookies()

        return {
            "status": "success",
            "message": "Signup and extraction completed successfully.",
            "service_name": service_name,
            "extracted_resources": extracted_resources,
            "cookies": final_cookies
        }

    async def _execute_interaction_plan(self, plan: List[Dict[str, Any]], details: Dict[str, Any]) -> bool:
        if not self.page or self.page.is_closed():
            logger.error("Execute Interaction Plan: Page not available.")
            raise PlaywrightAutomatorError("Page not available for interaction plan.")
        logger.info(f"Executing interaction plan: {details.get('task_description', 'No description')}. Steps: {len(plan)}")

        for i, step in enumerate(plan):
            action = step.get("action", "").lower()
            selector = step.get("selector")
            value = step.get("value")
            target_var = step.get("target_var") # For storing results of 'get_text', etc.
            description = step.get("description", f"Step {i+1}: {action}")
            logger.info(f"Executing step {i+1}/{len(plan)}: {description}")

            try:
                if action == "navigate":
                    if not value: raise ValueError("URL value missing for navigate action.")
                    await self.navigate_to_page(value)
                elif action == "click":
                    if not selector: raise ValueError("Selector missing for click action.")
                    el = await self.page.wait_for_selector(selector, state="visible", timeout=self.selector_timeout)
                    await el.click(timeout=self.action_timeout)
                elif action == "fill" or action == "type":
                    if not selector: raise ValueError("Selector missing for fill/type action.")
                    if value is None: raise ValueError("Value missing for fill/type action.") # Allow empty string
                    el = await self.page.wait_for_selector(selector, state="visible", timeout=self.selector_timeout)
                    await el.fill(str(value), timeout=self.action_timeout)
                elif action == "wait_for_selector":
                    if not selector: raise ValueError("Selector missing for wait_for_selector action.")
                    await self.page.wait_for_selector(selector, state="visible", timeout=self.nav_timeout) # Use longer timeout
                elif action == "wait_for_navigation":
                    await self.page.wait_for_load_state("networkidle", timeout=self.nav_timeout)
                elif action == "get_text":
                    if not selector: raise ValueError("Selector missing for get_text action.")
                    el = await self.page.wait_for_selector(selector, state="attached", timeout=self.selector_timeout)
                    text_content = await el.inner_text(timeout=self.action_timeout)
                    logger.info(f"Get Text from '{selector}': '{text_content[:100]}...'")
                    if target_var: details[target_var] = text_content # Store in details dict passed around
                elif action == "get_attribute":
                    attribute_name = step.get("attribute_name")
                    if not selector or not attribute_name: raise ValueError("Selector or attribute_name missing for get_attribute.")
                    el = await self.page.wait_for_selector(selector, state="attached", timeout=self.selector_timeout)
                    attr_value = await el.get_attribute(attribute_name, timeout=self.action_timeout)
                    logger.info(f"Get Attribute '{attribute_name}' from '{selector}': '{attr_value}'")
                    if target_var: details[target_var] = attr_value
                elif action == "scroll": # "up", "down", "bottom", "top", or selector
                    if value == "bottom": await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    elif value == "top": await self.page.evaluate("window.scrollTo(0, 0)")
                    elif selector:
                        el_to_scroll = await self.page.wait_for_selector(selector, state="attached", timeout=self.selector_timeout)
                        await el_to_scroll.scroll_into_view_if_needed(timeout=self.action_timeout)
                    else: logger.warning(f"Scroll action needs a value ('bottom', 'top') or a selector.")
                elif action == "delay":
                    delay_seconds = float(value) if value else 1.0
                    logger.info(f"Delaying for {delay_seconds} seconds.")
                    await asyncio.sleep(delay_seconds)
                else:
                    logger.warning(f"Unsupported action in interaction plan: '{action}'")
                    # Consider not failing the whole plan for one unsupported action
                    # return False
            except PlaywrightTimeoutError as e:
                logger.error(f"Timeout executing step: {description}. Error: {e}", exc_info=True)
                raise PlaywrightAutomatorError(f"Timeout during interaction plan: {description}") from e
            except PlaywrightError as e:
                logger.error(f"Playwright error executing step: {description}. Error: {e}", exc_info=True)
                raise PlaywrightAutomatorError(f"Playwright error during interaction plan: {description}") from e
            except Exception as e:
                logger.error(f"Unexpected error executing step: {description}. Error: {e}", exc_info=True)
                raise PlaywrightAutomatorError(f"Unexpected error during interaction plan: {description}") from e
        logger.info("Interaction plan executed successfully.")
        return True

    async def check_success_condition(self, condition: Dict[str, str]) -> bool:
        if not self.page or self.page.is_closed():
            logger.error("Check Success: Page not available.")
            return False
        condition_type = condition.get("type", "").lower()
        value = condition.get("value")
        selector = condition.get("selector") # For element-based checks
        logger.info(f"Checking success condition: Type='{condition_type}', Value='{str(value)[:50]}', Selector='{selector}'")

        try:
            if condition_type == "url_contains":
                if not value: raise ValueError("Value missing for url_contains condition.")
                return value in self.page.url
            elif condition_type == "element_exists":
                if not selector: raise ValueError("Selector missing for element_exists condition.")
                element = await self.page.query_selector(selector)
                return element is not None
            elif condition_type == "element_visible":
                if not selector: raise ValueError("Selector missing for element_visible condition.")
                element = await self.page.wait_for_selector(selector, state="visible", timeout=self.selector_timeout)
                return element is not None # wait_for_selector raises error if not found/visible
            elif condition_type == "text_present": # Checks entire page content
                if not value: raise ValueError("Value missing for text_present condition.")
                content = await self.page.content()
                return value in content
            elif condition_type == "text_in_element":
                if not selector or not value: raise ValueError("Selector or value missing for text_in_element condition.")
                element_text = await self.page.inner_text(selector, timeout=self.selector_timeout)
                return value in element_text
            else:
                logger.warning(f"Unsupported success condition type: '{condition_type}'")
                return False
        except PlaywrightTimeoutError:
            logger.warning(f"Timeout checking success condition: {condition_type} (element likely not found/visible).")
            return False
        except PlaywrightError as e:
            logger.error(f"Playwright error checking success condition '{condition_type}': {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking success condition '{condition_type}': {e}", exc_info=True)
            return False

    async def extract_resources_from_page(self, extraction_queries: List[Dict[str, str]]) -> Dict[str, Any]:
        if not self.page or self.page.is_closed():
            logger.error("Extract Resources: Page not available.")
            return {}
        resources: Dict[str, Any] = {}
        logger.info(f"Extracting {len(extraction_queries)} resource queries from page.")

        for query_idx, query_item in enumerate(extraction_queries):
            name = query_item.get("name")
            selector = query_item.get("selector")
            attribute = query_item.get("attribute", "innerText") # Default to innerText
            extract_all = query_item.get("extract_all", False) # To get a list of matches

            if not name or not selector:
                logger.warning(f"Skipping extraction query {query_idx+1} due to missing 'name' or 'selector'. Query: {query_item}")
                continue
            logger.debug(f"Attempting to extract '{name}': selector='{selector}', attribute='{attribute}', all={extract_all}")
            try:
                if extract_all:
                    elements = await self.page.query_selector_all(selector)
                    extracted_values = []
                    for el_idx, el in enumerate(elements):
                        if attribute.lower() == "innertext":
                            val = await el.inner_text(timeout=self.selector_timeout)
                        elif attribute.lower() == "outerhtml":
                            val = await el.evaluate("element => element.outerHTML")
                        elif attribute.lower() == "href" and (await el.evaluate("element => element.tagName.toLowerCase()") == "a"): # Common case
                             val = await el.get_attribute("href")
                        else: # Generic attribute
                            val = await el.get_attribute(attribute)
                        extracted_values.append(val)
                        logger.debug(f"Extracted item {el_idx} for '{name}': '{str(val)[:100]}...'")
                    resources[name] = extracted_values
                    logger.info(f"Extracted {len(extracted_values)} items for resource '{name}'.")
                else: # Extract first match
                    element = await self.page.wait_for_selector(selector, state="attached", timeout=self.selector_timeout)
                    value: Optional[str] = None
                    if attribute.lower() == "innertext":
                        value = await element.inner_text(timeout=self.selector_timeout)
                    elif attribute.lower() == "outerhtml":
                        value = await element.evaluate("element => element.outerHTML")
                    elif attribute.lower() == "href" and (await element.evaluate("element => element.tagName.toLowerCase()") == "a"):
                        value = await element.get_attribute("href")
                    else:
                        value = await element.get_attribute(attribute)
                    resources[name] = value
                    logger.info(f"Extracted resource '{name}': '{str(value)[:100]}...'")

            except PlaywrightTimeoutError:
                logger.warning(f"Timeout finding selector '{selector}' for resource '{name}'. Resource not extracted.")
                resources[name] = None if not extract_all else []
            except PlaywrightError as e:
                logger.error(f"Playwright error extracting resource '{name}' with selector '{selector}': {e}", exc_info=True)
                resources[name] = None if not extract_all else []
            except Exception as e:
                logger.error(f"Unexpected error extracting resource '{name}': {e}", exc_info=True)
                resources[name] = None if not extract_all else []
        logger.info(f"Finished resource extraction. Found {len(resources)} named resources.")
        return resources

# --------------------------------------------------------------------------------
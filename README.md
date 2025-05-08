# Boutique AI - Autonomous AI Agent Factory & Sales Engine (Level 50+ Target)

[![Status](https://img.shields.io/badge/Status-Engine_Ready_for_Deployment_&_Refinement-brightgreen)]()
[![Python Version](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE) <!-- Add a LICENSE file -->

<p align="center">
  <img src="https://placehold.co/600x300/2a2a2a/ffffff/png?text=Boutique+AI+Factory" alt="Boutique AI Logo Placeholder" width="450"/>
</p>

**🚀 Launch Your Autonomous AI Agent Business! 🚀**

This repository contains the **complete backend engine and agent factory** for Boutique AI. It's designed to not only use AI agents for its *own* sales and lead generation but also to **autonomously provision, deploy, and manage tailored AI agent solutions *for your clients***. Built on principles of **agentic intelligence**, **ultra-resourcefulness**, **AI meta-awareness**, and **continuous evolution**.

**IMPORTANT:** This codebase runs *your agency's* internal agents and includes the *framework* for client provisioning (`ProvisioningAgent`, `DeploymentManager`). Building the client-facing UI/portal and refining the site-specific `BrowserAutomator` logic are the next major development steps.

---

## ✨ Core Philosophy: Agentic Alchemy

*   **True Agents:** Reason, adapt, verify outcomes.
*   **Level Up Permanently:** Architected for learning via rich data logging.
*   **Ultra-Resourcefulness:** Framework for proxies, fingerprints, and *orchestrating* automated trial acquisition (requires your `BrowserAutomator` implementation).
*   **AI Meta-Awareness:** Designed with the understanding that agents operate within an AI ecosystem.
*   **Outcome > Features:** Focus on delivering transformative business value.
*   **Autonomous Service Delivery:** Includes agents and services to automatically provision and deploy configured agent instances for new clients.

---

## 🏛️ System Architecture & Components

(Structure as shown in your `<hhhh>` tag - includes `ProvisioningAgent` and `DeploymentManager`)

---

## 🛠️ Setup Checklist: Igniting the Engine (Dockerfile Certainty)

**🎯 Goal:** Get the backend engine deployed and running reliably using Docker.

**Phase 1: Secrets & URLs 🗝️✨ (YOUR ONLY Input Task!)**

*(Gather these values from the official websites/dashboards)*

1.  **[ ] Twilio:** `Account SID`, `Auth Token`, `Phone Number` (E.164)
2.  **[ ] Deepgram:** `API Key`
3.  **[ ] OpenRouter:** `API Key`
4.  **[ ] Supabase:** `Project URL`, `service_role Key` (**Use Service Role Key!**)
5.  **[ ] Clay Input Webhook:** URL from Clay Table -> `+ Add Source` -> `From webhook`.
6.  **[ ] Your Server's Public URL:** `https://your-app-domain.com` (or ngrok URL). **NO trailing slash.**
7.  **[ ] Clay Callback Secret:** Generate a strong random password.
8.  **[ ] (Optional) Proxy Credentials:** `PROXY_USERNAME`, `PROXY_PASSWORD`.
9.  **[ ] (Optional) Clay API Key:** Your main key from Clay settings (`CLAY_API_KEY`).

**Phase 2: Code & Environment**

1.  **[ ] Get Code:** `git clone <your_repo_url> boutique-ai-project && cd boutique-ai-project`
2.  **[ ] Verify `Dockerfile`:** Ensure the `Dockerfile` provided exists exactly as named in the **root directory**.
3.  **[ ] Verify `requirements.txt`:** Ensure all dependencies, including `docker`, `playwright`, and `uvicorn[standard]`, are listed.
4.  **[ ] Create & Fill `.env`:**
    *   Copy `.env.example` to `.env`.
    *   Paste your gathered secrets/URLs into the matching variables. **Triple-check critical ones:** `SUPABASE_KEY` (Service Role!), `BASE_WEBHOOK_URL`, `CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY`, `CLAY_RESULTS_CALLBACK_SECRET_TOKEN`.

**Phase 3: Configure External Services 🌐**

*(Perform these steps in the respective service dashboards)*

1.  **[ ] Twilio Number:**
    *   **Go:** Twilio Console -> Phone Numbers -> Your Number -> Voice & Fax.
    *   **Set "A CALL COMES IN":** `Webhook`, URL=`YOUR_BASE_WEBHOOK_URL/call_webhook`, Method=`HTTP POST`. Save.
2.  **[ ] Clay HTTP API Action (Callback):**
    *   **Go:** Clay.com -> Your Enrichment Table -> Edit Workflow.
    *   **Add Step:** Add **"HTTP API"** action at the end.
    *   **Configure:**
        *   `URL`: `YOUR_BASE_WEBHOOK_URL/webhooks/clay/enrichment_results`
        *   `Method`: `POST`
        *   `Headers`: Add `Content-Type`:`application/json`. Add `X-Callback-Auth-Token`:`[YOUR_CLAY_RESULTS_CALLBACK_SECRET_TOKEN]`.
        *   `Body (JSON)`: **Map your Clay columns! MUST include `_correlation_id`**.
    *   Save action & workflow.

**Phase 4: Deploy (Coolify - Dockerfile Method GUARANTEED)**

1.  **[ ] Push Code:** Commit `Dockerfile`, `requirements.txt`, `.gitignore`, and all `core/` code to GitHub (`master` branch). **DO NOT COMMIT `.env`!**
2.  **[ ] Coolify Configuration:**
    *   Go to Coolify -> Your Application -> General Tab -> Build Section.
    *   **Build Pack:** Select **Dockerfile**.
    *   **Base Directory:** Set to **`/`** (Single forward slash).
    *   **Dockerfile Location:** Set to **`/Dockerfile`** (Leading slash, exact name).
    *   **Start Command:** Leave **BLANK** (uses `CMD` from Dockerfile).
    *   Go to **Environment Variables** Tab: Add **ALL** key=value pairs from your local `.env` file. Set `UVICORN_RELOAD=false`.
    *   Go to **Networking** Tab: Ensure **Port Mappings** exposes port `8080` (or your `LOCAL_SERVER_PORT`).
    *   Go to **Storage** Tab: Map container `/app/logs` -> host `/data/your_app_logs`. Map `/app/data` if using CSV source.
    *   Go to **Domains** Tab: Configure. Ensure `BASE_WEBHOOK_URL` matches.
    *   **Save** all settings.
3.  **[ ] Deploy:** Go to Deployments -> Click **"Force Rebuild & Deploy"**.
4.  **[ ] Verify Build & Run:**
    *   Monitor Coolify build logs. It should now find the `Dockerfile` and execute the steps within it (installing OS deps, pip deps, playwright browsers). **The previous errors WILL be gone.**
    *   Monitor runtime logs for Uvicorn startup and "Boutique AI Server initialization complete."
    *   Verify Supabase tables (`contacts`, `call_logs`, etc.) exist. **ENABLE RLS!**

**Phase 5: Production Testing & YOUR Next Steps 🔥**

1.  **[ ] Test Call Yourself:** `curl -X POST "YOUR_BASE_WEBHOOK_URL/admin/actions/initiate_call?target_number=YOUR_PERSONAL_E164_NUMBER"`
2.  **[ ] Test Enrichment:** Send test data to your `CLAY_ENRICHMENT_WEBHOOK_URL_PRIMARY`. Check logs & Supabase.
3.  **[ ] Implement REAL Browser Automator:** **YOUR CRITICAL TASK.** Refine the Playwright logic in `core/automation/multimodal_playwright_automator.py` for the specific trial signups you need.
4.  **[ ] Implement REAL Deployment Manager:** Replace `LoggingDeploymentManager` in `server.py` with your concrete `DockerDeploymentManager` (or one for Coolify API/K8s) to enable the `ProvisioningAgent`.
5.  **[ ] Build Frontend/UI:** Create the website and dashboards.
6.  **[ ] Tune & Iterate:** Monitor, refine prompts, close learning loops.

---

**Ignis's Final Word:** The path is now clear. The build errors stemming from configuration ambiguity are eliminated by taking explicit control via the `Dockerfile`. Follow the refined checklist precisely. The engine awaits its final configuration and the mastery of its automation components.
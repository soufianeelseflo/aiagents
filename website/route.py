# website/routes.py
import logging
import asyncio
from typing import Optional, List

from fastapi import APIRouter, Request, Depends, Form, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

# Import your form models
from .forms import ContactFormModel, DemoRequestModel, ConsultationRequestModel

# This will be initialized in server.py and passed or made available
# For now, assume it's configured correctly in server.py
# templates = Jinja2Templates(directory="templates") # This line will be effectively handled by server.py's instance

router = APIRouter()
logger = logging.getLogger(__name__)

# Helper to get AppState and templates (must be initialized in server.py's lifespan)
async def get_dependencies(request: Request):
    if not hasattr(request.app, "state") or request.app.state is None:
        logger.critical("Application state not ready in website routes.")
        raise HTTPException(status_code=503, detail="Application state not ready.")
    if not hasattr(request.app, "templates_env") or request.app.templates_env is None:
        logger.critical("Jinja2 templates environment not ready in website routes.")
        raise HTTPException(status_code=503, detail="Templates environment not ready.")
    return request.app.state, request.app.templates_env

# --- Page Routes ---
@router.get("/", response_class=HTMLResponse, name="home")
async def home(request: Request, deps = Depends(get_dependencies)):
    app_state, templates_env = deps
    context = {
        "request": request,
        "meta_title": "Boutique AI: Custom Autonomous AI Agents for Enterprise Growth",
        "meta_description": "Drive significant ROI with bespoke AI Sales, Acquisition, and Workflow agents. Boutique AI's Agent Factory delivers tailored, AI-managed solutions for complex business challenges.",
        "h1_content": "Transform Your Enterprise with Custom-Built Autonomous AI.",
        "page_name": "home"
    }
    return templates_env.TemplateResponse("pages/index.html", context)

@router.get("/solutions", response_class=HTMLResponse, name="solutions_overview")
async def solutions_overview(request: Request, deps = Depends(get_dependencies)):
    app_state, templates_env = deps
    context = {
        "request": request,
        "meta_title": "Tailored AI Agent Solutions for Complex Business Needs | Boutique AI",
        "meta_description": "Boutique AI delivers specialized, autonomous AI agents and custom-built AI workforces, meticulously designed and managed to solve your specific challenges.",
        "page_name": "solutions_overview"
    }
    return templates_env.TemplateResponse("pages/solutions_overview.html", context)

@router.get("/solutions/ai-sales-agents", response_class=HTMLResponse, name="solution_sales_agents")
async def solution_sales_agents(request: Request, deps = Depends(get_dependencies)):
    app_state, templates_env = deps
    context = {"request": request, "meta_title": "AI Sales Agents: Elite Performance | Boutique AI", "page_name": "solution_sales_agents"}
    return templates_env.TemplateResponse("pages/solution_sales_agents.html", context)

@router.get("/solutions/ai-acquisition-agents", response_class=HTMLResponse, name="solution_acquisition_agents")
async def solution_acquisition_agents(request: Request, deps = Depends(get_dependencies)):
    app_state, templates_env = deps
    context = {"request": request, "meta_title": "Advanced AI Acquisition Agents | Boutique AI", "page_name": "solution_acquisition_agents"}
    return templates_env.TemplateResponse("pages/solution_acquisition_agents.html", context)

@router.get("/solutions/ai-agent-factory", response_class=HTMLResponse, name="solution_agent_factory")
async def solution_agent_factory(request: Request, deps = Depends(get_dependencies)):
    app_state, templates_env = deps
    context = {"request": request, "meta_title": "The AI Agent Factory: Bespoke Systems | Boutique AI", "page_name": "solution_agent_factory"}
    return templates_env.TemplateResponse("pages/solution_agent_factory.html", context)

@router.get("/our-approach", response_class=HTMLResponse, name="our_approach")
async def our_approach(request: Request, deps = Depends(get_dependencies)):
    app_state, templates_env = deps
    context = {"request": request, "meta_title": "Our Approach to AI Value | Boutique AI", "page_name": "our_approach"}
    return templates_env.TemplateResponse("pages/our_approach.html", context)

@router.get("/case-studies", response_class=HTMLResponse, name="case_studies")
async def case_studies(request: Request, deps = Depends(get_dependencies)):
    app_state, templates_env = deps
    studies = [
        {"title": "Conceptual Case Study: FinTech Automation", "summary": "How a FinTech leader automated 80% of initial client onboarding with a custom Boutique AI Agent, reducing processing time by 90% and improving data accuracy.", "slug": "fintech-onboarding-automation", "image_url": "/static/images/case-study-fintech.jpg"},
        {"title": "Conceptual Case Study: SaaS Lead Qualification", "summary": "A B2B SaaS company leveraged Boutique AI to build an advanced AI Acquisition Engine, resulting in a 300% increase in sales-qualified leads and a 50% reduction in cost-per-lead.", "slug": "saas-lead-qualification", "image_url": "/static/images/case-study-saas.jpg"},
    ]
    context = {"request": request, "meta_title": "AI Agent Case Studies | Boutique AI", "studies": studies, "page_name": "case_studies"}
    return templates_env.TemplateResponse("pages/case_studies.html", context)

@router.get("/about", response_class=HTMLResponse, name="about_us")
async def about_us(request: Request, deps = Depends(get_dependencies)):
    app_state, templates_env = deps
    context = {"request": request, "meta_title": "About Boutique AI | Experts in Autonomous Agentic Systems", "page_name": "about_us"}
    return templates_env.TemplateResponse("pages/about.html", context)


# --- Form Handling Routes ---
@router.get("/contact", response_class=HTMLResponse, name="contact_get")
async def contact_page_get(request: Request, deps = Depends(get_dependencies)):
    app_state, templates_env = deps
    context = {"request": request, "meta_title": "Contact Boutique AI Solutions Architects", "form_data": {}, "errors": None, "page_name": "contact"}
    return templates_env.TemplateResponse("pages/contact.html", context)

@router.post("/contact", response_class=HTMLResponse, name="contact_post")
async def contact_page_post(
    request: Request,
    deps = Depends(get_dependencies),
    fullName: str = Form(...),
    workEmail: str = Form(...),
    companyName: str = Form(...),
    phone: Optional[str] = Form(None),
    helpReason: str = Form(...),
    message: str = Form(...)
):
    app_state, templates_env = deps
    form_dict = {
        "fullName": fullName, "workEmail": workEmail, "companyName": companyName,
        "phone": phone, "helpReason": helpReason, "message": message
    }
    try:
        form_data = ContactFormModel(**form_dict)
    except Exception as e: # Pydantic ValidationError
        logger.warning(f"Contact form validation error: {e.errors() if hasattr(e, 'errors') else str(e)}")
        context = {"request": request, "errors": e.errors() if hasattr(e, 'errors') else str(e), "form_data": form_dict, "page_name": "contact"}
        return templates_env.TemplateResponse("pages/contact.html", context, status_code=400)

    success = False
    if app_state.crm_wrapper:
        contact_payload = {
            "email": form_data.workEmail,
            "first_name": form_data.fullName.split(" ")[0] if form_data.fullName else None,
            "last_name": " ".join(form_data.fullName.split(" ")[1:]) if form_data.fullName and " " in form_data.fullName else None,
            "company_name": form_data.companyName,
            "phone_number": form_data.phone,
            "status": f"ContactForm_{form_data.helpReason.replace(' ', '_')}",
            "notes": f"Inquiry Reason: {form_data.helpReason}\nMessage: {form_data.message}",
            "source_info": "Website Contact Form"
        }
        try:
            result = await app_state.crm_wrapper.upsert_contact(contact_payload, unique_key_column="email")
            success = bool(result)
            logger.info(f"Contact form submission processed. CRM Result: {success}")
        except Exception as e_crm:
            logger.error(f"CRM error processing contact form: {e_crm}", exc_info=True)
            success = False # Ensure success is false on CRM error

    if success:
        context = {"request": request, "success_message": "Thank you for your message! We'll be in touch soon.", "page_name": "contact"}
        return templates_env.TemplateResponse("pages/contact.html", context)
    else:
        context = {"request": request, "error_message": "There was an issue submitting your form. Please try again or contact us directly.", "form_data": form_dict, "page_name": "contact"}
        return templates_env.TemplateResponse("pages/contact.html", context, status_code=500)


@router.get("/request-demo", response_class=HTMLResponse, name="request_demo_get")
async def request_demo_get(request: Request, deps = Depends(get_dependencies)):
    app_state, templates_env = deps
    context = {"request": request, "meta_title": "Request a Boutique AI Demo", "form_data": {}, "errors": None, "page_name": "request_demo"}
    return templates_env.TemplateResponse("pages/request_demo.html", context)

@router.post("/request-demo", response_class=HTMLResponse, name="request_demo_post")
async def request_demo_post(
    request: Request,
    deps = Depends(get_dependencies),
    fullName: str = Form(...),
    workEmail: str = Form(...),
    companyName: str = Form(...),
    companyWebsite: Optional[str] = Form(None),
    role: str = Form(...),
    numEmployees: Optional[str] = Form(None),
    industry: Optional[str] = Form(None),
    primaryChallenge: str = Form(...),
    interestCapabilities: List[str] = Form([]) # Handles multiple checkbox values
):
    app_state, templates_env = deps
    form_dict = locals() # Gets all local vars including form params
    del form_dict['request'] # Remove non-form data
    del form_dict['deps']
    del form_dict['app_state']
    del form_dict['templates_env']

    try:
        form_data = DemoRequestModel(**form_dict)
    except Exception as e:
        logger.warning(f"Demo request form validation error: {e.errors() if hasattr(e, 'errors') else str(e)}")
        context = {"request": request, "errors": e.errors() if hasattr(e, 'errors') else str(e), "form_data": form_dict, "page_name": "request_demo"}
        return templates_env.TemplateResponse("pages/request_demo.html", context, status_code=400)

    success = False
    if app_state.crm_wrapper:
        contact_payload = {
            "email": form_data.workEmail, "first_name": form_data.fullName.split(" ")[0] if form_data.fullName else None,
            "last_name": " ".join(form_data.fullName.split(" ")[1:]) if form_data.fullName and " " in form_data.fullName else None,
            "company_name": form_data.companyName, "domain": form_data.companyWebsite, "job_title": form_data.role,
            "status": "Demo_Requested_Enterprise_Hot",
            "notes": f"Challenge: {form_data.primaryChallenge}\nInterested in: {', '.join(form_data.interestCapabilities)}\nEmployees: {form_data.numEmployees}\nIndustry: {form_data.industry}",
            "source_info": "Website Demo Request Form"
        }
        try:
            result = await app_state.crm_wrapper.upsert_contact(contact_payload, unique_key_column="email")
            success = bool(result)
            logger.info(f"Demo request processed. CRM Result: {success}")
        except Exception as e_crm:
            logger.error(f"CRM error processing demo request: {e_crm}", exc_info=True)

    if success:
        context = {"request": request, "success_message": "Demo request received! Our team will contact you shortly to schedule.", "page_name": "request_demo"}
        return templates_env.TemplateResponse("pages/request_demo.html", context)
    else:
        context = {"request": request, "error_message": "There was an issue submitting your demo request. Please try again.", "form_data": form_dict, "page_name": "request_demo"}
        return templates_env.TemplateResponse("pages/request_demo.html", context, status_code=500)


@router.get("/schedule-consultation", response_class=HTMLResponse, name="schedule_consultation_get")
async def schedule_consultation_get(request: Request, deps = Depends(get_dependencies)):
    app_state, templates_env = deps
    context = {"request": request, "meta_title": "Schedule AI Strategy Consultation | Boutique AI", "form_data": {}, "errors": None, "page_name": "schedule_consultation"}
    return templates_env.TemplateResponse("pages/schedule_consultation.html", context)

@router.post("/schedule-consultation", response_class=HTMLResponse, name="schedule_consultation_post")
async def schedule_consultation_post(
    request: Request,
    deps = Depends(get_dependencies),
    fullName: str = Form(...),
    workEmail: str = Form(...),
    companyName: str = Form(...),
    companyWebsite: Optional[str] = Form(None),
    role: str = Form(...),
    numEmployees: Optional[str] = Form(None),
    industry: Optional[str] = Form(None),
    primaryChallenge: str = Form(...),
    interestCapabilities: List[str] = Form([]),
    strategicGoals: str = Form(...),
    aiChallengeDescription: str = Form(...)
):
    app_state, templates_env = deps
    form_dict = locals()
    del form_dict['request']
    del form_dict['deps']
    del form_dict['app_state']
    del form_dict['templates_env']

    try:
        form_data = ConsultationRequestModel(**form_dict)
    except Exception as e:
        logger.warning(f"Consultation request form validation error: {e.errors() if hasattr(e, 'errors') else str(e)}")
        context = {"request": request, "errors": e.errors() if hasattr(e, 'errors') else str(e), "form_data": form_dict, "page_name": "schedule_consultation"}
        return templates_env.TemplateResponse("pages/schedule_consultation.html", context, status_code=400)

    success = False
    if app_state.crm_wrapper:
        contact_payload = {
            "email": form_data.workEmail, "first_name": form_data.fullName.split(" ")[0] if form_data.fullName else None,
            "last_name": " ".join(form_data.fullName.split(" ")[1:]) if form_data.fullName and " " in form_data.fullName else None,
            "company_name": form_data.companyName, "domain": form_data.companyWebsite, "job_title": form_data.role,
            "status": "Consultation_Requested_Strategic",
            "notes": f"Strategic Goals: {form_data.strategicGoals}\nAI Challenge: {form_data.aiChallengeDescription}\nChallenge: {form_data.primaryChallenge}\nInterested in: {', '.join(form_data.interestCapabilities)}\nEmployees: {form_data.numEmployees}\nIndustry: {form_data.industry}",
            "source_info": "Website Consultation Request Form"
        }
        try:
            result = await app_state.crm_wrapper.upsert_contact(contact_payload, unique_key_column="email")
            success = bool(result)
            logger.info(f"Consultation request processed. CRM Result: {success}")
        except Exception as e_crm:
            logger.error(f"CRM error processing consultation request: {e_crm}", exc_info=True)

    if success:
        context = {"request": request, "success_message": "Consultation request received! We'll reach out soon to confirm a time.", "page_name": "schedule_consultation"}
        return templates_env.TemplateResponse("pages/schedule_consultation.html", context)
    else:
        context = {"request": request, "error_message": "There was an issue submitting your consultation request. Please try again.", "form_data": form_dict, "page_name": "schedule_consultation"}
        return templates_env.TemplateResponse("pages/schedule_consultation.html", context, status_code=500)
# website/forms.py
from typing import Optional, List
from pydantic import BaseModel, EmailStr, HttpUrl

class ContactFormModel(BaseModel):
    fullName: str
    workEmail: EmailStr
    companyName: str
    phone: Optional[str] = None
    helpReason: str
    message: str

class DemoRequestModel(BaseModel):
    fullName: str
    workEmail: EmailStr
    companyName: str
    companyWebsite: Optional[HttpUrl] = None # Pydantic will validate if it's a URL
    role: str
    numEmployees: Optional[str] = None
    industry: Optional[str] = None
    primaryChallenge: str
    # For checkboxes, FastAPI can receive multiple values for the same name
    interestCapabilities: Optional[List[str]] = []

class ConsultationRequestModel(DemoRequestModel): # Inherits fields from DemoRequestModel
    strategicGoals: str
    aiChallengeDescription: str
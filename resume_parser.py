from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from crewai import Agent, Task, Crew
from langchain_ollama import ChatOllama
import json
import PyPDF2
import docx
import io
from datetime import datetime

# Get current date for real-time calculations
CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month

# -----------------------------
# 1️⃣ Configure Ollama LLM
# -----------------------------


llm = ChatOllama(
    model="ollama/deepseek-v3.1:671b-cloud", 
    # model="ollama/qwen3:8b",  
    base_url="http://localhost:11434" 
)

# -----------------------------
# 2️⃣ Define strict data schema
# -----------------------------
class Education(BaseModel):
    degree: Optional[str] = ""
    university: Optional[str] = ""
    graduation_year: Optional[str] = ""

class Experience(BaseModel):
    company: Optional[str] = ""
    title: Optional[str] = ""
    start_date: Optional[str] = ""
    end_date: Optional[str] = ""

class ResumeSchema(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    education: List[Education] = Field(default_factory=list)
    experiences: List[Experience] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    total_years_of_experience: Optional[float] = 0.0
    summary: Optional[str] = ""

class ValidationResult(BaseModel):
    is_valid: bool
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)

class ResumeResponse(BaseModel):
    extracted_data: ResumeSchema
    validation: ValidationResult
    file_info: dict

# -----------------------------
# 3️⃣ File parsing function
# -----------------------------
async def extract_text_from_file(file: UploadFile) -> str:
    """Extract text from PDF, DOCX, or TXT files"""
    content = await file.read()
    filename = file.filename.lower()
    
    try:
        if filename.endswith('.pdf'):
            # Extract from PDF
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        
        elif filename.endswith('.docx'):
            # Extract from DOCX
            docx_file = io.BytesIO(content)
            doc = docx.Document(docx_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        
        elif filename.endswith('.txt'):
            # Extract from TXT
            return content.decode('utf-8').strip()
        
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Please upload PDF, DOCX, or TXT file. Got: {filename}"
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Error parsing {filename}: {str(e)}"
        )

# -----------------------------
# 3.5️⃣ Backup experience calculator
# -----------------------------
def calculate_total_experience(experiences: List[Experience]) -> float:
    """
    Backup function to calculate total years of experience
    Handles 'Present', 'Current', etc. with real-time date
    """
    from dateutil import parser
    import re
    
    total_months = 0
    
    for exp in experiences:
        try:
            start_date = exp.start_date
            end_date = exp.end_date
            
            if not start_date:
                continue
            
            # Parse start date
            start_year = None
            start_month = 1
            
            # Try to extract year and month
            if re.search(r'\d{4}', start_date):
                start_year = int(re.search(r'\d{4}', start_date).group())
            if re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', start_date, re.IGNORECASE):
                month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                           'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
                month_str = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', 
                                     start_date, re.IGNORECASE).group().lower()[:3]
                start_month = month_map.get(month_str, 1)
            
            # Parse end date
            end_year = CURRENT_YEAR
            end_month = CURRENT_MONTH
            
            if end_date and not re.search(r'(present|current|now)', end_date, re.IGNORECASE):
                if re.search(r'\d{4}', end_date):
                    end_year = int(re.search(r'\d{4}', end_date).group())
                if re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', end_date, re.IGNORECASE):
                    month_str = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', 
                                         end_date, re.IGNORECASE).group().lower()[:3]
                    end_month = month_map.get(month_str, 12)
            
            # Calculate months for this job
            if start_year:
                months = (end_year - start_year) * 12 + (end_month - start_month)
                total_months += max(0, months)
        
        except Exception as e:
            print(f"Error calculating experience for {exp.company}: {e}")
            continue
    
    return round(total_months / 12, 1)

# -----------------------------
# 4️⃣ Create agents with Ollama
# -----------------------------
extractor = Agent(
    role="Resume Extractor",
    goal="Extract structured information from resumes with high accuracy.",
    backstory="You are an expert AI that parses resumes and extracts key information accurately. You always return valid JSON.",
    llm=llm,
    verbose=True,
)

validator = Agent(
    role="Data Validator",
    goal="Review extracted resume data for completeness and accuracy.",
    backstory="You are a quality assurance expert who validates extracted data, identifies missing information, and suggests improvements.",
    llm=llm,
    verbose=True,
)

# -----------------------------
# 5️⃣ Define the FastAPI app
# -----------------------------
app = FastAPI(
    title="Resume AI Extractor & Validator", 
    version="3.0",
    description="Upload resume (PDF/DOCX/TXT) and get extracted & validated data in one API call"
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "model": "deepseek-v3.1:671b-cloud",
        "supported_formats": ["PDF", "DOCX", "TXT"],
        "endpoint": "/process_resume"
    }

@app.post("/process_resume", response_model=ResumeResponse)
async def process_resume(file: UploadFile = File(...)):
    """
    Upload resume and get extracted + validated data in one call.
    
    - **file**: Upload a resume file (.pdf, .docx, or .txt)
    
    Returns:
    - Extracted resume data (name, email, phone, education, experience, skills)
    - Validation report (issues and suggestions)
    - File information
    """
    
    # Step 1: Extract text from uploaded file
    content = await extract_text_from_file(file)
    
    if not content:
        raise HTTPException(status_code=400, detail="File appears to be empty or unreadable")

    # Step 2: Create extraction task
    extract_task = Task(
        description=(
            f"Extract resume data from the following text and return ONLY valid JSON:\n"
            "{\n"
            '  "name": "full name",\n'
            '  "email": "email address",\n'
            '  "phone": "phone number",\n'
            '  "education": [{"degree": "", "university": "", "graduation_year": ""}],\n'
            '  "experiences": [{"company": "", "title": "", "start_date": "", "end_date": ""}],\n'
            '  "skills": ["skill1", "skill2"],\n'
            '  "total_years_of_experience": 0.0,\n'
            '  "summary": "brief professional summary"\n'
            "}\n\n"
            "CRITICAL INSTRUCTIONS FOR DATE HANDLING:\n"
            f"- Current date is: {datetime.now().strftime('%B %Y')} (Month {CURRENT_MONTH}, Year {CURRENT_YEAR})\n"
            f"- If end_date is 'Present', 'Current', 'Now', or similar, use '{CURRENT_MONTH}/{CURRENT_YEAR}' for calculations\n"
            "- Calculate 'total_years_of_experience' by analyzing ALL work experience dates\n"
            "- For each job, calculate: (end_year - start_year) + (end_month - start_month)/12\n"
            "- Handle overlapping periods: count each month only once\n"
            "- Return total as a decimal (e.g., 3.5 for 3 years 6 months)\n"
            "- Be precise: include partial years based on months\n\n"
            "EXAMPLES:\n"
            f"- Job: Jan 2020 to Present → Calculate as: Jan 2020 to {datetime.now().strftime('%b %Y')} = ~{CURRENT_YEAR - 2020 + (CURRENT_MONTH - 1)/12:.1f} years\n"
            "- Job: Jun 2022 to Dec 2023 → 1.5 years\n"
            "- Job: 2020 to 2023 (no months) → 3.0 years\n\n"
            "Create a brief 'summary' (2-3 sentences) highlighting:\n"
            "- Total years of experience\n"
            "- Key skills and expertise areas\n"
            "- Notable companies or achievements\n\n"
            f"Resume text:\n{content}"
        ),
        expected_output="Valid JSON object with resume data including accurate real-time total experience",
        agent=extractor,
    )

    # Step 3: Create validation task
    validate_task = Task(
        description=(
            "Review the extracted resume data and provide validation feedback in JSON format:\n"
            "{\n"
            '  "is_valid": true/false,\n'
            '  "issues": ["list of problems found"],\n'
            '  "suggestions": ["list of improvement suggestions"]\n'
            "}\n\n"
            f"Current date for reference: {datetime.now().strftime('%B %Y')}\n\n"
            "Check for:\n"
            "- Missing required fields (name, email, phone)\n"
            "- Incomplete education or experience entries\n"
            "- Verify 'Present' or 'Current' end dates were properly calculated to current date\n"
            f"- Ensure jobs ending in 'Present' are calculated up to {CURRENT_MONTH}/{CURRENT_YEAR}\n"
            "- Accuracy of total_years_of_experience calculation\n"
            "- Quality of professional summary\n"
            "- Data quality issues\n"
            "- Formatting problems\n\n"
            "IMPORTANT: Do NOT flag 'Present' as missing end_date. It's valid and should be calculated to current date."
        ),
        expected_output="JSON validation report",
        agent=validator,
        context=[extract_task]
    )

    # Step 4: Execute both tasks with CrewAI
    crew = Crew(
        agents=[extractor, validator], 
        tasks=[extract_task, validate_task],
        verbose=True
    )
    crew.kickoff()

    # Step 5: Parse outputs
    try:
        # Parse extraction result
        extract_output = str(extract_task.output.raw).strip()
        if "```json" in extract_output:
            extract_output = extract_output.split("```json")[1].split("```")[0].strip()
        elif "```" in extract_output:
            extract_output = extract_output.split("```")[1].split("```")[0].strip()
        
        resume_data = json.loads(extract_output)
        structured = ResumeSchema(**resume_data)

        # Parse validation result
        validate_output = str(validate_task.output.raw).strip()
        if "```json" in validate_output:
            validate_output = validate_output.split("```json")[1].split("```")[0].strip()
        elif "```" in validate_output:
            validate_output = validate_output.split("```")[1].split("```")[0].strip()
        
        validation = json.loads(validate_output)
        validation_result = ValidationResult(**validation)

    except Exception as e:
        # Fallback on parsing errors
        structured = ResumeSchema(
            total_years_of_experience=0.0,
            summary=""
        )
        validation_result = ValidationResult(
            is_valid=False, 
            issues=[f"Parsing error: {str(e)}"],
            suggestions=["Please check the resume format and try again"]
        )

    # Step 6: Return complete response
    return ResumeResponse(
        extracted_data=structured,
        validation=validation_result,
        file_info={
            "filename": file.filename,
            "content_type": file.content_type,
            "text_length": len(content),
            "format": file.filename.split('.')[-1].upper()
        }
    )

# -----------------------------
# 6️⃣ Run the server
# -----------------------------
# Install: pip install fastapi uvicorn crewai langchain-community PyPDF2 python-docx
# Run: uvicorn app:app --reload --host 0.0.0.0 --port 8000
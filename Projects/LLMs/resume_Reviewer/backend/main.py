# Enhanced Backend with FastAPI
# Install: pip install fastapi uvicorn python-multipart pdfplumber huggingface-hub groq python-dotenv

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import numpy as np
import pdfplumber
from numpy import dot
from numpy.linalg import norm
from huggingface_hub import InferenceClient
from groq import Groq
from typing import Optional
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Resume Analyzer API")

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize API clients
HF_API_KEY = os.getenv("HF_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if not HF_API_KEY or not GROQ_API_KEY:
    raise ValueError("Please set HF_API_KEY and GROQ_API_KEY in .env file")

hf_client = InferenceClient(token=HF_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)


def load_resume_from_bytes(pdf_bytes):
    """
    Extracts text from PDF bytes.
    """
    text = ""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_file_path = tmp_file.name
    
    try:
        with pdfplumber.open(tmp_file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    finally:
        os.unlink(tmp_file_path)
    
    return text.strip()


def compute_embeddings(texts, client):
    """
    Generates embeddings for texts using sentence-transformer.
    """
    embeddings = client.feature_extraction(
        texts,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    return np.array(embeddings, dtype="float32")


def compute_similarity(resume_text, job_desc):
    """
    Computes cosine similarity between resume and job description.
    """
    v1, v2 = compute_embeddings([resume_text, job_desc], hf_client)
    
    n1 = norm(v1)
    n2 = norm(v2)
    
    if n1 == 0 or n2 == 0:
        return 0.0
    
    similarity = dot(v1, v2) / (n1 * n2)
    return float(similarity)


def extract_keywords(text, top_n=15):
    """
    Extract important keywords from text using frequency analysis.
    """
    # Simple keyword extraction - you can enhance with NLP libraries
    words = text.lower().split()
    word_freq = {}
    
    # Filter out common words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                  'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                  'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that'}
    
    for word in words:
        word = word.strip('.,!?;:()"\'').lower()
        if len(word) > 3 and word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:top_n]]


def generate_enhanced_review(resume_text, job_desc, score, resume_keywords, jd_keywords):
    """
    Generate comprehensive resume feedback with enhanced analysis.
    """
    # Find keyword overlap
    common_keywords = set(resume_keywords).intersection(set(jd_keywords))
    missing_keywords = set(jd_keywords) - set(resume_keywords)
    
    prompt = f"""
You are an expert career coach, ATS optimization consultant, and resume strategist.

Job Description:
{job_desc}

Resume:
{resume_text}

Resume-Job Match Score: {score:.2f}

Key Insights:
- Common Keywords Found: {', '.join(common_keywords) if common_keywords else 'None'}
- Missing Important Keywords: {', '.join(list(missing_keywords)[:10]) if missing_keywords else 'None'}

Provide a comprehensive, structured review with these sections:

1. **STRENGTHS** (3-5 bullet points)
   - Highlight specific qualifications, experiences, or skills that align well with the job
   - Note any standout achievements or certifications

2. **GAPS & WEAKNESSES** (3-5 bullet points)
   - Identify missing skills, experiences, or qualifications mentioned in the job description
   - Note any formatting or content issues that might hurt ATS compatibility

3. **ATS OPTIMIZATION TIPS** (3-4 actionable recommendations)
   - Specific keywords to add
   - Formatting improvements
   - Section enhancements

4. **IMPROVEMENT SUGGESTIONS** (4-6 specific, actionable items)
   - Concrete ways to strengthen the resume for this specific job
   - How to reframe existing experience
   - What new content to add

5. **OVERALL HIRING VERDICT**
   - Would you recommend this candidate for an interview? (Strong Yes/Yes/Maybe/No)
   - 2-3 sentence summary of candidacy strength
   - One key action item to improve chances

Be specific, actionable, and constructive. Focus on practical improvements.
"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=2000
    )
    
    return response.choices[0].message.content.strip()


@app.get("/")
async def root():
    """
    Health check endpoint.
    """
    return {
        "message": "AI Resume Analyzer API is running",
        "version": "2.0",
        "status": "healthy"
    }


@app.post("/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: Optional[UploadFile] = File(None),
    job_description_text: Optional[str] = Form(None)
):
    """
    Analyze resume against job description.
    Accepts resume PDF and either job description file or text.
    """
    try:
        # Validate inputs
        if not resume.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Resume must be a PDF file")
        
        # Get job description
        if job_description:
            job_desc = (await job_description.read()).decode('utf-8')
        elif job_description_text:
            job_desc = job_description_text
        else:
            raise HTTPException(
                status_code=400,
                detail="Please provide either job_description file or job_description_text"
            )
        
        # Load and process resume
        resume_bytes = await resume.read()
        resume_text = load_resume_from_bytes(resume_bytes)
        
        if not resume_text:
            raise HTTPException(status_code=400, detail="Could not extract text from resume")
        
        # Extract keywords
        resume_keywords = extract_keywords(resume_text)
        jd_keywords = extract_keywords(job_desc)
        
        # Compute similarity score
        score = compute_similarity(resume_text, job_desc)
        
        # Generate enhanced review
        review = generate_enhanced_review(
            resume_text, 
            job_desc, 
            score, 
            resume_keywords, 
            jd_keywords
        )
        
        return JSONResponse({
            "success": True,
            "match_score": round(score, 3),
            "review": review,
            "resume_keywords": resume_keywords[:10],
            "jd_keywords": jd_keywords[:10],
            "common_keywords": list(set(resume_keywords).intersection(set(jd_keywords)))[:10]
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/keywords")
async def extract_keywords_endpoint(
    resume: UploadFile = File(...),
    job_description: Optional[UploadFile] = File(None),
    job_description_text: Optional[str] = Form(None)
):
    """
    Extract and compare keywords from resume and job description.
    """
    try:
        # Get job description
        if job_description:
            job_desc = (await job_description.read()).decode('utf-8')
        elif job_description_text:
            job_desc = job_description_text
        else:
            raise HTTPException(
                status_code=400,
                detail="Please provide either job_description file or job_description_text"
            )
        
        # Load resume
        resume_bytes = await resume.read()
        resume_text = load_resume_from_bytes(resume_bytes)
        
        # Extract keywords
        resume_keywords = extract_keywords(resume_text, top_n=20)
        jd_keywords = extract_keywords(job_desc, top_n=20)
        
        common = set(resume_keywords).intersection(set(jd_keywords))
        missing = set(jd_keywords) - set(resume_keywords)
        
        return {
            "resume_keywords": resume_keywords,
            "job_description_keywords": jd_keywords,
            "common_keywords": list(common),
            "missing_keywords": list(missing),
            "match_percentage": round(len(common) / len(jd_keywords) * 100, 1) if jd_keywords else 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
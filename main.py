import re
from utils import clean_json_fences, parse_json, ollama_available
from retriever import build_bm25_index
from ranking import (
    compute_location_score,
    compute_skill_score,
    compute_experience_score,
    compute_final_score
)
from database import fetch_all_jobs
from langchain_community.llms import Ollama


# ---------------------------- SKILL EXTRACTION ----------------------------

def load_llm_model():
    return Ollama(
        model="llama3.2:3b",
        temperature=0.3,
        num_predict=500
    )


def extract_skills_llm(resume):
    """Extract technical skills from resume using LLM"""
    
    if not ollama_available():
        return extract_skills_fallback(resume)

    llm = load_llm_model()
    prompt = f"""
You are a technical skill extraction expert. Analyze this resume and extract ONLY the technical skills mentioned.

Rules:
1. Extract programming languages, frameworks, libraries, tools, databases, cloud platforms
2. Return skills exactly as they appear in the resume (preserve casing like PyTorch, TensorFlow)
3. Do NOT add skills that are not explicitly mentioned
4. Return ONLY a JSON array of strings
5. Example format: ["Python", "TensorFlow", "AWS", "Docker", "MySQL"]

Resume:
{resume[:3000]}

Return only the JSON array, nothing else.
"""

    resp = llm.invoke(prompt).strip()
    skills = parse_json(resp)

    if not skills or not isinstance(skills, list):
        return extract_skills_fallback(resume)

    # Clean and normalize skills (lowercase for matching)
    cleaned_skills = [s.lower().strip() for s in skills if isinstance(s, str) and s.strip()]
    return list(set(cleaned_skills))  # Remove duplicates


def extract_skills_fallback(resume):
    """Fallback: Extract skills by finding them in database job skills"""
    resume_lower = resume.lower()
    
    # Get all unique skills from database
    jobs = fetch_all_jobs()
    if not jobs:
        return []
    
    # Collect all skills from database
    db_skills = set()
    for job in jobs:
        if job.get("skills"):
            skills_list = [s.strip().lower() for s in job["skills"].split(",") if s.strip()]
            db_skills.update(skills_list)
    
    # Find which database skills appear in the resume
    found_skills = []
    for skill in db_skills:
        # Use word boundary for accurate matching
        if re.search(rf"\b{re.escape(skill)}\b", resume_lower):
            found_skills.append(skill)
    
    return found_skills if found_skills else []


# ---------------------------- MAIN MATCHER ----------------------------

def match_jobs(resume_text, extracted_skills, user_location="Hyderabad"):
    """Match jobs based on extracted skills from resume"""
    
    jobs = fetch_all_jobs()
    if not jobs:
        return []

    if not extracted_skills:
        return []

    # Filter jobs that match at least one skill from resume
    filtered = []
    for job in jobs:
        job_skills_str = job.get("skills", "").lower()
        job_skills_list = [s.strip() for s in job_skills_str.split(",") if s.strip()]
        
        # Check if any extracted skill matches job skills
        skill_match = any(
            skill.lower() in job_skills_str or 
            any(skill.lower() in js or js in skill.lower() for js in job_skills_list)
            for skill in extracted_skills
        )
        
        if skill_match:
            filtered.append(job)

    if not filtered:
        return []

    # Use BM25 for semantic ranking
    bm25, _ = build_bm25_index(filtered)
    docs = bm25.get_relevant_documents(resume_text)
    
    if not docs:
        # Fallback to all filtered jobs
        docs_data = [(i, job) for i, job in enumerate(filtered)]
    else:
        k = min(15, len(docs))
        top_docs = docs[:k]
        
        docs_data = []
        for rank, doc in enumerate(top_docs):
            idx = doc.metadata.get("idx", 0) if hasattr(doc, 'metadata') else 0
            docs_data.append((idx, filtered[idx]))

    results = []
    
    for idx, (job_idx, job) in enumerate(docs_data):
        # Compute semantic score based on rank
        semantic = (1 - (idx / len(docs_data))) * 100 if len(docs_data) > 1 else 100
        
        # Compute other scores
        loc = compute_location_score(user_location, job["location"])
        skill = compute_skill_score(resume_text, job["skills"])
        exp = compute_experience_score(resume_text, job["description"])

        final = compute_final_score(loc, skill, exp, semantic)

        results.append({
            "id": job["id"],
            "title": job["title"],
            "company": job["company"],
            "location": job["location"],
            "description": job["description"],
            "skills": job["skills"],
            "final_score": round(final, 2)
        })

    # Return top 10 matches
    return sorted(results, key=lambda x: x["final_score"], reverse=True)[:10]
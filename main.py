import re
from utils import extract_text, clean_json_fences, parse_json, ollama_available
from retriever import build_bm25_index
from database import fetch_all_jobs
from langchain_community.llms import Ollama
from ranking import (
    compute_location_score,
    compute_skill_score,
    compute_experience_score,
    compute_final_score
)

# ---------------------------- ROLE DETECTION ----------------------------

ROLE_SKILLS = {
    "QA Engineer": [
        "manual testing", "functional testing", "regression testing",
        "test cases", "test plans", "qa", "bug reporting",
        "stlc", "sdlc", "jira", "test case design", "defect reporting",
        "ui testing", "api testing", "postman", "test execution"
    ],

    "QA Automation Engineer": [
        "automation testing", "selenium", "cypress", "pytest",
        "automation framework", "webdriver", "test automation",
        "locust", "load testing", "ui automation", "automation scripts"
    ],

    "AI Engineer": [
        "machine learning", "deep learning", "tensorflow", "pytorch",
        "llm", "rag", "embedding", "transformers", "vector db"
    ],

    "Data Scientist": [
        "statistics", "pandas", "numpy", "ml", "eda",
        "predictive modeling", "data visualization"
    ],

    "Software Engineer": [
        "python", "java", "c++", "backend", "frontend", "rest api",
        "debugging", "database", "sql"
    ],

    "Backend Developer": [
        "fastapi", "flask", "django", "microservices", "graphql"
    ]
}


def detect_role_weighted(resume):
    r = resume.lower()
    scores = {role: sum(1 for s in skills if s in r)
              for role, skills in ROLE_SKILLS.items()}
    best = sorted(scores.items(), key=lambda x: x[1], reverse=True)

# if everything is too low, default to QA roles for safety
    if best[0][1] == 0:
        return ["Software Engineer"]

    return [best[0][0], best[1][0]]


def load_llm_model():
    return Ollama(
        model="llama3.2:3b",
        temperature=0.7,
        num_predict=300
    )


def recommend_jobs_llm(resume):

    if not ollama_available():
        return detect_role_weighted(resume)

    llm = load_llm_model()
    prompt = f"""
Classify resume text into job roles.
Return ONLY a JSON array of job titles.

Resume:
{resume}
"""

    resp = llm.invoke(prompt).strip()
    roles = parse_json(resp)

    if not roles:
        return detect_role_weighted(resume)

    return roles[:3]


# ---------------------------- MAIN MATCHER ----------------------------

def match_jobs(resume_text, recommended_roles, user_location="Hyderabad"):

    jobs = fetch_all_jobs()
    if not jobs:
        return []

    filtered = [
        j for j in jobs
        if any(r.lower() in j["title"].lower() for r in recommended_roles)
    ]

    if not filtered:
        return []

    bm25, _ = build_bm25_index(filtered)

    docs = bm25.get_relevant_documents(resume_text)
    if not docs:
        return []

    k = min(10, len(docs))
    top_docs = docs[:k]

    idxs = []
    semantic_scores = []

    for rank, doc in enumerate(top_docs):
        # Prefer metadata-based mapping back to original job index
        idx = doc.metadata.get("idx", 0) if hasattr(doc, 'metadata') else 0

        idxs.append(idx)
        dist = (rank / (k - 1)) if k > 1 else 0
        semantic_scores.append((1 / (1 + dist)) * 100)

    results = []

    for sem, idx in zip(semantic_scores, idxs):
        job = filtered[idx]

        loc = compute_location_score(user_location, job["location"])
        skill = compute_skill_score(resume_text, job["skills"])
        exp = compute_experience_score(resume_text, job["description"])

        final = compute_final_score(loc, skill, exp, sem)

        results.append({
            "id": job["id"],
            "title": job["title"],
            "company": job["company"],
            "location": job["location"],
            "description": job["description"],
            "skills": job["skills"],
            "final_score": round(final, 2)
        })

    return sorted(results, key=lambda x: x["final_score"], reverse=True)[:5]

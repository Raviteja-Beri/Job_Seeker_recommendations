import re

# ---------------------------- YEARS EXTRACTION ----------------------------

def extract_years(text):
    text = text.lower()

    m = re.search(r"(\d+)\s*(years?|yrs?)", text)
    if m:
        return int(m.group(1))

    m = re.search(r"(\d+)\s*\+\s*(years?|yrs?)", text)
    if m:
        return int(m.group(1))

    m = re.search(r"(\d+)\s*-\s*(\d+)\s*(years?|yrs?)", text)
    if m:
        return int(m.group(1))

    return 0


# ---------------------------- LOCATION SCORE ----------------------------

def compute_location_score(user_location, job_location):
    if not user_location or not job_location:
        return 0

    u = user_location.lower()
    j = job_location.lower()

    if u == j:
        return 100
    if u in j or j in u:
        return 70
    return 40


# ---------------------------- SKILL SCORE ----------------------------

def compute_skill_score(resume_text, job_skills):
    if not job_skills:
        return 0

    resume = resume_text.lower()
    skills = [s.strip().lower() for s in job_skills.split(",") if s.strip()]

    matches = sum(1 for s in skills if re.search(rf"\b{s}\b", resume))

    return (matches / len(skills)) * 100 if skills else 0


# ---------------------------- EXPERIENCE SCORE ----------------------------

def compute_experience_score(resume_text, job_desc):
    resume_years = extract_years(resume_text)
    job_years = extract_years(job_desc)

    if job_years == 0:
        return 50

    diff = abs(resume_years - job_years)

    if diff == 0:
        return 100
    elif diff <= 2:
        return 70
    elif diff <= 4:
        return 50
    return 20


# ---------------------------- FINAL SCORE ----------------------------

def compute_final_score(location, skills, experience, semantic):
    return (
        0.50 * location +
        0.30 * skills +
        0.15 * experience +
        0.05 * semantic
    )

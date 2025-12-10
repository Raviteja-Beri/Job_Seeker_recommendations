import streamlit as st
from utils import extract_text, ollama_available
from main import extract_skills_llm, match_jobs
from database import fetch_all_jobs


st.title("AI-Powered Resume Job Matcher")
st.markdown("Upload your resume and let AI find the best job matches based on your skills")

uploaded = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

col1, col2 = st.columns(2)
with col1:
    location = st.text_input("Preferred Job Location", "Hyderabad")
with col2:
    llm_status = "Active" if ollama_available() else "Offline (using fallback)"
    st.text_input("LLM Status", llm_status, disabled=True)

if uploaded:
    with st.spinner("Extracting resume text..."):
        resume_text = extract_text(uploaded)

    st.subheader("Extracted Resume Preview")
    with st.expander("View full resume text"):
        st.text(resume_text[:2000] + "..." if len(resume_text) > 2000 else resume_text)

    with st.spinner("Analyzing skills with AI..."):
        extracted_skills = extract_skills_llm(resume_text)

    st.subheader("Extracted Skills")
    if extracted_skills:
        # Display skills as badges
        skills_html = " ".join([
            f'<span style="background-color: #0066cc; color: white; padding: 5px 10px; '
            f'border-radius: 15px; margin: 3px; display: inline-block; font-size: 14px;">'
            f'{skill}</span>' 
            for skill in extracted_skills
        ])
        st.markdown(skills_html, unsafe_allow_html=True)
        
        # Show skill count
        st.success(f"Found **{len(extracted_skills)}** technical skills in your resume")
    else:
        st.error("No technical skills extracted from resume. Please ensure your resume contains clear technical skills.")
        st.stop()

    with st.spinner("Finding matching jobs..."):
        matches = match_jobs(resume_text, extracted_skills, location)

    st.subheader("💼 Top Job Matches")
    
    if matches:
        st.success(f"Found **{len(matches)}** matching jobs based on your skills!")
        
        for i, job in enumerate(matches, 1):
            with st.expander(
                f"#{i} - {job['title']} at {job['company']} "
                f"(Match: {job['final_score']}%)"
            ):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Company:** {job['company']}")
                    st.markdown(f"**Location:** {job['location']}")
                    st.markdown(f"**Description:** {job['description']}")
                
                with col2:
                    st.metric("Match Score", f"{job['final_score']}%")
                    st.markdown(f"**Required Skills:**")
                    job_skills = job['skills'].split(',')
                    
                    # Highlight matching skills
                    for skill in job_skills[:5]:  # Show first 5 skills
                        skill_clean = skill.strip().lower()
                        if any(s.lower() == skill_clean or s.lower() in skill_clean or skill_clean in s.lower() 
                               for s in extracted_skills):
                            st.markdown(f"**{skill.strip()}**")  # Matching skill
                        else:
                            st.markdown(f"• {skill.strip()}")  # Non-matching skill
    else:
        st.warning("No matching jobs found in database.")
        
        # Show helpful information
        st.info(f"""
        **Your extracted skills:** {', '.join(extracted_skills)}
        
        **Possible reasons:**
        - Database doesn't have jobs matching your skills
        - Skills in database are named differently
        - Try adding more skills to your resume
        """)
        
        # Show database stats
        total_jobs = len(fetch_all_jobs())
        st.info(f"Total jobs in database: **{total_jobs}**")

# Sidebar with instructions
with st.sidebar:
    st.header("ℹHow it works")
    st.markdown("""
    1. **Upload** your resume (PDF, DOCX, or TXT)
    2. **AI extracts** your technical skills using LLM
    3. **Smart matching** finds jobs from database
    4. **Get ranked** results based on:
       - Location match (50%)
       - Skill match (30%)
       - Experience match (15%)
       - Semantic similarity (5%)
    """)
    
    st.header("Tips")
    st.markdown("""
    - Clearly list your technical skills
    - Mention years of experience
    - Use standard skill names (Python, not Py)
    - Keep resume well-structured
    - Ensure Ollama is running for better extraction
    """)
    
    st.header("System Info")
    if ollama_available():
        st.success("LLM Active - Using AI extraction")
    else:
        st.warning("LLM Offline - Using database fallback")
        st.caption("Start Ollama with: `ollama serve`")
import streamlit as st
from main import extract_text, recommend_jobs_llm, match_jobs, ollama_available
from database import fetch_all_jobs


st.title("RAG-Powered Resume Job Matcher")

uploaded = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

location = st.text_input("Preferred Job Location", "Hyderabad")
exp_years = st.text_input("Your Experience (Optional)")

if uploaded:
    resume_text = extract_text(uploaded)

    st.subheader("Extracted Resume Preview")
    st.write(resume_text[:2000])

    roles = recommend_jobs_llm(resume_text)

    st.subheader("Recommended Job Roles")
    st.json(roles)

    # Debug: show jobs count and sample titles to help explain empty matches
    jobs = fetch_all_jobs()
    matches = match_jobs(resume_text, roles, location)

    st.subheader("Final Ranked Job Matches")
    st.json(matches)
    

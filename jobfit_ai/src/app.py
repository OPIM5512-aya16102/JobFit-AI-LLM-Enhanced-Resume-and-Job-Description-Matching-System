import os
import sys
import pandas as pd
import streamlit as st
from docx import Document
import pdfplumber

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import clean_text
from similarity import calculate_final_score
from skill_extractor import compare_skills
from llm_feedback import generate_basic_feedback, generate_llm_feedback
from section_analyzer import analyze_resume_sections


def read_docx(uploaded_file):
    document = Document(uploaded_file)
    text = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text.append(paragraph.text.strip())

    return "\n".join(text)


def read_pdf(uploaded_file):
    text = []

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text.append(page_text)

    return "\n".join(text)


def read_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8")


def extract_uploaded_text(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".docx"):
        return read_docx(uploaded_file)

    if file_name.endswith(".pdf"):
        return read_pdf(uploaded_file)

    if file_name.endswith(".txt"):
        return read_txt(uploaded_file)

    raise ValueError("Unsupported file type.")


def calculate_weighted_final_score(tfidf_score, semantic_score, skill_coverage):
    return round(
        (0.30 * tfidf_score)
        + (0.40 * semantic_score)
        + (0.30 * skill_coverage),
        2
    )


def analyze_single_job(resume_text, job_text, job_name):
    clean_resume = clean_text(resume_text)
    clean_job = clean_text(job_text)

    scores = calculate_final_score(clean_resume, clean_job)
    skill_results = compare_skills(clean_resume, clean_job)

    tfidf_score = round(scores["tfidf_score"] * 100, 2)
    semantic_score = round(scores["embedding_score"] * 100, 2)

    total_skills = (
        len(skill_results["matched_skills"])
        + len(skill_results["missing_skills"])
    )

    if total_skills > 0:
        skill_coverage = round(
            len(skill_results["matched_skills"]) / total_skills * 100,
            2
        )
    else:
        skill_coverage = 0

    final_score = calculate_weighted_final_score(
        tfidf_score=tfidf_score,
        semantic_score=semantic_score,
        skill_coverage=skill_coverage
    )

    return {
        "job_name": job_name,
        "tfidf_score": tfidf_score,
        "semantic_score": semantic_score,
        "skill_coverage": skill_coverage,
        "final_match_score": final_score,
        "skill_results": skill_results
    }


def rank_multiple_jobs(resume_text, job_files):
    rows = []

    for job_file in job_files:
        try:
            job_text = extract_uploaded_text(job_file)
        except Exception:
            continue

        if not job_text.strip():
            continue

        result = analyze_single_job(
            resume_text=resume_text,
            job_text=job_text,
            job_name=job_file.name
        )

        matched_skills = ", ".join(result["skill_results"]["matched_skills"])
        missing_skills = ", ".join(result["skill_results"]["missing_skills"])

        rows.append(
            {
                "job_name": result["job_name"],
                "tfidf_score": result["tfidf_score"],
                "semantic_score": result["semantic_score"],
                "skill_coverage": result["skill_coverage"],
                "final_match_score": result["final_match_score"],
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "num_matched_skills": len(result["skill_results"]["matched_skills"]),
                "num_missing_skills": len(result["skill_results"]["missing_skills"])
            }
        )

    ranking_df = pd.DataFrame(rows)

    if ranking_df.empty:
        return ranking_df

    ranking_df = ranking_df.sort_values(
        by="final_match_score",
        ascending=False
    ).reset_index(drop=True)

    ranking_df.insert(
        0,
        "rank",
        ranking_df.index + 1
    )

    return ranking_df


st.set_page_config(
    page_title="JobFit AI",
    page_icon="📄",
    layout="wide"
)

st.title("JobFit AI")
st.subheader("LLM-Enhanced Resume and Job Description Matching System")

st.write(
    "Upload your resume and analyze one job description or rank multiple job descriptions "
    "using TF-IDF, semantic similarity, skill coverage, and optional AI feedback."
)

st.markdown("---")

with st.sidebar:
    st.header("Settings")

    app_mode = st.radio(
        "Choose Mode",
        [
            "Single Job Analysis",
            "Multi-Job Ranking"
        ]
    )

    use_llm_feedback = st.checkbox(
        "Use LLM Feedback",
        value=False,
        help="Requires OPENAI_API_KEY in your .env file. Only used in Single Job Analysis."
    )

    st.caption(
        "Portfolio-safe setup: API keys are loaded from environment variables and should not be committed to GitHub."
    )

resume_file = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx"]
)

if app_mode == "Single Job Analysis":

    job_file = st.file_uploader(
        "Upload Job Description",
        type=["txt"]
    )

    job_text_input = st.text_area(
        "Or Paste Job Description",
        height=250
    )

    job_name = st.text_input(
        "Job Name",
        value="custom_job"
    )

    run_button = st.button("Analyze Match")

    if run_button:

        if resume_file is None:
            st.error("Please upload your resume as a PDF or DOCX file.")
            st.stop()

        try:
            resume_text = extract_uploaded_text(resume_file)
        except Exception as error:
            st.error(f"Could not read resume file: {error}")
            st.stop()

        if job_file is not None:
            try:
                job_text = extract_uploaded_text(job_file)
                job_name = job_file.name
            except Exception as error:
                st.error(f"Could not read job description file: {error}")
                st.stop()
        else:
            job_text = job_text_input

        if not resume_text.strip():
            st.error("Resume text is empty. Try another file.")
            st.stop()

        if not job_text.strip():
            st.error("Please upload or paste a job description.")
            st.stop()

        result = analyze_single_job(
            resume_text=resume_text,
            job_text=job_text,
            job_name=job_name
        )

        section_scores = analyze_resume_sections(
            resume_text,
            job_text
        )

        tfidf_score = result["tfidf_score"]
        semantic_score = result["semantic_score"]
        skill_coverage = result["skill_coverage"]
        final_score = result["final_match_score"]
        skill_results = result["skill_results"]

        matched_skills = ", ".join(skill_results["matched_skills"])
        missing_skills = ", ".join(skill_results["missing_skills"])

        st.markdown("---")
        st.header("Match Results")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("TF-IDF Score", f"{tfidf_score}%")
        col2.metric("Semantic Score", f"{semantic_score}%")
        col3.metric("Skill Coverage", f"{skill_coverage}%")
        col4.metric("Final Match Score", f"{final_score}%")

        st.caption(
            "Final Match Score = 30% TF-IDF + 40% semantic similarity + 30% skill coverage."
        )

        st.markdown("---")

        col5, col6 = st.columns(2)

        with col5:
            st.subheader("Matched Skills")

            if skill_results["matched_skills"]:
                for skill in skill_results["matched_skills"]:
                    st.success(skill)
            else:
                st.warning("No matched skills found.")

        with col6:
            st.subheader("Missing Skills")

            if skill_results["missing_skills"]:
                for skill in skill_results["missing_skills"]:
                    st.error(skill)
            else:
                st.success("No missing skills found from tracked skill list.")

        st.markdown("---")
        st.header("Skill Overlap Analysis")

        skill_rows = []

        for skill in skill_results["matched_skills"]:
            skill_rows.append(
                {
                    "Skill": skill,
                    "Status": "Matched",
                    "Value": 1
                }
            )

        for skill in skill_results["missing_skills"]:
            skill_rows.append(
                {
                    "Skill": skill,
                    "Status": "Missing",
                    "Value": 0
                }
            )

        skill_df = pd.DataFrame(skill_rows)

        if not skill_df.empty:

            st.dataframe(
                skill_df,
                use_container_width=True
            )

            summary_df = (
                skill_df
                .groupby("Status")
                .size()
                .reset_index(name="Count")
            )

            st.subheader("Matched vs Missing Skills")

            st.bar_chart(
                summary_df.set_index("Status")
            )

        else:
            st.info("No tracked skills found in this job description.")

        st.markdown("---")
        st.header("Resume Section Analysis")

        section_rows = []

        for section_name, values in section_scores.items():
            section_rows.append(
                {
                    "Section": section_name.title(),
                    "TF-IDF Score": values["tfidf_score"],
                    "Semantic Score": values["semantic_score"],
                    "Final Score": values["final_score"],
                    "Text Length": values["text_length"]
                }
            )

        section_df = pd.DataFrame(section_rows)

        section_df = section_df.sort_values(
            by="Final Score",
            ascending=False
        )

        st.dataframe(
            section_df,
            use_container_width=True
        )

        st.bar_chart(
            section_df.set_index("Section")["Final Score"]
        )

        if use_llm_feedback:
            feedback = generate_llm_feedback(
                resume_text=resume_text,
                job_text=job_text,
                job_name=job_name,
                final_score=final_score,
                matched_skills=matched_skills,
                missing_skills=missing_skills
            )
        else:
            feedback = generate_basic_feedback(
                job_name=job_name,
                final_score=final_score,
                matched_skills=matched_skills,
                missing_skills=missing_skills
            )

        st.markdown("---")
        st.header("Resume Feedback")

        if use_llm_feedback:
            st.caption("LLM feedback enabled.")
        else:
            st.caption("Basic rule-based feedback enabled. Turn on LLM Feedback in the sidebar to use the API.")

        st.text(feedback)

        results_df = pd.DataFrame(
            [
                {
                    "job_name": job_name,
                    "tfidf_score": tfidf_score,
                    "semantic_score": semantic_score,
                    "skill_coverage": skill_coverage,
                    "final_match_score": final_score,
                    "matched_skills": matched_skills,
                    "missing_skills": missing_skills,
                    "num_matched_skills": len(skill_results["matched_skills"]),
                    "num_missing_skills": len(skill_results["missing_skills"])
                }
            ]
        )

        st.markdown("---")
        st.header("Download Results")

        csv = results_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Match Results CSV",
            data=csv,
            file_name="jobfit_ai_match_results.csv",
            mime="text/csv"
        )

        section_csv = section_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Section Analysis CSV",
            data=section_csv,
            file_name="jobfit_ai_section_analysis.csv",
            mime="text/csv"
        )

        if not skill_df.empty:
            skill_csv = skill_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Skill Overlap CSV",
                data=skill_csv,
                file_name="jobfit_ai_skill_overlap.csv",
                mime="text/csv"
            )


if app_mode == "Multi-Job Ranking":

    st.info(
        "Upload multiple .txt job descriptions to rank them against your resume."
    )

    job_files = st.file_uploader(
        "Upload Multiple Job Descriptions",
        type=["txt"],
        accept_multiple_files=True
    )

    run_ranking_button = st.button("Rank Jobs")

    if run_ranking_button:

        if resume_file is None:
            st.error("Please upload your resume as a PDF or DOCX file.")
            st.stop()

        if not job_files:
            st.error("Please upload at least one .txt job description.")
            st.stop()

        try:
            resume_text = extract_uploaded_text(resume_file)
        except Exception as error:
            st.error(f"Could not read resume file: {error}")
            st.stop()

        if not resume_text.strip():
            st.error("Resume text is empty. Try another file.")
            st.stop()

        ranking_df = rank_multiple_jobs(
            resume_text=resume_text,
            job_files=job_files
        )

        if ranking_df.empty:
            st.error("No valid job descriptions were found.")
            st.stop()

        st.markdown("---")
        st.header("Ranked Job Matches")

        st.dataframe(
            ranking_df,
            use_container_width=True
        )

        st.subheader("Top Job Matches")

        chart_df = ranking_df[[
            "job_name",
            "final_match_score"
        ]].set_index("job_name")

        st.bar_chart(chart_df)

        best_job = ranking_df.iloc[0]

        st.markdown("---")
        st.header("Best Match")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Job", best_job["job_name"])
        col2.metric("Final Match", f"{best_job['final_match_score']}%")
        col3.metric("Skill Coverage", f"{best_job['skill_coverage']}%")
        col4.metric("Matched Skills", int(best_job["num_matched_skills"]))

        st.markdown("---")
        st.header("Download Ranking Results")

        ranking_csv = ranking_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Multi-Job Ranking CSV",
            data=ranking_csv,
            file_name="jobfit_ai_multi_job_ranking.csv",
            mime="text/csv"
        )
from similarity import calculate_final_score
from preprocess import clean_text


SECTION_KEYWORDS = {
    "summary": ["summary", "profile", "professional summary"],
    "skills": ["skills", "technical skills", "tools", "technologies"],
    "experience": ["experience", "work experience", "professional experience", "employment"],
    "projects": ["projects", "project experience", "portfolio"],
    "education": ["education", "academic background"]
}


def split_resume_sections(resume_text):
    lines = resume_text.splitlines()

    sections = {
        "summary": [],
        "skills": [],
        "experience": [],
        "projects": [],
        "education": [],
        "other": []
    }

    current_section = "other"

    for line in lines:
        clean_line = line.strip()
        lower_line = clean_line.lower()

        if not clean_line:
            continue

        matched_section = None

        for section, keywords in SECTION_KEYWORDS.items():
            for keyword in keywords:
                if lower_line == keyword or lower_line.startswith(keyword):
                    matched_section = section
                    break
            if matched_section:
                break

        if matched_section:
            current_section = matched_section
            continue

        sections[current_section].append(clean_line)

    return {
        section: "\n".join(text)
        for section, text in sections.items()
    }


def analyze_resume_sections(resume_text, job_text):
    sections = split_resume_sections(resume_text)

    section_scores = {}

    clean_job = clean_text(job_text)

    for section_name, section_text in sections.items():
        if section_text.strip():
            clean_section = clean_text(section_text)
            scores = calculate_final_score(clean_section, clean_job)

            section_scores[section_name] = {
                "tfidf_score": round(scores["tfidf_score"] * 100, 2),
                "semantic_score": round(scores["embedding_score"] * 100, 2),
                "final_score": round(scores["final_score"] * 100, 2),
                "text_length": len(section_text)
            }
        else:
            section_scores[section_name] = {
                "tfidf_score": 0,
                "semantic_score": 0,
                "final_score": 0,
                "text_length": 0
            }

    return section_scores


if __name__ == "__main__":
    sample_resume = """
    Summary
    Data scientist with experience in machine learning and analytics.

    Skills
    Python SQL pandas scikit-learn AWS

    Experience
    Built machine learning models for manufacturing and insurance.

    Projects
    Developed NLP resume matching app using sentence transformers.

    Education
    M.S. Data Science
    """

    sample_job = """
    We are looking for a data scientist with Python, SQL, machine learning,
    NLP, AWS, and model deployment experience.
    """

    results = analyze_resume_sections(sample_resume, sample_job)

    print(results)
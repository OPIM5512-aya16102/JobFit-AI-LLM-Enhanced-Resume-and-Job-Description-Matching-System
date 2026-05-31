SKILLS = [
    "python", "sql", "r", "sas", "excel", "tableau", "power bi",
    "pandas", "numpy", "scikit-learn", "sklearn", "tensorflow", "pytorch",
    "machine learning", "deep learning", "nlp", "natural language processing",
    "llm", "large language model", "generative ai", "rag",
    "aws", "gcp", "azure", "vertex ai", "docker", "git", "github",
    "flask", "streamlit", "api", "etl", "data engineering",
    "statistics", "regression", "classification", "xgboost",
    "random forest", "logistic regression", "model deployment"
]


def extract_skills(text):
    text = text.lower()
    found_skills = []

    for skill in SKILLS:
        if skill in text:
            found_skills.append(skill)

    return sorted(set(found_skills))


def compare_skills(resume_text, job_text):
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_text))

    matched_skills = sorted(resume_skills.intersection(job_skills))
    missing_skills = sorted(job_skills.difference(resume_skills))

    return {
        "resume_skills": sorted(resume_skills),
        "job_skills": sorted(job_skills),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
    }


if __name__ == "__main__":
    resume = "Python SQL machine learning AWS pandas scikit-learn"
    job = "Python SQL NLP GCP Docker machine learning"

    results = compare_skills(resume, job)

    print(results)
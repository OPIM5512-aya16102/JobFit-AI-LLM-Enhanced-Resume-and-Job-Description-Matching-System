import os
from docx import Document


RESUME_FOLDER = "data/resumes"
JOB_DESC_FOLDER = "data/job_descriptions"


def read_docx(file_path):
    """Read text from a DOCX resume."""
    document = Document(file_path)

    text = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text.append(paragraph.text.strip())

    return "\n".join(text)


def read_txt(file_path):
    """Read text from a TXT job description."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def load_resume():
    """Load the first DOCX resume from the resume folder."""
    for file_name in os.listdir(RESUME_FOLDER):
        if file_name.endswith(".docx"):
            file_path = os.path.join(RESUME_FOLDER, file_name)
            return file_name, read_docx(file_path)

    raise FileNotFoundError("No .docx resume found in data/resumes/")


def load_job_descriptions():
    """Load all TXT job descriptions."""
    job_descriptions = {}

    for file_name in os.listdir(JOB_DESC_FOLDER):
        if file_name.endswith(".txt"):
            file_path = os.path.join(JOB_DESC_FOLDER, file_name)
            job_descriptions[file_name] = read_txt(file_path)

    if not job_descriptions:
        raise FileNotFoundError("No .txt job descriptions found in data/job_descriptions/")

    return job_descriptions


if __name__ == "__main__":
    resume_name, resume_text = load_resume()
    job_descriptions = load_job_descriptions()

    print("\n==============================")
    print("RESUME LOADED")
    print("==============================")
    print(f"File: {resume_name}")
    print(resume_text[:1000])

    print("\n==============================")
    print("JOB DESCRIPTIONS LOADED")
    print("==============================")

    for job_name, job_text in job_descriptions.items():
        print(f"\n--- {job_name} ---")
        print(job_text[:1000])
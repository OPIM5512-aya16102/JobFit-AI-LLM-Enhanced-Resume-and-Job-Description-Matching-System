import os
from dotenv import load_dotenv

load_dotenv()


def generate_basic_feedback(
    job_name,
    final_score,
    matched_skills,
    missing_skills
):
    feedback = []

    feedback.append(f"Job: {job_name}")
    feedback.append(f"Final Match Score: {final_score}%")

    if final_score >= 80:
        feedback.append("\nOverall Assessment: Strong Match")
    elif final_score >= 60:
        feedback.append("\nOverall Assessment: Moderate Match")
    else:
        feedback.append("\nOverall Assessment: Weak Match")

    feedback.append("\nMatched Skills:")
    feedback.append(matched_skills if matched_skills else "No matched skills identified.")

    feedback.append("\nMissing Skills:")
    feedback.append(missing_skills if missing_skills else "No missing skills identified.")

    feedback.append("\nRecommendation:")

    if final_score >= 80:
        feedback.append("Resume appears well aligned with the job description.")
    elif final_score >= 60:
        feedback.append("Consider strengthening experience and project descriptions around the missing skills.")
    else:
        feedback.append("Resume should be tailored significantly before applying.")

    feedback.append("\nInterview Preparation:")
    feedback.append("Be prepared to discuss projects that demonstrate the matched technical skills.")

    return "\n".join(feedback)


def generate_llm_feedback(
    resume_text,
    job_text,
    job_name,
    final_score,
    matched_skills,
    missing_skills
):
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return generate_basic_feedback(
            job_name=job_name,
            final_score=final_score,
            matched_skills=matched_skills,
            missing_skills=missing_skills
        )

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        prompt = f"""
You are an expert technical recruiter and resume coach.

Analyze this resume against the job description.

Job Name:
{job_name}

Final Match Score:
{final_score}%

Matched Skills:
{matched_skills}

Missing Skills:
{missing_skills}

Resume:
{resume_text}

Job Description:
{job_text}

Return these sections:

1. Overall Match Assessment
2. Why This Candidate Matches
3. Missing or Weak Areas
4. Resume Bullet Recommendations
5. Keywords to Add Truthfully
6. Interview Questions to Practice
7. Hiring Manager Concerns
8. Final Recommendation

Do not invent experience.
Keep it practical and concise.
"""

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        )

        return response.output_text

    except Exception as error:
        return (
            "LLM feedback failed. Basic feedback was used instead.\n\n"
            f"Error: {error}\n\n"
            + generate_basic_feedback(
                job_name=job_name,
                final_score=final_score,
                matched_skills=matched_skills,
                missing_skills=missing_skills
            )
        )
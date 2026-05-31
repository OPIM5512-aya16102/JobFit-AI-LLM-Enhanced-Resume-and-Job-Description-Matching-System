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
        feedback.append("Resume may need targeted tailoring, but review transferable skills before assuming weak fit.")

    feedback.append("\nInterview Preparation:")
    feedback.append("Be prepared to explain how your projects and past experience transfer to the job requirements.")

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
You are an expert technical recruiter, ML hiring manager, and resume coach.

Your job is to evaluate this candidate against the job description fairly.

Important evaluation rules:

1. Do NOT over-penalize missing exact keywords.
2. Do NOT assume the candidate lacks a skill only because the exact job-posting phrase is missing.
3. Look for transferable skills, related projects, and equivalent technical experience.
4. Distinguish between:
   - Direct matches
   - Transferable matches
   - Domain knowledge gaps
   - Tool-specific gaps
   - True technical gaps
5. Weight demonstrated project experience more heavily than keyword repetition.
6. Treat domain gaps differently from technical gaps.
   Example: If the job asks for fraud detection but the candidate has insurance risk modeling or predictive modeling experience, recognize that as transferable ML/risk experience.
7. Be realistic but not overly harsh.
8. Do not invent experience the candidate does not have.
9. Recommend truthful resume improvements only.
10. Explain how the candidate can position existing experience for the role.

Job Name:
{job_name}

Algorithmic Match Score:
{final_score}%

Matched Skills Detected:
{matched_skills}

Potential Missing Keywords Detected:
{missing_skills}

Resume:
{resume_text}

Job Description:
{job_text}

Return these sections:

1. Overall Match Assessment
   - Give a balanced assessment.
   - Mention whether gaps are mostly technical, domain-specific, or wording/keyword-related.

2. Direct Matches
   - List skills, tools, and experiences that directly align.

3. Transferable Strengths
   - Identify related experience that maps to the job even if exact keywords are missing.
   - Explain why each transferable skill matters.

4. Missing or Weak Areas
   - Separate true technical gaps from domain or terminology gaps.
   - Label each gap as Critical, Moderate, or Easily Learnable.

5. Resume Bullet Recommendations
   - Suggest bullet improvements based only on experience already present.
   - Do not fabricate experience.

6. Keywords to Add Truthfully
   - Suggest keywords only if supported by the resume or projects.

7. Interview Questions to Practice
   - Include technical, project-based, and domain-transfer questions.

8. Hiring Manager Concerns
   - Explain likely concerns and how the candidate can address them.

9. Final Recommendation
   - State whether this is a strong, moderate, or stretch application.
   - Consider both direct matches and transferable skills.

Keep the response practical, specific, and concise.
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
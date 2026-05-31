import pandas as pd

from preprocess import clean_text
from similarity import calculate_final_score
from skill_extractor import compare_skills


def calculate_weighted_final_score(
    tfidf_score,
    semantic_score,
    skill_coverage
):
    return round(
        (0.30 * tfidf_score)
        + (0.40 * semantic_score)
        + (0.30 * skill_coverage),
        2
    )


def rank_jobs(
    resume_text,
    jobs
):
    """
    jobs = {
        "job1.txt": "...",
        "job2.txt": "...",
    }
    """

    clean_resume = clean_text(resume_text)

    results = []

    for job_name, job_text in jobs.items():

        clean_job = clean_text(job_text)

        scores = calculate_final_score(
            clean_resume,
            clean_job
        )

        skill_results = compare_skills(
            clean_resume,
            clean_job
        )

        tfidf_score = round(
            scores["tfidf_score"] * 100,
            2
        )

        semantic_score = round(
            scores["embedding_score"] * 100,
            2
        )

        total_skills = (
            len(skill_results["matched_skills"])
            + len(skill_results["missing_skills"])
        )

        if total_skills > 0:
            skill_coverage = round(
                len(skill_results["matched_skills"])
                / total_skills
                * 100,
                2
            )
        else:
            skill_coverage = 0

        final_score = calculate_weighted_final_score(
            tfidf_score,
            semantic_score,
            skill_coverage
        )

        results.append(
            {
                "job_name": job_name,
                "tfidf_score": tfidf_score,
                "semantic_score": semantic_score,
                "skill_coverage": skill_coverage,
                "final_match_score": final_score,
                "matched_skills": len(
                    skill_results["matched_skills"]
                ),
                "missing_skills": len(
                    skill_results["missing_skills"]
                )
            }
        )

    results_df = pd.DataFrame(results)

    return results_df.sort_values(
        "final_match_score",
        ascending=False
    )
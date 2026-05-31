import os
import pandas as pd

from resume_parser import load_resume, load_job_descriptions
from preprocess import clean_text
from similarity import calculate_final_score
from skill_extractor import compare_skills
from llm_feedback import generate_basic_feedback


if __name__ == "__main__":

    os.makedirs("outputs/feedback", exist_ok=True)

    resume_name, resume_text = load_resume()
    job_descriptions = load_job_descriptions()

    clean_resume = clean_text(resume_text)

    results = []

    print("\n==============================")
    print("RESUME MATCH RESULTS")
    print("==============================")
    print(f"Resume: {resume_name}")

    for job_name, job_text in job_descriptions.items():

        clean_job = clean_text(job_text)

        if len(clean_job) == 0:
            print(f"\nSkipping {job_name}: empty job description.")
            continue

        scores = calculate_final_score(clean_resume, clean_job)
        skill_results = compare_skills(clean_resume, clean_job)

        matched_skills = ", ".join(skill_results["matched_skills"])
        missing_skills = ", ".join(skill_results["missing_skills"])

        row = {
            "resume": resume_name,
            "job_description": job_name,
            "tfidf_score": round(scores["tfidf_score"] * 100, 2),
            "semantic_score": round(scores["embedding_score"] * 100, 2),
            "final_match_score": round(scores["final_score"] * 100, 2),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "num_matched_skills": len(skill_results["matched_skills"]),
            "num_missing_skills": len(skill_results["missing_skills"])
        }

        results.append(row)

        feedback = generate_basic_feedback(
            job_name=row["job_description"],
            final_score=row["final_match_score"],
            matched_skills=matched_skills,
            missing_skills=missing_skills
        )

        feedback_file_name = job_name.replace(".txt", "_feedback.txt")
        feedback_path = os.path.join("outputs/feedback", feedback_file_name)

        with open(feedback_path, "w", encoding="utf-8") as file:
            file.write(feedback)

        print("\n------------------------------")
        print(f"Job: {job_name}")
        print(f"Final Match Score: {row['final_match_score']}%")
        print(f"Matched Skills: {matched_skills}")
        print(f"Missing Skills: {missing_skills}")
        print(f"Feedback saved to: {feedback_path}")

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="final_match_score",
        ascending=False
    )

    output_path = "outputs/resume_match_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\n==============================")
    print("RESULTS SAVED")
    print("==============================")
    print(f"Saved to: {output_path}")
    print("Feedback files saved to: outputs/feedback/")
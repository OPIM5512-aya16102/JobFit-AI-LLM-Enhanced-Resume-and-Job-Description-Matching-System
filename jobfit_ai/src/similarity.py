from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def calculate_tfidf_similarity(resume_text, job_text):
    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    return similarity


def calculate_embedding_similarity(resume_text, job_text):
    embeddings = embedding_model.encode([resume_text, job_text])

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return similarity


def calculate_final_score(resume_text, job_text):
    tfidf_score = calculate_tfidf_similarity(resume_text, job_text)
    embedding_score = calculate_embedding_similarity(resume_text, job_text)

    final_score = (0.4 * tfidf_score) + (0.6 * embedding_score)

    return {
        "tfidf_score": tfidf_score,
        "embedding_score": embedding_score,
        "final_score": final_score
    }


if __name__ == "__main__":
    resume = "python sql machine learning pandas scikit learn"
    job = "machine learning engineer role using python sql tensorflow aws"

    scores = calculate_final_score(resume, job)

    print(scores)
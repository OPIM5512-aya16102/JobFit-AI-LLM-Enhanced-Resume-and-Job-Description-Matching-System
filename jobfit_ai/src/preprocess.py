import re


def clean_text(text):
    """
    Basic NLP preprocessing.
    """

    # lowercase
    text = text.lower()

    # remove urls
    text = re.sub(r"http\S+", "", text)

    # remove emails
    text = re.sub(r"\S+@\S+", "", text)

    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


if __name__ == "__main__":

    sample = """
    Ahmed Ahmed
    Data Scientist
    Email: test@gmail.com
    Python, SQL, Machine Learning!
    """

    print(clean_text(sample))
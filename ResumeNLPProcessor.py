import re
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class ResumeNLPProcessor:
    def __init__(self, resumes, filenames):
        self.resumes = resumes
        self.filenames = filenames
        self.tfidf = joblib.load("saved_models/tfidf_vectorizer.pkl")
        self.model = joblib.load("saved_models/classifier_model.pkl")
        self.encoder = joblib.load("saved_models/label_encoder.pkl")
        self.bert_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.cleaned_resumes = [self._clean_text(r) for r in resumes]
        self.tfidf_vectors = self.tfidf.transform(self.cleaned_resumes)
        self.bert_vectors = self.bert_model.encode(resumes)

    def _clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def rank_resumes(self, job_description, method="tfidf", top_k=5):
        job_clean = self._clean_text(job_description)

        if method == "tfidf":
            job_vec = self.tfidf.transform([job_clean])
            similarities = cosine_similarity(job_vec, self.tfidf_vectors).flatten()
        elif method == "bert":
            job_vec = self.bert_model.encode([job_clean])[0]
            similarities = cosine_similarity([job_vec], self.bert_vectors).flatten()
        else:
            raise ValueError("Invalid method. Use 'tfidf' or 'bert'.")

        top_indices = similarities.argsort()[::-1][:top_k]
        results = []
        for i in top_indices:
            clean_text = self.cleaned_resumes[i]
            pred = self.model.predict(self.tfidf.transform([clean_text]))[0]
            target = self.encoder.inverse_transform([pred])[0]
            results.append({
                "Filename": self.filenames[i],
                "Similarity_Score": float(round(similarities[i], 4)),
                "Predicted_Domain": target
            })
        return results

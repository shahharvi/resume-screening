# backend/ResumeNLPProcessor.py

import numpy as np
import pandas as pd
import joblib
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

class ResumeNLPProcessor:
    def __init__(self, resumes):
        self.resumes = resumes
        self.df = pd.DataFrame({"ResumeText": resumes})
        self.df.fillna("", inplace=True)

        # Load TF-IDF
        self.tfidf = joblib.load("saved_models/tfidf_vectorizer.pkl")

        # Load BERT
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.bert_model.eval()

        # Precompute vectors
        self.tfidf_vectors = self.tfidf.transform(self.df["ResumeText"])
        self.bert_vectors = np.vstack([self.get_bert_vector(text) for text in self.df["ResumeText"]])

    def get_bert_vector(self, text):
        inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy().flatten()

    def rank_resumes(self, job_description, method="tfidf", top_k=5):
        if method == "tfidf":
            jd_vec = self.tfidf.transform([job_description])
            sims = cosine_similarity(jd_vec, self.tfidf_vectors).flatten()
        elif method == "bert":
            jd_vec = self.get_bert_vector(job_description)
            sims = cosine_similarity([jd_vec], self.bert_vectors).flatten()
        else:
            raise ValueError("Method must be 'tfidf' or 'bert'")

        top_indices = sims.argsort()[::-1][:top_k]
        return [
            {
                "Rank": i + 1,
                "ResumeIndex": int(idx),
                "SimilarityScore": float(sims[idx])
            }
            for i, idx in enumerate(top_indices)
        ]

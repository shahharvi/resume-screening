import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import joblib

class ResumeNLPProcessor:
    def __init__(self):
        self.tfidf = joblib.load("saved_models/tfidf_vectorizer.pkl")
        self.model = joblib.load("saved_models/classifier_model.pkl")
        self.label_encoder = joblib.load("saved_models/label_encoder.pkl")

        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.bert_model.eval()

        self.resume_texts = []
        self.filenames = []

    def set_resumes(self, texts, filenames):
        self.resume_texts = texts
        self.filenames = filenames

    def _get_bert_embedding(self, text):
        inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy().flatten()

    def rank_resumes(self, job_description, method="tfidf", top_k=5):
        job_text = job_description.strip()
        results = []

        if method == "tfidf":
            all_texts = [job_text] + self.resume_texts
            vectors = self.tfidf.transform(all_texts).toarray()
            jd_vec, resume_vecs = vectors[0], vectors[1:]
        elif method == "bert":
            jd_vec = self._get_bert_embedding(job_text)
            resume_vecs = [self._get_bert_embedding(text) for text in self.resume_texts]
        else:
            return []

        sims = cosine_similarity([jd_vec], resume_vecs).flatten()

        # Predict roles
        tfidf_resume_vecs = self.tfidf.transform(self.resume_texts)
        predictions = self.model.predict(tfidf_resume_vecs)
        predicted_labels = self.label_encoder.inverse_transform(predictions)

        for i in range(len(self.resume_texts)):
            results.append({
                "Filename": self.filenames[i],
                "Predicted_Label": predicted_labels[i],
                "Similarity_Score": float(sims[i])
            })

        # Return top K
        results.sort(key=lambda x: x["Similarity_Score"], reverse=True)
        return results[:top_k]

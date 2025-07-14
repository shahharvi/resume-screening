# backend/train_model.py

import pandas as pd
import re
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score


# 1. Load Dataset

df = pd.read_csv("data.csv")
print(f"Loaded {len(df)} resumes")

# Fill only text columns with empty string if null
for col in ["Summary", "Skills", "Certifications", "Experience", "Education", "Target"]:
    if col in df.columns:
        df[col] = df[col].fillna("").astype(str)


# 2. Text Preprocessing

text_columns = ["Summary", "Skills", "Certifications", "Experience", "Education"]
df["Text"] = df[text_columns].agg(" ".join, axis=1)

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()

df["Cleaned"] = df["Text"].apply(clean_text)


# 3. TF-IDF Vectorization

print("Generating TF-IDF vectors...")
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df["Cleaned"])

# 4. Encode Target Labels

print("Encoding labels...")
le = LabelEncoder()
y = le.fit_transform(df["Target"])


# 5. Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)
print(f" Training on {X_train.shape[0]} samples, Testing on {X_test.shape[0]} samples")


# 6. Train Classifier

print("Training Logistic Regression...")
clf = LogisticRegression(max_iter=1000, verbose=1)
clf.fit(X_train, y_train)

# 7. Evaluation

y_pred = clf.predict(X_test)
print("\n Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print(" Accuracy:", accuracy_score(y_test, y_pred))


# 8. Save Artifacts

os.makedirs("saved_models", exist_ok=True)
joblib.dump(tfidf, "saved_models/tfidf_vectorizer.pkl")
joblib.dump(clf, "saved_models/classifier_model.pkl")
joblib.dump(le, "saved_models/label_encoder.pkl")
print("✅ Model + Vectorizer + LabelEncoder saved in backend/saved_models/")

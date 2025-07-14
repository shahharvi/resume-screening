# 🤖 AI-Powered Resume Screening System

A smart resume ranking system that uses NLP and ML techniques to screen multiple resumes against a given job description in real time.

---

## 📌 Features

- 📄 Upload multiple resumes (PDF)
- 📑 Upload job description
- 🤝 Match resumes using:
  - TF-IDF
  - BERT embeddings
- 📊 Rank resumes based on similarity score
- ⚡ Real-time, accurate matching
- 💻 Streamlit frontend + Flask backend

---

## 🚀 Technologies Used

| Layer        | Stack                              |
|--------------|-------------------------------------|
| Frontend     | Streamlit                          |
| Backend      | Flask                              |
| NLP Models   | TF-IDF, BERT (HuggingFace)         |
| ML           | Logistic Regression                |
| Libraries    | scikit-learn, sentence-transformers, pandas, NumPy, PyMuPDF |

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/resume-screening.git
cd resume-screening

# Create a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate    # For Windows

# Install dependencies
pip install -r requirements.txt
python app.py
streamlit run streamlit_app.py

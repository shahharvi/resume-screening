from flask import Flask, request, jsonify
from ResumeNLPProcessor import ResumeNLPProcessor
import fitz  # PyMuPDF

app = Flask(__name__)
processor = ResumeNLPProcessor()
uploaded_files = []

def extract_text_from_pdf(file_stream):
    text = ""
    with fitz.open(stream=file_stream, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

@app.route("/upload", methods=["POST"])
def upload_resumes():
    global uploaded_files
    uploaded_files = request.files.getlist("files")
    texts = [extract_text_from_pdf(f.read()) for f in uploaded_files]
    processor.set_resumes(texts, [f.filename for f in uploaded_files])
    return jsonify({"message": f"{len(uploaded_files)} resumes processed."}), 200

@app.route("/rank", methods=["POST"])
def rank():
    data = request.get_json()
    jd = data.get("job_description", "")
    method = data.get("method", "tfidf")
    top_k = int(data.get("top_k", 5))

    results = processor.rank_resumes(jd, method=method, top_k=top_k)
    return jsonify(results), 200

if __name__ == "__main__":
    app.run(debug=True)

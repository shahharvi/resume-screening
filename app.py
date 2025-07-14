from flask import Flask, request, jsonify
from ResumeNLPProcessor import ResumeNLPProcessor
import fitz  # PyMuPDF
from io import BytesIO

app = Flask(__name__)
processor = None
resume_texts = []
resume_filenames = []

def extract_text_from_pdf(file_bytes):
    text = ""
    with fitz.open(stream=BytesIO(file_bytes), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

@app.route("/upload", methods=["POST"])
def upload_resumes():
    global resume_texts, resume_filenames, processor
    uploaded_files = request.files.getlist("files")
    if not uploaded_files:
        return jsonify({"error": "No files uploaded"}), 400

    resume_texts = []
    resume_filenames = []

    for f in uploaded_files:
        try:
            text = extract_text_from_pdf(f.read())
            if text:
                resume_texts.append(text)
                resume_filenames.append(f.filename)
        except Exception as e:
            print(f"❌ Failed to process {f.filename}: {e}")

    try:
        processor = ResumeNLPProcessor(resume_texts, resume_filenames)
        return jsonify({"message": f"{len(resume_texts)} resumes uploaded"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/rank", methods=["POST"])
def rank():
    global processor
    if not processor:
        return jsonify({"error": "Resumes not uploaded yet"}), 400

    try:
        data = request.get_json()
        job_description = data.get("job_description", "")
        method = data.get("method", "tfidf")
        top_k = int(data.get("top_k", 5))
        ranked = processor.rank_resumes(job_description, method=method, top_k=top_k)
        return jsonify(ranked), 200
    except Exception as e:
        return jsonify({"error": f"Failed to rank resumes: {e}"}), 500

if __name__ == "__main__":
    print("✅ Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True)

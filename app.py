# app.py

from flask import Flask, request, jsonify
from ResumeNLPProcessor import ResumeNLPProcessor

app = Flask(__name__)

processor = None
resumes = []
filenames = []

@app.route("/upload", methods=["POST"])
def upload():
    global resumes, processor, filenames
    files = request.files.getlist("files")
    resumes = [f.read().decode("utf-8", errors="ignore") for f in files]
    filenames = [f.filename for f in files]
    processor = ResumeNLPProcessor(resumes)
    return jsonify({"message": f"{len(resumes)} resumes uploaded."}), 200

@app.route("/rank", methods=["POST"])
def rank():
    global processor, filenames
    if processor is None:
        return jsonify({"error": "Resumes not uploaded yet."}), 400
    data = request.get_json()
    jd = data.get("job_description", "")
    method = data.get("method", "tfidf")
    top_k = int(data.get("top_k", 5))
    results = processor.rank_resumes(jd, method, top_k)
    for r in results:
        r["Filename"] = filenames[r["ResumeIndex"]]
    return jsonify(results), 200

if __name__ == "__main__":
    app.run(debug=True)

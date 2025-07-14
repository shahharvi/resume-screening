import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Resume Ranking System", layout="centered")
st.title("📄 AI Resume Ranking Based on Job Description")

st.markdown("Upload multiple resumes (PDF), enter a job description, and select a matching method.")

uploaded_files = st.file_uploader("📎 Upload Resumes", type=["pdf"], accept_multiple_files=True)
job_description = st.text_area("💼 Job Description", height=150)
method = st.selectbox("⚙️ Select Matching Method", ["tfidf", "bert"])

if st.button("🚀 Rank Resumes"):
    if not uploaded_files or not job_description.strip():
        st.warning("⚠️ Please upload resumes and provide a job description.")
    else:
        with st.spinner("⏳ Uploading and processing resumes..."):
            files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files]
            try:
                upload_resp = requests.post("http://localhost:5000/upload", files=files)
                if upload_resp.status_code != 200:
                    st.error(f"❌ Upload failed: {upload_resp.text}")
                else:
                    top_k = len(uploaded_files)
                    rank_payload = {
                        "job_description": job_description,
                        "method": method,
                        "top_k": top_k
                    }
                    rank_resp = requests.post("http://localhost:5000/rank", json=rank_payload)
                    try:
                        results = rank_resp.json()
                        if rank_resp.status_code == 200:
                            df = pd.DataFrame(results)
                            st.success(f"✅ Top {top_k} resumes ranked using `{method}` method")
                            st.dataframe(df)
                        else:
                            st.error(results.get("error", "Failed to rank resumes"))
                    except Exception:
                        st.error("❌ Failed to parse response from backend.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")

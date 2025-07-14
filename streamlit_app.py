import streamlit as st
import requests

st.set_page_config(page_title="Resume Screening", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>📄 Resume Screening System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Match resumes with job description using Machine Learning</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Upload resumes ---
st.subheader("📤 Upload Resume PDFs")
uploaded_files = st.file_uploader("Upload multiple resumes", type=["pdf"], accept_multiple_files=True)

# --- Job Description ---
st.subheader("📝 Job Description")
job_description = st.text_area("Paste your job description here", height=180, placeholder="Looking for a Python developer skilled in ML, NLP, etc.")

# --- Method Selection ---
method = st.radio("⚙️ Select Matching Method", options=["tfidf", "bert"], horizontal=True)


# --- Top-K Slider ---
max_k = len(uploaded_files)

if max_k > 1:
    top_k = st.slider("🔢 Top N Resumes to Show", min_value=1, max_value=max_k, value=min(5, max_k))
elif max_k == 1:
    top_k = 1
    st.info("ℹ️ Only one resume uploaded, ranking top 1.")
else:
    top_k = 0


# --- Submit ---
if st.button("🚀 Rank Resumes"):
    if not uploaded_files or not job_description.strip():
        st.warning("⚠️ Please upload resumes and enter a job description.")
    else:
        with st.spinner("Matching resumes..."):
            try:
                files = [("files", (file.name, file.read(), file.type)) for file in uploaded_files]
                upload_response = requests.post("http://localhost:5000/upload", files=files)

                if upload_response.status_code != 200:
                    st.error("❌ Upload failed.")
                else:
                    payload = {
                        "job_description": job_description,
                        "method": method,
                        "top_k": top_k
                    }
                    rank_response = requests.post("http://localhost:5000/rank", json=payload)

                    if rank_response.status_code == 200:
                        results = rank_response.json()
                        st.success("✅ Ranking Complete!")
                        st.markdown("---")

                        for idx, res in enumerate(results, 1):
                            st.markdown(f"""
                                <div style="background-color:#f9f9f9;padding:15px;border-radius:10px;margin-bottom:10px;">
                                <h4>📄 Resume {idx}: {res.get('Filename', 'N/A')}</h4>
                                <b>🔖 Predicted Role:</b> {res.get("Predicted_Label", "N/A")}<br>
                                <b>📊 Similarity Score:</b> {res.get("Similarity_Score", "N/A")}
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error("❌ Ranking failed.")
            except Exception as e:
                st.error(f"⚠️ Unexpected error: {e}")

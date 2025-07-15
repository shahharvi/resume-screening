import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="📄 Resume Screening System", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>📄 Resume Screening System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Match resumes with job description using Machine Learning</p>", unsafe_allow_html=True)
st.markdown("---")

# Upload resumes
st.subheader("📤 Upload Resume PDFs")
uploaded_files = st.file_uploader("Upload multiple resumes", type=["pdf"], accept_multiple_files=True)

# Job description
st.subheader("📝 Job Description")
job_description = st.text_area("Paste your job description here", height=180, placeholder="Looking for a Python developer skilled in ML, NLP, etc.")

# Method selection
method = st.radio("⚙️ Select Matching Method", options=["tfidf", "bert"], horizontal=True)

# Top-K
max_k = len(uploaded_files)
if max_k > 1:
    top_k = st.slider("🔢 Top N Resumes to Show", min_value=1, max_value=max_k, value=min(5, max_k))
elif max_k == 1:
    top_k = 1
    st.info("ℹ️ Only one resume uploaded, showing top 1.")
else:
    top_k = 0

# Submit
if st.button("🚀 Rank Resumes"):
    if not uploaded_files or not job_description.strip():
        st.warning("⚠️ Please upload resumes and enter a job description.")
    else:
        with st.spinner("🔍 Matching resumes..."):
            try:
                # Upload resumes to backend
                files = [("files", (file.name, file.read(), file.type)) for file in uploaded_files]
                upload_response = requests.post("http://localhost:5000/upload", files=files)

                if upload_response.status_code != 200:
                    st.error("❌ Upload failed.")
                else:
                    # Rank resumes
                    payload = {
                        "job_description": job_description,
                        "method": method,
                        "top_k": top_k
                    }
                    rank_response = requests.post("http://localhost:5000/rank", json=payload)

                    if rank_response.status_code == 200:
                        results = rank_response.json()
                        st.success("✅ Ranking Complete!")

                        # Convert to DataFrame
                        df = pd.DataFrame(results)

                        # Add ranking
                        df["Rank"] = df["Similarity_Score"].rank(method="first", ascending=False).astype(int)
                        df = df.sort_values("Rank")

                        # Display
                        st.markdown("### 📊 Ranked Resumes")
                        st.dataframe(df[["Rank", "Filename", "Predicted_Label", "Similarity_Score"]],
                                     use_container_width=True, hide_index=True)
                    else:
                        st.error("❌ Ranking failed.")
            except Exception as e:
                st.error(f"⚠️ Unexpected error: {e}")

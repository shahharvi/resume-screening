import os
import re
import pandas as pd
from pdfminer.high_level import extract_text

# Base path to resumes organized by folders (e.g., ACCOUNTANT/, ENGINEER/)
BASE_PATH = "dataset/resumes/"  # change as per your structure

# Function to extract each resume field
def extract_fields(text):
    text = re.sub(r'\s+', ' ', text)

    def extract_section(start_keywords, end_keywords):
        pattern = re.compile(rf"({'|'.join(start_keywords)})(.*?)(?={'|'.join(end_keywords)}|$)", re.IGNORECASE)
        match = pattern.search(text)
        return match.group(2).strip() if match else f"No {'/'.join(start_keywords).lower()} available"

    return {
        "Summary": extract_section(["Summary", "Objective"], ["Skills", "Certifications", "Experience", "Education"]),
        "Certifications": extract_section(["Certifications", "Licenses"], ["Skills", "Experience", "Education"]),
        "Skills": extract_section(["Skills", "Technical Skills", "Highlights"], ["Experience", "Education"]),
        "Experience": extract_section(["Experience", "Work History", "Professional Experience"], ["Education", "Projects"]),
        "Education": extract_section(["Education", "Academics", "Academic Background"], ["Skills", "Experience", "Certifications"])
    }

# Build dataset by scanning folders
def build_dataset(base_path):
    data = []
    for folder in os.listdir(base_path):
        role_path = os.path.join(base_path, folder)
        if os.path.isdir(role_path):
            for filename in os.listdir(role_path):
                if filename.endswith(".pdf"):
                    full_path = os.path.join(role_path, filename)
                    try:
                        text = extract_text(full_path)
                        fields = extract_fields(text)
                        fields["Target"] = folder  # Label
                        data.append(fields)
                        print(f"[✓] Parsed: {filename}")
                    except Exception as e:
                        print(f"[✗] Failed: {filename} ({e})")
    return pd.DataFrame(data)

# Main execution
if __name__ == "__main__":
    df = build_dataset(BASE_PATH)
    df.to_csv("data.csv", index=False)
    print("\n✅ Resume dataset saved as 'data.csv'")


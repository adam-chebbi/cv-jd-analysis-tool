import streamlit as st
import os
import logging
from src.extractor import SkillExtractor
from src.matcher import SkillMatcher, MatchResult
from typing import List, Tuple

# Set up logging
logging.basicConfig(
    filename=os.path.join("logs", 'app.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_extractor():
    """Cache the SkillExtractor instance."""
    try:
        extractor = SkillExtractor()
        logger.info("Loaded SkillExtractor")
        return extractor
    except Exception as e:
        logger.error(f"Failed to load SkillExtractor: {str(e)}")
        st.error("Error loading NLP model. Please check logs.")
        return None

@st.cache_resource
def load_matcher():
    """Cache the SkillMatcher instance."""
    try:
        matcher = SkillMatcher()
        logger.info("Loaded SkillMatcher")
        return matcher
    except Exception as e:
        logger.error(f"Failed to load SkillMatcher: {str(e)}")
        st.error("Error loading matcher. Please check logs.")
        return None

def save_uploaded_file(uploaded_file, upload_dir: str = "uploads/") -> str:
    """Save uploaded file to disk and return path."""
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    logger.info(f"Saved file: {file_path}")
    return file_path

def main():
    st.title("CV/JD Analysis and Selection Tool")
    st.markdown("Upload CVs and a JD to extract skills and match semantically.")

    extractor = load_extractor()
    matcher = load_matcher()
    if not extractor or not matcher:
        st.stop()

    # JD Upload
    st.header("Upload Job Description (JD)")
    jd_file = st.file_uploader("Choose JD file (PDF or TXT)", type=["pdf", "txt"])
    if jd_file:
        jd_path = save_uploaded_file(jd_file)
        jd_skills = extractor.process_jd(jd_path)
        if jd_skills:
            st.success("JD skills extracted successfully!")
            st.subheader("Extracted JD Skills")
            st.write(jd_skills)
        else:
            st.error("Failed to extract skills from JD. Check file format.")

    # CV Uploads (Multiple)
    st.header("Upload CVs (Multiple PDFs)")
    cv_files = st.file_uploader("Choose CV PDFs", type="pdf", accept_multiple_files=True)
    cv_skills_list: List[Tuple[str, List[str]]] = []
    if cv_files:
        for cv_file in cv_files:
            cv_path = save_uploaded_file(cv_file)
            cv_skills = extractor.process_cv(cv_path)
            if cv_skills:
                cv_skills_list.append((cv_file.name, cv_skills))
        if cv_skills_list:
            st.success(f"{len(cv_skills_list)} CVs processed successfully!")

    # Matching Button
    if st.button("Perform Matching") and jd_skills and cv_skills_list:
        with st.spinner("Matching CVs to JD..."):
            try:
                ranked_results = matcher.rank_cvs_against_jd(cv_skills_list, jd_skills)
                if ranked_results:
                    st.success("Matching complete!")
                    st.subheader("Ranked CV Matches")
                    
                    # Display results in a table
                    data = []
                    for result in ranked_results:
                        data.append({
                            "CV Name": result.cv_id,
                            "Similarity Score": f"{result.similarity_score:.2f}",
                            "Matched Skills": ", ".join(result.matched_skills),
                            "CV Skills Count": result.total_cv_skills,
                            "JD Skills Count": result.total_jd_skills
                        })
                    
                    st.table(data)
                    
                    # Visual: Progress bars for scores
                    st.subheader("Similarity Scores Visualization")
                    for result in ranked_results:
                        st.write(f"**{result.cv_id}**")
                        st.progress(result.similarity_score)
                else:
                    st.warning("No CVs matched above the similarity threshold.")
            except Exception as e:
                logger.error(f"Matching error: {str(e)}")
                st.error("An error occurred during matching. Check logs.")

if __name__ == "__main__":
    main()
import streamlit as st
import os
import logging
import pandas as pd
from src.extractor import SkillExtractor
from src.matcher import SkillMatcher, MatchResult
from typing import List, Tuple, Optional
from joblib import dump, load

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

def get_cached_skills(file_name: str, cache_dir: str = "cache/") -> Optional[List[str]]:
    """Load cached skills if available."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{file_name}_skills.pkl")
    if os.path.exists(cache_path):
        try:
            skills = load(cache_path)
            logger.info(f"Loaded cached skills for {file_name}")
            return skills
        except Exception as e:
            logger.error(f"Failed to load cache for {file_name}: {str(e)}")
    return None

def cache_skills(file_name: str, skills: List[str], cache_dir: str = "cache/"):
    """Cache extracted skills to disk."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{file_name}_skills.pkl")
    dump(skills, cache_path)
    logger.info(f"Cached skills for {file_name}")

def main():
    st.title("CV/JD Analysis and Selection Tool")
    st.markdown("Upload CVs and a JD to extract skills and match semantically.")

    extractor = load_extractor()
    matcher = load_matcher()
    if not extractor or not matcher:
        st.stop()

    # Configuration: Similarity Threshold
    st.sidebar.header("Settings")
    similarity_threshold = st.sidebar.slider(
        "Similarity Threshold", 0.0, 1.0, matcher.config['matcher']['similarity_threshold'], 0.05
    )
    matcher.config['matcher']['similarity_threshold'] = similarity_threshold

    # JD Upload
    st.header("Upload Job Description (JD)")
    jd_file = st.file_uploader("Choose JD file (PDF or TXT)", type=["pdf", "txt"])
    jd_skills = []
    if jd_file:
        jd_path = save_uploaded_file(jd_file)
        cached_jd_skills = get_cached_skills(jd_file.name)
        if cached_jd_skills:
            jd_skills = cached_jd_skills
        else:
            jd_skills = extractor.process_jd(jd_path)
            if jd_skills:
                cache_skills(jd_file.name, jd_skills)
        if jd_skills:
            st.success("JD skills extracted successfully!")
            st.subheader("Extracted JD Skills")
            st.write(jd_skills)
        else:
            st.error("Failed to extract skills from JD. Check file format or content.")

    # CV Uploads (Multiple)
    st.header("Upload CVs (Multiple PDFs)")
    cv_files = st.file_uploader("Choose CV PDFs", type="pdf", accept_multiple_files=True)
    cv_skills_list: List[Tuple[str, List[str]]] = []
    if cv_files:
        texts = []
        file_names = []
        is_jd_list = [False] * len(cv_files)
        for cv_file in cv_files:
            cv_path = save_uploaded_file(cv_file)
            cached_cv_skills = get_cached_skills(cv_file.name)
            if cached_cv_skills:
                cv_skills_list.append((cv_file.name, cached_cv_skills))
            else:
                text = extractor.extract_text_from_pdf(cv_path)
                if text:
                    texts.append(text)
                    file_names.append(cv_file.name)
        if texts:
            batch_skills = extractor.batch_extract_skills(texts, is_jd_list)
            for name, skills in zip(file_names, batch_skills):
                if skills:
                    cache_skills(name, skills)
                    cv_skills_list.append((name, skills))
        if cv_skills_list:
            st.success(f"{len(cv_skills_list)} CVs processed successfully!")

    # Matching Button
    if st.button("Perform Matching", key="perform_matching"):
        if jd_skills and cv_skills_list:
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
                        
                        df = pd.DataFrame(data)
                        st.table(df)
                        
                        # Download results as CSV
                        csv = df.to_csv(index=False)
                        st.download_button("Download Results", csv, "cv_jd_matches.csv", "text/csv")
                        
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
        else:
            st.error("Please upload a JD and at least one CV before matching.", key="error_no_files")

if __name__ == "__main__":
    main()
from src.matcher import SkillMatcher

def main():
    matcher = SkillMatcher()
    
    # Sample skills (from previous tests)
    sample_cv_skills = ["python", "java", "project management", "data analysis"]
    sample_jd_skills = ["python", "javascript", "sql", "teamwork", "data analysis"]
    
    # Test single match
    result = matcher.match_cv_to_jd(sample_cv_skills, sample_jd_skills, "CV_001")
    if result:
        print(f"Match Result for {result.cv_id}:")
        print(f"  Similarity Score: {result.similarity_score:.2f}")
        print(f"  Matched Skills: {result.matched_skills}")
        print(f"  Total CV Skills: {result.total_cv_skills}, JD Skills: {result.total_jd_skills}")
    else:
        print("No match above threshold")
    
    # Test ranking multiple CVs
    cv_list = [
        ("CV_001", sample_cv_skills),
        ("CV_002", ["java", "c++", "communication"]),  # Lower match
        ("CV_003", ["python", "machine learning", "sql"])  # High match
    ]
    ranked_results = matcher.rank_cvs_against_jd(cv_list, sample_jd_skills)
    print("\nRanked CVs:")
    for res in ranked_results:
        print(f"  {res.cv_id}: Score {res.similarity_score:.2f}")

if __name__ == "__main__":
    main()
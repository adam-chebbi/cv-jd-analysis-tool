import spacy
import yaml
import logging
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class MatchResult:
    """Data class to hold matching results for reusability."""
    cv_id: str
    similarity_score: float
    matched_skills: List[str]
    total_cv_skills: int
    total_jd_skills: int

class SkillMatcher:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the SkillMatcher with configuration and Spacy model."""
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Set up logging
        os.makedirs(self.config['paths']['logs'], exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(self.config['paths']['logs'], 'matcher.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load Spacy model
        try:
            self.nlp = spacy.load(self.config['extractor']['model'])
            self.logger.info(f"Loaded Spacy model: {self.config['extractor']['model']}")
        except Exception as e:
            self.logger.error(f"Failed to load Spacy model: {str(e)}")
            raise

    def compute_similarity(self, cv_skills: List[str], jd_skills: List[str]) -> Tuple[float, List[str]]:
        """Compute semantic similarity between two lists of skills."""
        if not cv_skills or not jd_skills:
            self.logger.warning("Empty skill lists provided for similarity computation")
            return 0.0, []
        
        try:
            similarities = []
            matched_skills = []
            
            # Process skills into Spacy documents for vector similarity
            cv_doc = self.nlp(" ".join(cv_skills))
            jd_doc = self.nlp(" ".join(jd_skills))
            
            # Compute overall document similarity (semantic overlap)
            overall_similarity = cv_doc.similarity(jd_doc)
            
            # Pairwise skill matching for detailed results
            for cv_skill in cv_skills:
                cv_skill_doc = self.nlp(cv_skill)
                max_sim = 0.0
                best_match = None
                for jd_skill in jd_skills:
                    jd_skill_doc = self.nlp(jd_skill)
                    sim = cv_skill_doc.similarity(jd_skill_doc)
                    if sim > max_sim:
                        max_sim = sim
                        best_match = jd_skill
                if max_sim >= self.config['matcher']['similarity_threshold']:
                    similarities.append(max_sim)
                    matched_skills.append(f"{cv_skill} -> {best_match} (sim: {max_sim:.2f})")
            
            # Average pairwise similarities (if any matches)
            avg_pairwise_sim = sum(similarities) / len(similarities) if similarities else 0.0
            
            # Final score: weighted average of overall and pairwise (emphasize pairwise for precision)
            final_score = (overall_similarity + avg_pairwise_sim) / 2
            
            self.logger.info(f"Computed similarity: {final_score:.2f} between CV skills {cv_skills} and JD skills {jd_skills}")
            return final_score, matched_skills
        except Exception as e:
            self.logger.error(f"Error computing similarity: {str(e)}")
            return 0.0, []

    def match_cv_to_jd(self, cv_skills: List[str], jd_skills: List[str], cv_id: str = "CV_1") -> Optional[MatchResult]:
        """Match a single CV to a JD and return results."""
        score, matched = self.compute_similarity(cv_skills, jd_skills)
        if score >= self.config['matcher']['similarity_threshold']:
            return MatchResult(
                cv_id=cv_id,
                similarity_score=score,
                matched_skills=matched,
                total_cv_skills=len(cv_skills),
                total_jd_skills=len(jd_skills)
            )
        self.logger.info(f"CV {cv_id} similarity score {score:.2f} below threshold {self.config['matcher']['similarity_threshold']}")
        return None

    def rank_cvs_against_jd(self, cv_skills_list: List[Tuple[str, List[str]]], jd_skills: List[str]) -> List[MatchResult]:
        """Rank multiple CVs against a single JD."""
        results = []
        for cv_id, cv_skills in cv_skills_list:
            match_result = self.match_cv_to_jd(cv_skills, jd_skills, cv_id)
            if match_result:
                results.append(match_result)
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Return top N matches
        top_n = self.config['matcher']['top_n_matches']
        self.logger.info(f"Ranked {len(results)} CVs against JD, returning top {min(top_n, len(results))}")
        return results[:top_n]
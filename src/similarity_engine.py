"""
similarity_engine.py
--------------------
Computes similarity between resume and job description text using
TF-IDF vectorization and cosine similarity.

Also handles the skill gap analysis (matched vs missing skills)
and the composite scoring that combines everything into a final
match percentage.
"""

import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SimilarityEngine:
    """
    Computes various similarity and matching metrics between
    a resume and a job description.
    """
    
    def __init__(self, tfidf_params=None):
        if tfidf_params is None:
            from config import TFIDF_PARAMS
            tfidf_params = TFIDF_PARAMS
        
        self.tfidf_params = tfidf_params
        self.vectorizer = None
    
    def compute_tfidf_similarity(self, resume_text, jd_text):
        """
        Compute cosine similarity between resume and job description
        using TF-IDF vectors.
        
        Creates a new vectorizer each time since we're comparing
        just two documents. The vectorizer learns the vocabulary
        from both documents together.
        
        Returns a float between 0 and 1.
        """
        if not resume_text or not jd_text:
            logger.warning("Empty text provided for similarity computation.")
            return 0.0
        
        try:
            self.vectorizer = TfidfVectorizer(**self.tfidf_params)
            
            # fit and transform both documents together
            tfidf_matrix = self.vectorizer.fit_transform([resume_text, jd_text])
            
            # compute cosine similarity between the two vectors
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            
            score = float(similarity[0][0])
            
            # clamp to [0, 1] just in case of floating point weirdness
            return max(0.0, min(1.0, score))
        
        except Exception as e:
            logger.error(f"Error computing TF-IDF similarity: {e}")
            return 0.0
    
    def compute_skill_match(self, resume_skills, jd_skills):
        """
        Compare the skills found in the resume against those in the
        job description.
        
        Parameters
        ----------
        resume_skills : list
            Flat list of skills found in the resume.
        jd_skills : list
            Flat list of skills found in the job description.
        
        Returns
        -------
        dict with:
            - matched: skills present in both
            - missing: skills in JD but not in resume
            - extra: skills in resume but not in JD
            - match_percentage: what fraction of JD skills are covered
        """
        if not jd_skills:
            return {
                "matched": [],
                "missing": [],
                "extra": list(resume_skills) if resume_skills else [],
                "match_percentage": 0.0,
            }
        
        # normalize skill names for comparison
        resume_set = set(s.lower().strip() for s in resume_skills)
        jd_set = set(s.lower().strip() for s in jd_skills)
        
        # find intersections and differences
        matched_lower = resume_set & jd_set
        missing_lower = jd_set - resume_set
        extra_lower = resume_set - jd_set
        
        # map back to original casing for display
        # (build lookup dicts from the original lists)
        resume_lookup = {s.lower().strip(): s for s in resume_skills}
        jd_lookup = {s.lower().strip(): s for s in jd_skills}
        
        matched = [resume_lookup.get(s, s) for s in matched_lower]
        missing = [jd_lookup.get(s, s) for s in missing_lower]
        extra = [resume_lookup.get(s, s) for s in extra_lower]
        
        # percentage of JD skills that the resume covers
        match_pct = (len(matched) / len(jd_set)) * 100 if jd_set else 0.0
        
        return {
            "matched": sorted(matched),
            "missing": sorted(missing),
            "extra": sorted(extra),
            "match_percentage": round(match_pct, 1),
        }
    
    def compute_composite_score(self, tfidf_score, skill_match_pct,
                                 experience_years, education_level,
                                 required_experience=0, required_education=0):
        """
        Calculate a weighted composite score from all the individual
        components. This gives us a single number that represents
        how well the candidate matches the job.
        
        Parameters
        ----------
        tfidf_score : float
            Cosine similarity (0 to 1)
        skill_match_pct : float
            Percentage of required skills matched (0 to 100)
        experience_years : int
            Candidate's years of experience
        education_level : int
            Candidate's education level (1-5 scale)
        required_experience : int
            Required years of experience from JD
        required_education : int
            Required education level from JD
        
        Returns
        -------
        dict with overall score and component breakdown
        """
        from config import SCORING_WEIGHTS
        
        # normalize each component to 0-100 scale
        sim_score = tfidf_score * 100
        skill_score = skill_match_pct
        
        # experience score: ratio of actual to required, capped at 100
        if required_experience > 0:
            exp_score = min(100, (experience_years / required_experience) * 100)
        else:
            # if no specific requirement, give partial credit based on having any
            exp_score = min(100, experience_years * 20)  # 5+ years = full marks
        
        # education score
        if required_education > 0:
            edu_score = min(100, (education_level / required_education) * 100)
        else:
            edu_score = min(100, education_level * 25)  # level 4+ = full marks
        
        # weighted combination
        weights = SCORING_WEIGHTS
        overall = (
            sim_score * weights["tfidf_similarity"]
            + skill_score * weights["skill_match"]
            + exp_score * weights["experience"]
            + edu_score * weights["education"]
        )
        
        # clamp to 0-100
        overall = max(0, min(100, overall))
        
        return {
            "overall_score": round(overall, 1),
            "components": {
                "TF-IDF Similarity": round(sim_score, 1),
                "Skill Match": round(skill_score, 1),
                "Experience": round(exp_score, 1),
                "Education": round(edu_score, 1),
            },
            "weights": weights,
        }
    
    def get_top_tfidf_terms(self, text, top_n=15):
        """
        Get the top N terms by TF-IDF weight for a given text.
        Useful for showing what the model considers important.
        
        Note: must be called after compute_tfidf_similarity so that
        the vectorizer is fitted.
        """
        if self.vectorizer is None:
            return []
        
        try:
            tfidf_vector = self.vectorizer.transform([text])
            feature_names = self.vectorizer.get_feature_names_out()
            
            # get the TF-IDF scores and sort them
            scores = tfidf_vector.toarray().flatten()
            top_indices = scores.argsort()[-top_n:][::-1]
            
            terms = []
            for idx in top_indices:
                if scores[idx] > 0:
                    terms.append({
                        "term": feature_names[idx],
                        "score": round(float(scores[idx]), 4)
                    })
            
            return terms
        
        except Exception as e:
            logger.error(f"Error getting top TF-IDF terms: {e}")
            return []

"""
Tests for the similarity engine module.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from src.similarity_engine import SimilarityEngine


class TestSimilarityEngine(unittest.TestCase):
    
    def setUp(self):
        self.engine = SimilarityEngine()
    
    def test_identical_documents(self):
        text = "python machine learning data science deep learning neural networks"
        score = self.engine.compute_tfidf_similarity(text, text)
        # identical documents should have very high similarity
        self.assertGreaterEqual(score, 0.95)
    
    def test_completely_different(self):
        text1 = "python machine learning data science tensorflow keras"
        text2 = "cooking recipes italian pasta tomato sauce olive oil"
        score = self.engine.compute_tfidf_similarity(text1, text2)
        # unrelated documents should have very low similarity
        self.assertLess(score, 0.15)
    
    def test_partial_overlap(self):
        text1 = "python machine learning data science sql pandas"
        text2 = "python data analysis sql database excel reporting"
        score = self.engine.compute_tfidf_similarity(text1, text2)
        # should be somewhere in the middle
        self.assertGreater(score, 0.1)
        self.assertLess(score, 0.9)
    
    def test_score_range(self):
        text1 = "software developer with java spring boot experience"
        text2 = "looking for a java developer with spring framework knowledge"
        score = self.engine.compute_tfidf_similarity(text1, text2)
        # score should always be between 0 and 1
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_empty_text(self):
        score = self.engine.compute_tfidf_similarity("", "some text here")
        self.assertEqual(score, 0.0)
    
    def test_both_empty(self):
        score = self.engine.compute_tfidf_similarity("", "")
        self.assertEqual(score, 0.0)
    
    def test_skill_match_basic(self):
        resume_skills = ["Python", "Java", "SQL", "Docker"]
        jd_skills = ["Python", "Java", "AWS", "Docker", "Kubernetes"]
        
        result = self.engine.compute_skill_match(resume_skills, jd_skills)
        
        self.assertIn("python", [s.lower() for s in result["matched"]])
        self.assertIn("aws", [s.lower() for s in result["missing"]])
        self.assertGreater(result["match_percentage"], 0)
    
    def test_skill_match_all_matched(self):
        skills = ["Python", "Java", "SQL"]
        result = self.engine.compute_skill_match(skills, skills)
        self.assertEqual(result["match_percentage"], 100.0)
        self.assertEqual(len(result["missing"]), 0)
    
    def test_skill_match_none_matched(self):
        resume = ["Python", "Java"]
        jd = ["Rust", "Go", "Elixir"]
        result = self.engine.compute_skill_match(resume, jd)
        self.assertEqual(len(result["matched"]), 0)
        self.assertEqual(result["match_percentage"], 0.0)
    
    def test_skill_match_empty_jd(self):
        result = self.engine.compute_skill_match(["Python"], [])
        self.assertEqual(result["match_percentage"], 0.0)
    
    def test_composite_score_range(self):
        result = self.engine.compute_composite_score(
            tfidf_score=0.6,
            skill_match_pct=70,
            experience_years=3,
            education_level=3,
        )
        self.assertGreaterEqual(result["overall_score"], 0)
        self.assertLessEqual(result["overall_score"], 100)
    
    def test_top_tfidf_terms(self):
        resume = "python developer with machine learning and deep learning experience"
        jd = "looking for python machine learning engineer"
        # need to compute similarity first so vectorizer is fitted
        self.engine.compute_tfidf_similarity(resume, jd)
        terms = self.engine.get_top_tfidf_terms(resume, top_n=5)
        self.assertIsInstance(terms, list)
        self.assertLessEqual(len(terms), 5)


if __name__ == "__main__":
    unittest.main()

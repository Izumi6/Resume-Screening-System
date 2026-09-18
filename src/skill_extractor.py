"""
skill_extractor.py
------------------
Extracts skills, education, and work experience from resume text.

This uses a combination of pattern matching against our skills database
and some regex heuristics for education/experience. It's not as fancy
as a full NER model, but it works surprisingly well for most resumes
and is much easier to explain and debug.
"""

import re
import json
import os
import logging

logger = logging.getLogger(__name__)


class SkillExtractor:
    """
    Extracts structured information from resume text.
    
    Loads a skills database (JSON) and uses it to find mentions of
    known skills in the text. Also has regex-based extractors for
    education qualifications and years of experience.
    """
    
    def __init__(self, skills_db_path=None):
        if skills_db_path is None:
            # default path relative to project root
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            skills_db_path = os.path.join(base, "data", "skills_database.json")
        
        self.skills_db = self._load_skills_db(skills_db_path)
    
    def _load_skills_db(self, path):
        """Load the skills database from JSON file."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Skills database not found at {path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in skills database: {e}")
            return {}
    
    def extract_skills(self, text):
        """
        Find all skills mentioned in the text.
        
        Compares against our skills database using case-insensitive
        matching. Also handles common abbreviations and aliases
        (e.g., 'js' for 'javascript', 'ml' for 'machine learning').
        
        Returns a dict with skills grouped by category.
        """
        if not text:
            return {}
        
        text_lower = text.lower()
        found_skills = {}
        
        for category, skills_list in self.skills_db.items():
            matched = []
            for skill_entry in skills_list:
                # each entry can be a string or a dict with aliases
                if isinstance(skill_entry, dict):
                    skill_name = skill_entry["name"]
                    variants = skill_entry.get("aliases", []) + [skill_name]
                else:
                    skill_name = skill_entry
                    variants = [skill_entry]
                
                # check if any variant appears in the text
                for variant in variants:
                    variant_lower = variant.lower()
                    # use word boundary matching to avoid partial matches
                    # e.g., we don't want "r" matching inside "react"
                    pattern = r'\b' + re.escape(variant_lower) + r'\b'
                    if re.search(pattern, text_lower):
                        if skill_name not in matched:
                            matched.append(skill_name)
                        break  # found it, no need to check other variants
            
            if matched:
                found_skills[category] = matched
        
        return found_skills
    
    def extract_skills_flat(self, text):
        """
        Same as extract_skills but returns a flat list instead of
        grouped by category. Useful for quick comparisons.
        """
        categorized = self.extract_skills(text)
        flat = []
        for skills in categorized.values():
            flat.extend(skills)
        return list(set(flat))  # deduplicate
    
    def extract_education(self, text):
        """
        Extract education qualifications from the text.
        
        Looks for common degree patterns like B.Tech, M.Sc, MBA, PhD, etc.
        Returns a list of dicts with degree name and level (numeric).
        """
        if not text:
            return []
        
        # import here to avoid circular import issues
        from config import EDUCATION_LEVELS
        
        text_lower = text.lower()
        found_education = []
        
        # patterns for various degree formats
        degree_patterns = [
            (r'\b(?:ph\.?d\.?|doctorate)\b', "PhD", 5),
            (r'\b(?:m\.?tech|m\.?\s*tech)\b', "M.Tech", 4),
            (r'\b(?:m\.?s\.?c|m\.?\s*sc)\b', "M.Sc", 4),
            (r'\b(?:m\.?s\.?\b)', "M.S.", 4),
            (r'\b(?:m\.?b\.?a\.?|mba)\b', "MBA", 4),
            (r'\bmaster(?:s|\'s)?\s+(?:of|in)\b', "Masters", 4),
            (r'\b(?:b\.?tech|b\.?\s*tech)\b', "B.Tech", 3),
            (r'\b(?:b\.?e\.?\b)', "B.E.", 3),
            (r'\b(?:b\.?s\.?c|b\.?\s*sc)\b', "B.Sc", 3),
            (r'\b(?:b\.?c\.?a)\b', "BCA", 3),
            (r'\b(?:m\.?c\.?a)\b', "MCA", 4),
            (r'\bbachelor(?:s|\'s)?\s+(?:of|in)\b', "Bachelors", 3),
            (r'\b(?:diploma)\b', "Diploma", 2),
            (r'\b(?:high\s*school|12th|hsc|intermediate)\b', "High School", 1),
        ]
        
        seen_levels = set()
        for pattern, degree_name, level in degree_patterns:
            if re.search(pattern, text_lower):
                if level not in seen_levels:  # avoid duplicates at same level
                    found_education.append({
                        "degree": degree_name,
                        "level": level,
                    })
                    seen_levels.add(level)
        
        # sort by level descending (highest qualification first)
        found_education.sort(key=lambda x: x["level"], reverse=True)
        
        return found_education
    
    def extract_experience_years(self, text):
        """
        Try to estimate the candidate's years of experience.
        
        Uses regex patterns to find phrases like '5 years of experience',
        '3+ years', 'over 2 years', etc. Returns the maximum value found,
        or 0 if nothing is detected.
        """
        if not text:
            return 0
        
        text_lower = text.lower()
        
        patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp)',
            r'(?:experience|exp)\s*(?:of|:)?\s*(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:in\s+)',
            r'over\s+(\d+)\s+(?:years?|yrs?)',
            r'(\d+)\+\s*(?:years?|yrs?)',
        ]
        
        years_found = []
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            for m in matches:
                try:
                    val = int(m)
                    if 0 < val < 50:  # sanity check
                        years_found.append(val)
                except ValueError:
                    continue
        
        if years_found:
            return max(years_found)
        
        # fallback: count distinct year mentions (like 2019, 2020, 2021)
        # to roughly estimate experience duration
        year_mentions = re.findall(r'\b(20\d{2})\b', text)
        if len(year_mentions) >= 2:
            years = [int(y) for y in year_mentions]
            span = max(years) - min(years)
            if 0 < span < 40:
                return span
        
        return 0
    
    def get_highest_education_level(self, text):
        """
        Get the numeric level of the highest education found.
        Returns 0 if no education is detected.
        """
        education = self.extract_education(text)
        if education:
            return education[0]["level"]  # already sorted descending
        return 0

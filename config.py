"""
Configuration file for the Resume Screening System.

All the constants, file paths, scoring weights, and model parameters
are defined here so we don't have to hardcode stuff everywhere else.
"""

import os

# --- Project paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SKILLS_DB_PATH = os.path.join(DATA_DIR, "skills_database.json")
TRAINING_DATA_PATH = os.path.join(DATA_DIR, "training_data", "resume_dataset.csv")

# create directories if they don't exist yet
for _dir in [DATA_DIR, MODELS_DIR, os.path.join(DATA_DIR, "training_data")]:
    os.makedirs(_dir, exist_ok=True)


# --- TF-IDF parameters ---
TFIDF_PARAMS = {
    "max_features": 5000,
    "ngram_range": (1, 2),      # unigrams + bigrams
    "sublinear_tf": True,       # apply log normalization
    "min_df": 1,
    "stop_words": "english",
}


# --- Scoring weights ---
# these control how much each component contributes to the final score
SCORING_WEIGHTS = {
    "tfidf_similarity": 0.35,
    "skill_match": 0.35,
    "experience": 0.15,
    "education": 0.15,
}


# --- Education level mapping ---
# higher number = higher qualification
EDUCATION_LEVELS = {
    "high school": 1,
    "diploma": 2,
    "associate": 2,
    "bachelor": 3,
    "b.tech": 3,
    "b.e.": 3,
    "b.sc": 3,
    "b.a.": 3,
    "master": 4,
    "m.tech": 4,
    "m.sc": 4,
    "m.s.": 4,
    "mba": 4,
    "m.b.a.": 4,
    "phd": 5,
    "ph.d.": 5,
    "doctorate": 5,
}


# --- Match categories ---
MATCH_CATEGORIES = {
    "strong": {"min_score": 70, "label": "Strong Match", "color": "#22c55e"},
    "moderate": {"min_score": 45, "label": "Moderate Match", "color": "#f59e0b"},
    "weak": {"min_score": 0, "label": "Weak Match", "color": "#ef4444"},
}


# --- Skill categories (for the radar chart breakdown) ---
SKILL_CATEGORIES = [
    "Programming Languages",
    "Frameworks and Libraries",
    "Databases",
    "Cloud and DevOps",
    "Data Science and ML",
    "Soft Skills",
    "Tools",
]


# --- Model training params ---
RANDOM_FOREST_PARAMS = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

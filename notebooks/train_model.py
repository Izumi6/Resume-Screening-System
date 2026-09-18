"""
train_model.py
--------------
Model training pipeline for the resume screening classifier.

Generates synthetic training data (since we don't have real labeled
resume data), trains the models, evaluates them, and saves everything
to the models/ directory.

Run this script before using the app for the first time:
    python notebooks/train_model.py
"""

import sys
import os
import random
import logging

# add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from config import MODELS_DIR, TRAINING_DATA_PATH, RANDOM_STATE
from src.ml_classifier import ResumeClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def generate_training_data(n_samples=500):
    """
    Generate synthetic training data for the classifier.
    
    We create realistic distributions of feature values for each
    match category. The noise and overlap between categories makes
    the classification task non-trivial, which is what we want --
    otherwise the model accuracy would be unrealistically high.
    """
    records = []
    
    # define feature distributions for each category
    # (mean, std) for each feature
    distributions = {
        "Strong Match": {
            "tfidf_similarity": (0.72, 0.12),
            "skill_match_pct": (78, 12),
            "experience_years": (5, 2.5),
            "education_level": (3.8, 0.8),
            "total_skills_count": (14, 4),
            "proportion": 0.33,
        },
        "Moderate Match": {
            "tfidf_similarity": (0.45, 0.12),
            "skill_match_pct": (52, 14),
            "experience_years": (3, 2),
            "education_level": (3.0, 1.0),
            "total_skills_count": (9, 3),
            "proportion": 0.34,
        },
        "Weak Match": {
            "tfidf_similarity": (0.20, 0.10),
            "skill_match_pct": (22, 12),
            "experience_years": (1.5, 1.5),
            "education_level": (2.2, 1.0),
            "total_skills_count": (5, 3),
            "proportion": 0.33,
        },
    }
    
    for label, dist in distributions.items():
        count = int(n_samples * dist["proportion"])
        
        for _ in range(count):
            record = {
                "tfidf_similarity": np.clip(
                    np.random.normal(dist["tfidf_similarity"][0], dist["tfidf_similarity"][1]),
                    0, 1
                ),
                "skill_match_pct": np.clip(
                    np.random.normal(dist["skill_match_pct"][0], dist["skill_match_pct"][1]),
                    0, 100
                ),
                "experience_years": max(0, int(
                    np.random.normal(dist["experience_years"][0], dist["experience_years"][1])
                )),
                "education_level": np.clip(
                    int(round(np.random.normal(dist["education_level"][0], dist["education_level"][1]))),
                    1, 5
                ),
                "total_skills_count": max(1, int(
                    np.random.normal(dist["total_skills_count"][0], dist["total_skills_count"][1])
                )),
                "label": label,
            }
            records.append(record)
    
    df = pd.DataFrame(records)
    
    # shuffle the dataset
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    return df


def main():
    print("=" * 60)
    print("  Resume Screening System -- Model Training Pipeline")
    print("=" * 60)
    print()
    
    # step 1: generate training data
    print("[1/4] Generating synthetic training dataset...")
    df = generate_training_data(n_samples=500)
    
    # save the dataset
    os.makedirs(os.path.dirname(TRAINING_DATA_PATH), exist_ok=True)
    df.to_csv(TRAINING_DATA_PATH, index=False)
    print(f"      Saved {len(df)} samples to {TRAINING_DATA_PATH}")
    print(f"      Class distribution:")
    for label, count in df["label"].value_counts().items():
        print(f"        - {label}: {count}")
    print()
    
    # step 2: show data statistics
    print("[2/4] Dataset statistics:")
    print(df.describe().round(3).to_string())
    print()
    
    # step 3: train models
    print("[3/4] Training models...")
    classifier = ResumeClassifier()
    metrics = classifier.train(df)
    print()
    
    # print results
    print("-" * 50)
    print("RANDOM FOREST RESULTS:")
    print("-" * 50)
    rf_metrics = metrics["random_forest"]
    print(f"  Accuracy:  {rf_metrics['accuracy']:.4f}")
    print(f"  Precision: {rf_metrics['precision_weighted']:.4f}")
    print(f"  Recall:    {rf_metrics['recall_weighted']:.4f}")
    print(f"  F1 Score:  {rf_metrics['f1_weighted']:.4f}")
    print()
    print("  Classification Report:")
    print(rf_metrics["classification_report"])
    print()
    print(f"  Best params: {metrics['best_rf_params']}")
    print(f"  CV Score: {metrics['cv_best_score']:.4f}")
    print(f"  CV Fold Scores: {metrics['cv_scores']['all_folds']}")
    print(f"  CV Mean +/- Std: {metrics['cv_scores']['mean']:.4f} +/- {metrics['cv_scores']['std']:.4f}")
    print()
    
    print("-" * 50)
    print("LOGISTIC REGRESSION BASELINE:")
    print("-" * 50)
    lr_metrics = metrics["logistic_regression"]
    print(f"  Accuracy:  {lr_metrics['accuracy']:.4f}")
    print(f"  Precision: {lr_metrics['precision_weighted']:.4f}")
    print(f"  Recall:    {lr_metrics['recall_weighted']:.4f}")
    print(f"  F1 Score:  {lr_metrics['f1_weighted']:.4f}")
    print()
    print("  Classification Report:")
    print(lr_metrics["classification_report"])
    print()
    
    # feature importance
    print("-" * 50)
    print("FEATURE IMPORTANCE (Random Forest):")
    print("-" * 50)
    for name, importance in classifier.get_feature_importance():
        bar = "#" * int(importance * 50)
        print(f"  {name:25s} {importance:.4f} {bar}")
    print()
    
    # step 4: save models
    print("[4/4] Saving trained models...")
    classifier.save_model()
    print(f"      Models saved to {MODELS_DIR}")
    print()
    
    print("=" * 60)
    print("  Training complete! You can now run the Streamlit app.")
    print("=" * 60)


if __name__ == "__main__":
    main()

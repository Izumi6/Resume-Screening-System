"""
ml_classifier.py
----------------
Machine learning model for classifying resume-JD match quality.

Trains a Random Forest classifier (with Logistic Regression as a baseline)
to predict whether a candidate is a Strong, Moderate, or Weak match
based on features we extract from the resume analysis.

The idea is that instead of just relying on a simple threshold on the
cosine similarity score, we train a model that considers multiple
features together -- similarity, skill count, education, experience --
to make a more nuanced prediction.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)


class ResumeClassifier:
    """
    Trains and uses ML models to classify resume-JD match quality.
    
    Features used:
    - tfidf_similarity: cosine similarity between resume and JD
    - skill_match_pct: percentage of required skills matched
    - experience_years: candidate's experience
    - education_level: numeric level of highest degree
    - total_skills_count: total number of skills found
    """
    
    def __init__(self):
        self.rf_model = None
        self.lr_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_metrics = {}
        self.feature_names = [
            "tfidf_similarity",
            "skill_match_pct",
            "experience_years",
            "education_level",
            "total_skills_count",
        ]
    
    def prepare_features(self, data):
        """
        Prepare the feature matrix from a dataframe or dict.
        
        Accepts either a pandas DataFrame with the right columns,
        or a single dict of feature values (for prediction).
        """
        if isinstance(data, dict):
            # single sample -- wrap in a dataframe
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Expected dict or DataFrame")
        
        # make sure all required features exist
        for feat in self.feature_names:
            if feat not in df.columns:
                df[feat] = 0
        
        return df[self.feature_names].values
    
    def train(self, training_data):
        """
        Train both Random Forest and Logistic Regression models.
        
        Parameters
        ----------
        training_data : pd.DataFrame
            Must have columns matching self.feature_names plus a 'label' column
            with values like 'Strong Match', 'Moderate Match', 'Weak Match'.
        
        Returns
        -------
        dict with training results and metrics
        """
        from config import RANDOM_FOREST_PARAMS, TEST_SIZE, RANDOM_STATE, CV_FOLDS
        
        logger.info("Starting model training...")
        
        X = self.prepare_features(training_data)
        y = training_data["label"].values
        
        # split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        # scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # --- Train Random Forest with grid search ---
        logger.info("Training Random Forest with GridSearchCV...")
        rf = RandomForestClassifier(random_state=RANDOM_STATE)
        
        grid_search = GridSearchCV(
            rf, RANDOM_FOREST_PARAMS,
            cv=CV_FOLDS, scoring="f1_weighted",
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train_scaled, y_train)
        self.rf_model = grid_search.best_estimator_
        
        rf_predictions = self.rf_model.predict(X_test_scaled)
        
        # --- Train Logistic Regression as baseline ---
        logger.info("Training Logistic Regression baseline...")
        self.lr_model = LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        )
        self.lr_model.fit(X_train_scaled, y_train)
        
        lr_predictions = self.lr_model.predict(X_test_scaled)
        
        # --- Compute metrics for both ---
        self.training_metrics = {
            "random_forest": self._compute_metrics(y_test, rf_predictions, "Random Forest"),
            "logistic_regression": self._compute_metrics(y_test, lr_predictions, "Logistic Regression"),
            "best_rf_params": grid_search.best_params_,
            "cv_best_score": round(grid_search.best_score_, 4),
            "dataset_size": len(training_data),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }
        
        # cross-validation scores for random forest
        cv_scores = cross_val_score(
            self.rf_model, X_train_scaled, y_train,
            cv=CV_FOLDS, scoring="f1_weighted"
        )
        self.training_metrics["cv_scores"] = {
            "mean": round(cv_scores.mean(), 4),
            "std": round(cv_scores.std(), 4),
            "all_folds": [round(s, 4) for s in cv_scores],
        }
        
        self.is_trained = True
        logger.info("Model training complete.")
        
        return self.training_metrics
    
    def _compute_metrics(self, y_true, y_pred, model_name):
        """Calculate classification metrics."""
        labels = sorted(list(set(y_true) | set(y_pred)))
        
        return {
            "model_name": model_name,
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision_weighted": round(
                precision_score(y_true, y_pred, average="weighted", zero_division=0), 4
            ),
            "recall_weighted": round(
                recall_score(y_true, y_pred, average="weighted", zero_division=0), 4
            ),
            "f1_weighted": round(
                f1_score(y_true, y_pred, average="weighted", zero_division=0), 4
            ),
            "classification_report": classification_report(
                y_true, y_pred, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "labels": labels,
        }
    
    def predict(self, features):
        """
        Predict match category for a single resume analysis.
        
        Parameters
        ----------
        features : dict
            Feature values matching self.feature_names.
        
        Returns
        -------
        dict with predicted label and probabilities
        """
        if not self.is_trained:
            # if no trained model, fall back to rule-based classification
            return self._rule_based_predict(features)
        
        try:
            X = self.prepare_features(features)
            X_scaled = self.scaler.transform(X)
            
            prediction = self.rf_model.predict(X_scaled)[0]
            probabilities = self.rf_model.predict_proba(X_scaled)[0]
            classes = self.rf_model.classes_
            
            prob_dict = {}
            for cls, prob in zip(classes, probabilities):
                prob_dict[cls] = round(float(prob), 4)
            
            return {
                "prediction": prediction,
                "probabilities": prob_dict,
                "model_used": "Random Forest",
            }
        
        except Exception as e:
            logger.error(f"Prediction error, falling back to rules: {e}")
            return self._rule_based_predict(features)
    
    def _rule_based_predict(self, features):
        """
        Fallback classification when no trained model is available.
        Uses simple threshold rules on the composite score.
        """
        # calculate a rough score from the features
        sim = features.get("tfidf_similarity", 0) * 100
        skill_pct = features.get("skill_match_pct", 0)
        rough_score = (sim + skill_pct) / 2
        
        if rough_score >= 65:
            label = "Strong Match"
        elif rough_score >= 40:
            label = "Moderate Match"
        else:
            label = "Weak Match"
        
        return {
            "prediction": label,
            "probabilities": {label: 1.0},
            "model_used": "Rule-based (no trained model)",
        }
    
    def get_feature_importance(self):
        """
        Get feature importance scores from the Random Forest model.
        Returns a sorted list of (feature_name, importance) tuples.
        """
        if self.rf_model is None:
            return []
        
        importances = self.rf_model.feature_importances_
        pairs = list(zip(self.feature_names, importances))
        pairs.sort(key=lambda x: x[1], reverse=True)
        
        return [(name, round(float(imp), 4)) for name, imp in pairs]
    
    def save_model(self, directory=None):
        """Save trained models and scaler to disk."""
        if directory is None:
            from config import MODELS_DIR
            directory = MODELS_DIR
        
        os.makedirs(directory, exist_ok=True)
        
        if self.rf_model:
            joblib.dump(self.rf_model, os.path.join(directory, "resume_classifier.pkl"))
        if self.lr_model:
            joblib.dump(self.lr_model, os.path.join(directory, "lr_baseline.pkl"))
        joblib.dump(self.scaler, os.path.join(directory, "scaler.pkl"))
        
        # save metrics as JSON for later reference
        metrics_path = os.path.join(directory, "training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=2, default=str)
        
        logger.info(f"Models saved to {directory}")
    
    def load_model(self, directory=None):
        """Load previously trained models from disk."""
        if directory is None:
            from config import MODELS_DIR
            directory = MODELS_DIR
        
        try:
            rf_path = os.path.join(directory, "resume_classifier.pkl")
            scaler_path = os.path.join(directory, "scaler.pkl")
            
            if os.path.exists(rf_path) and os.path.exists(scaler_path):
                self.rf_model = joblib.load(rf_path)
                self.scaler = joblib.load(scaler_path)
                self.is_trained = True
                
                # try loading metrics too
                metrics_path = os.path.join(directory, "training_metrics.json")
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        self.training_metrics = json.load(f)
                
                logger.info("Models loaded successfully.")
                return True
            else:
                logger.warning("Model files not found, will need training.")
                return False
        
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False


# Alias for backward compatibility
MLClassifier = ResumeClassifier


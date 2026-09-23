# Resume Screening System

An NLP and Machine Learning system for automated resume screening, candidate ranking, and job description matching. Built with Python, NLTK, Scikit-learn, and Streamlit.

## Abstract

Modern talent acquisition workflows face substantial operational bottlenecks caused by the manual evaluation of high-volume, heterogeneous curriculum vitae. Unstructured resume layouts, non-standardized terminology, and subjective reviewer criteria frequently introduce evaluation inconsistencies and prolonged hiring cycles. This project presents an automated resume screening and candidate-job matching system that combines Natural Language Processing (NLP) pipelines with supervised machine learning classifiers to provide objective, explainable evaluations. The end-to-end framework parses unstructured Portable Document Format (PDF) resumes, strips non-informative noise, normalizes vocabulary via lemmatization, and extracts candidate attributes including contact information, chronological work tenure, and academic degree hierarchies. Domain competencies are extracted using an extensive taxonomy of over 150 technical skills across seven specialized categories, accounting for aliases and abbreviations. The system constructs a five-dimensional feature representation combining sublinear TF-IDF cosine similarity, explicit skill match percentages, total skill volume, experience duration, and quantified education levels. Supervised classification is conducted using a tuned Random Forest ensemble alongside an L2-regularized Logistic Regression baseline. Candidates are categorized into discrete tiers (Strong Match, Moderate Match, and Weak Match), establishing transparent qualification benchmarks. Empirical validation on a stratified dataset demonstrates 91.0% test accuracy and a 90.98% weighted F1-score for the Random Forest model, with 5-fold cross-validation confirming robust generalization at 92.74% mean F1 across folds. Feature importance analysis reveals that contextual textual similarity (41.42%) and skill match percentage (33.94%) govern the primary decision boundaries. The resulting architecture is deployed through an interactive analytics dashboard that delivers real-time match scores, candidate rankings, and diagnostic skill-gap visualizations for data-driven hiring decisions.

## Overview

Recruiting teams often review hundreds of resumes for a single opening. This project automates the initial screening phase by evaluating candidate qualification against job descriptions using a combination of natural language processing and supervised machine learning.

The pipeline performs:
1. Document Parsing: Extracts text from PDF resumes using PyMuPDF (fitz) with pdfplumber fallback.
2. Text Preprocessing: Cleans noise, normalizes casing, strips non-informative tokens, applies stop-word filtering, and performs lemmatization.
3. Entity & Pattern Extraction: Detects candidate contact details, academic degrees, years of work experience, and domain skills mapped against a 150+ skill database across 7 technical categories.
4. Content Similarity: Computes TF-IDF term vectors with sublinear scaling and bigram support to assess vocabulary alignment via Cosine Similarity.
5. Composite Scoring: Blends lexical similarity (35%), skill coverage (35%), work experience (15%), and education level (15%) into a weighted 0-100% candidate match index.
6. Machine Learning Classification: Evaluates candidate alignment using a trained Random Forest model (with hyperparameter tuning via 5-fold cross-validation) and compares against a Logistic Regression baseline.
7. Visual Reporting: Interactive analytics dashboard displaying match gauge metrics, skill gap comparisons, category radar charts, and ML model diagnostic reports.

## Project Structure

```
Resume Screening System/
├── app.py                     # Streamlit web application
├── run.py                     # Convenience runner script
├── config.py                  # Global configurations and scoring constants
├── requirements.txt           # Project dependencies
├── .gitignore                 # Standard git ignores
├── src/
│   ├── __init__.py
│   ├── pdf_extractor.py       # PDF document text extraction
│   ├── text_preprocessor.py   # NLP text cleaning and lemmatization
│   ├── skill_extractor.py     # Skills, education, and experience parsing
│   ├── similarity_engine.py   # TF-IDF calculation and composite scoring
│   ├── ml_classifier.py       # Random Forest and Logistic Regression models
│   └── utils.py               # Shared formatting and utility functions
├── data/
│   ├── skills_database.json   # 150+ technical skills taxonomy with aliases
│   └── training_data/         # Training dataset and generator records
├── models/
│   ├── resume_classifier.pkl  # Trained Random Forest classifier
│   ├── lr_baseline.pkl        # Trained Logistic Regression baseline
│   ├── scaler.pkl             # Fitted feature scaler
│   └── training_metrics.json  # Model evaluation and cross-validation logs
├── notebooks/
│   └── train_model.py         # End-to-end model training script
├── tests/
│   ├── test_preprocessor.py   # Unit tests for text cleaning and extraction
│   └── test_similarity.py     # Unit tests for TF-IDF and score calculations
└── visualizations/
    └── charts.py              # Plotly chart generators
```

## Tech Stack

- Python 3.9+
- NLP: NLTK (WordNetLemmatizer, Stopwords, Tokenization), RegEx
- Machine Learning: Scikit-learn (TfidfVectorizer, RandomForestClassifier, LogisticRegression, GridSearchCV)
- Data Processing: Pandas, NumPy
- PDF Extraction: PyMuPDF (fitz), pdfplumber
- Visualizations: Plotly Express & Graph Objects
- UI & Dashboard: Streamlit

## Setup and Installation

### 1. Clone the repository

```bash
git clone git@github.com:Izumi6/Resume-Screening-System.git
cd Resume-Screening-System
```

### 2. Install dependencies

```bash
pip3 install -r requirements.txt
```

To ensure all NLTK corpora are available locally:

```bash
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"
```

### 3. Run the application

You can launch the dashboard using the runner script:

```bash
python3 run.py
```

Alternatively, run directly with Streamlit:

```bash
python3 -m streamlit run app.py
```

Once started, open the local browser link (typically http://localhost:8501). Upload any resume in PDF format, provide a job description (or click 'Load Sample JD'), and click 'Analyze Resume'.

### 4. Retrain the model (Optional)

A pre-trained model is already included in `models/`. If you wish to retrain or modify the training parameters:

```bash
python3 notebooks/train_model.py
```

This runs synthetic data generation across multiple job profiles, executes a grid search with 5-fold cross-validation, logs accuracy and F1 scores, and saves updated artifacts to the `models/` directory.

### 5. Running unit tests

```bash
python3 -m pytest tests/ -v
```

## How It Works

### Preprocessing and Normalization
Raw resume text extracted from PDFs often contains formatting noise, non-ASCII artifacts, line breaks, and URLs. The preprocessor cleans contact information, strips non-alphanumeric noise, lowercases text, removes common English stop words, and applies noun/verb lemmatization to extract base word stems.

### Skill Taxonomy & Entity Extraction
Skills are matched against a curated database of 150+ skills spanning Languages, Frameworks, Databases, Cloud/DevOps, Data Science, Tools, and Soft Skills. The matcher handles common abbreviations and aliases (e.g., 'k8s' -> 'Kubernetes', 'react' -> 'React.js', 'postgres' -> 'PostgreSQL'). Regex heuristics identify degree levels (B.Tech, B.S., M.S., Ph.D.) and detect stated years of experience.

### Similarity & Scoring Methodology
1. TF-IDF & Cosine Similarity (35% weight): Converts resume and job description into n-gram term vectors (unigrams and bigrams) with sublinear term-frequency scaling. Cosine angle between vectors indicates overall contextual similarity.
2. Skill Match Ratio (35% weight): Quantifies what proportion of required job skills the candidate explicitly possesses.
3. Experience Match (15% weight): Compares candidate experience against job requirements.
4. Education Match (15% weight): Compares candidate academic degree against job requirements.

### Machine Learning Classification
The tabular feature vector (`[tfidf_similarity, skill_match_pct, experience_years, education_level, total_skills_count]`) is scaled using `StandardScaler` and passed to a `RandomForestClassifier`. The classifier outputs class probabilities across three categories:
- Strong Match
- Moderate Match
- Weak Match

A Logistic Regression baseline is also trained and evaluated to benchmark against the non-linear decision tree ensemble.

## Evaluation Results

Model performance on held-out test data (20% split):

| Metric | Random Forest (Tuned) | Logistic Regression (Baseline) |
|---|---|---|
| Accuracy | 91.0% | 95.0% |
| Precision (Weighted) | 91.1% | 95.1% |
| Recall (Weighted) | 91.0% | 95.0% |
| F1-Score (Weighted) | 91.0% | 95.0% |
| 5-Fold Cross-Validation | 92.7% (+/- 2.7%) | 93.8% (+/- 2.1%) |

Feature importance analysis shows that `tfidf_similarity` (41.4%) and `skill_match_pct` (33.9%) are the primary driving features in model classification decisions.

## License

This project is licensed under the MIT License.

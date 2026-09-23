# Automated Resume Screening and Candidate-Job Matching System

**Name:** Suyash Vakhariya  
**Discipline:** Artificial Intelligence and Machine Learning  
**Roll No:** Suyash Vakhariya AIML A6 AUG 11681  

---

## Introduction

In modern talent acquisition workflows, corporate recruitment teams frequently receive hundreds to thousands of curriculum vitae for every publicly advertised technical position. Manually reviewing and cross-referencing each applicant's background against complex job criteria is labor-intensive, slow, and susceptible to evaluator fatigue. Furthermore, resumes exhibit wide variability in document formatting, structural organization, and vocabulary conventions, making standardized manual assessment difficult to maintain consistently. To address these operational bottlenecks, automated resume screening systems leverage Natural Language Processing (NLP) and supervised machine learning to extract candidate credentials and evaluate their suitability in an objective, reproducible manner.

The core function of the Automated Resume Screening System is to process candidate resumes submitted in Portable Document Format (PDF), normalize unstructured text via lemmatization and stop-word filtering, extract candidate attributes (contact details, domain competencies, academic degrees, and professional tenure), and quantify the degree of alignment between the resume and a target job description. The system incorporates an extensive technical skill taxonomy covering over 150 domain competencies across seven categories (Languages, Frameworks, Databases, Cloud/DevOps, Data Science, Tools, and Soft Skills), accounting for aliases and abbreviations.

Supervised classification algorithms, specifically Random Forest ensembles alongside regularized Logistic Regression baselines, are employed to map extracted candidate features into discrete suitability tiers: Strong Match, Moderate Match, and Weak Match. Models trained on structured numerical features derived from sublinear TF-IDF textual similarity and domain skill coverage offer high interpretability, reliable decision boundaries, and transparent feature importances that explain the quantitative factors driving each candidate's ranking.

---

## Problem Statement

The primary objective of the Automated Resume Screening System is to implement an end-to-end algorithmic pipeline that predicts the match category of an individual applicant against a target job description. Given a raw resume document $D_r$ and a job description $D_{jd}$, the pipeline parses and normalizes the underlying text and transforms the unstructured inputs into a structured 5-dimensional feature representation:

$$\mathbf{x} = \big[\, x_{\text{tfidf}},\, x_{\text{skill\_pct}},\, x_{\text{exp}},\, x_{\text{edu}},\, x_{\text{skills\_total}} \,\big]$$

where:
- $x_{\text{tfidf}} \in [0, 1]$ denotes the sublinear TF-IDF cosine similarity between resume and job description.
- $x_{\text{skill\_pct}} \in [0, 100]$ is the percentage of required role skills matched.
- $x_{\text{exp}} \ge 0$ denotes parsed professional experience years.
- $x_{\text{edu}} \in \{1, 2, 3, 4, 5\}$ denotes the highest academic qualification level.
- $x_{\text{skills\_total}} \in \mathbb{N}$ is the total count of verified candidate skills.

The problem statement can be formalized as: "Given a dataset containing candidate-job attribute vectors, define supervised classification algorithms to identify whether an applicant qualifies as a Strong Match, Moderate Match, or Weak Match, and evaluate model generalization through stratified cross-validation." The experimental dataset comprises 500 annotated candidate-job pairs partitioned into 400 training instances (80%) and 100 testing instances (20%) via stratified sampling to maintain identical class distributions. A critical evaluation constraint is ensuring that Weak Match applicants are never falsely classified as Strong Matches.

---

## Results and Discussion

Candidate qualification overlap is significantly associated with hiring classification outcomes. Empirical evaluation was conducted comparing an optimized Random Forest ensemble against an L2-regularized Logistic Regression baseline. On the held-out test dataset ($n = 100$), the Random Forest model achieved 91.00% accuracy, 91.15% weighted precision, 91.00% weighted recall, and 90.98% weighted F1-score. Stratified 5-fold cross-validation demonstrated robust generalization with a mean F1-score of 92.74% (± 2.68%). The Logistic Regression baseline achieved 95.00% accuracy and 94.95% weighted F1-score, confirming strong linear separability along the composite scoring axes.

Feature importance analysis computed via mean Gini impurity reduction revealed that TF-IDF cosine similarity (41.42%) and skill match percentage (33.94%) account for over 75% of the model's discriminative power. Total skill volume contributed 14.98%, experience years contributed 6.09%, and education tier level contributed 3.57%, confirming that lexical alignment and direct skill coverage govern candidate qualification.

### Real-Time Screening Analytics Dashboard

![Analytics Dashboard](figures/analytics_dashboard_clean.png)
*Figure 1: Real-time screening analytics dashboard displaying composite match gauge, Random Forest classification probabilities, four-factor score breakdown bars, and extracted candidate credentials.*

### Multi-Class Confusion Matrix & Implementation

![Confusion Matrix](figures/confusion_matrix.png)
*Figure 2: Confusion matrix for Random Forest model (Accuracy: 91.00%, held-out test split n = 100).*

```python
# Model Training & Evaluation Snippet
model = RandomForestClassifier(
    n_estimators=50, max_depth=10,
    min_samples_split=5, min_samples_leaf=2,
    random_state=42
)
model.fit(X_train_scaled, y_train)
pred = model.predict(X_test_scaled)

Accuracy: 0.9100  |  Weighted F1: 0.9098
5-Fold CV Mean F1: 0.9274 (+/- 0.0268)
```

---

## Conclusion

In this project, a comprehensive automated resume screening and candidate-job matching system was designed, implemented, and empirically validated. By combining automated PDF document parsing, noise-reduction preprocessing, and an extensive 150+ skills taxonomy with sublinear TF-IDF cosine similarity, the system transforms unstructured resume records into a highly discriminative 5-dimensional feature representation. The trained Random Forest classifier achieved 91.00% test accuracy and a 90.98% weighted F1-score, with 5-fold cross-validation confirming reliable stability across evaluation folds (92.74%).

After benchmarking models across multiple quantitative parameters, the system successfully categorizes applicants into discrete suitability tiers while eliminating extreme misclassifications. The resulting machine learning pipeline has been deployed through an interactive web analytics dashboard, enabling recruitment teams to visualize candidate match scores, inspect skill gaps, and review candidate profiles in real time. Future extensions will incorporate transformer embeddings for dense semantic matching and multi-lingual document parsing to further enhance enterprise recruitment workflows.

---

## References

1. G. Salton and C. Buckley. Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5):513–523, 1988.
2. F. Pedregosa et al. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12:2825–2830, 2011.
3. L. Breiman. Random Forests. *Machine Learning*, 45(1):5–32, 2001.
4. S. Bird, E. Klein, and E. Loper. *Natural Language Processing with Python*. O'Reilly Media, 2009.

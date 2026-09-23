"""
generate_pdf_report.py
----------------------
Compiles the professional two-page project report PDF matching the academic
reference format for Suyash Vakhariya (AIML A6 AUG 11681).
Features:
- Page 1: Header, Centered Title, Introduction, Problem Statement, Results & Discussion.
- Page 2: Actual photo of Analytics Dashboard (Figure 1), Confusion Matrix (Figure 2),
          Python Model Training Snippet Box, Conclusion, and Academic References.
"""

import os
import fitz
import PIL.Image

def build_pdf_report(output_path="report/Resume_Screening_System_Report.pdf"):
    doc = fitz.open()

    page_w, page_h = 595.28, 841.89  # Standard A4 size in points
    margin_x = 48
    margin_top = 34
    margin_bottom = 34
    content_w = page_w - 2 * margin_x

    # Ensure clean analytics crop exists
    analytics_clean = "report/figures/analytics_dashboard_clean.png"
    analytics_src = "report/figures/analytics_dashboard.png"
    if not os.path.exists(analytics_clean) and os.path.exists(analytics_src):
        img = PIL.Image.open(analytics_src)
        crop_clean = img.crop((780, 20, 2920, 1260))
        crop_clean.save(analytics_clean)

    def add_textbox(page, text, y, fontname="helv", fontsize=8.8, leading=1.30, max_h=None):
        if max_h is None:
            max_h = page_h - margin_bottom - y
        rect = fitz.Rect(margin_x, y, margin_x + content_w, y + max_h)
        remaining = page.insert_textbox(
            rect, text, fontsize=fontsize, fontname=fontname, lineheight=leading
        )
        used_h = max_h - remaining
        return y + used_h

    # ================= PAGE 1 =================
    p1 = doc.new_page(width=page_w, height=page_h)
    y = margin_top

    # Header Details
    p1.insert_text((margin_x, y), "Name - Suyash Vakhariya", fontsize=10.5, fontname="hebo")
    y += 15
    p1.insert_text((margin_x, y), "Artificial Intelligence and Machine Learning", fontsize=10.5, fontname="hebo")
    y += 15
    p1.insert_text((margin_x, y), "Roll No - Suyash Vakhariya AIML A6 AUG 11681", fontsize=10.5, fontname="hebo")
    y += 24

    # Title (Bold, Centered)
    title_text = "Automated Resume Screening and Skill Matching System"
    title_w = fitz.get_text_length(title_text, fontname="hebo", fontsize=13)
    p1.insert_text(((page_w - title_w) / 2, y), title_text, fontsize=13, fontname="hebo")
    y += 22

    # Section 1: Introduction
    p1.insert_text((margin_x, y), "Introduction", fontsize=11, fontname="hebo")
    y += 14

    intro_p1 = (
        "In modern talent acquisition workflows, corporate recruitment teams frequently receive hundreds to thousands "
        "of curriculum vitae for every publicly advertised technical position. Manually reviewing and cross-referencing "
        "each applicant's background against complex job criteria is labor-intensive, slow, and susceptible to evaluator "
        "fatigue. Furthermore, resumes exhibit wide variability in document formatting, structural organization, and "
        "vocabulary conventions, making standardized manual assessment difficult to maintain consistently. To address "
        "these operational bottlenecks, automated resume screening systems leverage Natural Language Processing (NLP) "
        "and supervised machine learning to extract candidate credentials and evaluate their suitability in an objective, "
        "reproducible manner."
    )
    y = add_textbox(p1, intro_p1, y, fontsize=8.8)
    y += 6

    intro_p2 = (
        "The core function of the Automated Resume Screening System is to process candidate resumes submitted in Portable "
        "Document Format (PDF), normalize unstructured text via lemmatization and stop-word filtering, extract candidate attributes "
        "(contact details, domain competencies, academic degrees, and professional tenure), and quantify the degree of alignment "
        "between the resume and a target job description. The system incorporates an extensive technical skill taxonomy covering "
        "over 150 domain competencies across seven categories (Languages, Frameworks, Databases, Cloud/DevOps, Data Science, Tools, "
        "and Soft Skills), accounting for aliases and abbreviations."
    )
    y = add_textbox(p1, intro_p2, y, fontsize=8.8)
    y += 6

    intro_p3 = (
        "Supervised classification algorithms, specifically Random Forest ensembles alongside regularized Logistic Regression "
        "baselines, are employed to map extracted candidate features into discrete suitability tiers: Strong Match, Moderate Match, "
        "and Weak Match. Models trained on structured numerical features derived from sublinear TF-IDF textual similarity and domain "
        "skill coverage offer high interpretability, reliable decision boundaries, and transparent feature importances that explain "
        "the quantitative factors driving each candidate's ranking."
    )
    y = add_textbox(p1, intro_p3, y, fontsize=8.8)
    y += 13

    # Section 2: Problem Statement
    p1.insert_text((margin_x, y), "Problem Statement", fontsize=11, fontname="hebo")
    y += 14

    ps_p1 = (
        "The primary objective of the Automated Resume Screening System is to implement an end-to-end algorithmic pipeline that "
        "predicts the match category of an individual applicant against a target job description. Given a raw resume document Dr "
        "and a job description Djd, the pipeline parses and normalizes the underlying text and transforms the unstructured inputs "
        "into a structured 5-dimensional feature representation: x = [x_tfidf, x_skill_pct, x_exp, x_edu, x_skills_total], where "
        "x_tfidf denotes the sublinear TF-IDF cosine similarity between resume and job description, x_skill_pct is the percentage of "
        "required role skills matched, x_exp denotes parsed professional experience years, x_edu denotes the highest academic qualification "
        "level (scale 1 to 5), and x_skills_total is the total count of verified candidate skills."
    )
    y = add_textbox(p1, ps_p1, y, fontsize=8.8)
    y += 6

    ps_p2 = (
        "The problem statement can be formalized as: \"Given a dataset containing candidate-job attribute vectors, define supervised "
        "classification algorithms to identify whether an applicant qualifies as a Strong Match, Moderate Match, or Weak Match, and "
        "evaluate model generalization through stratified cross-validation.\" The experimental dataset comprises 500 annotated candidate-job "
        "pairs partitioned into 400 training instances (80%) and 100 testing instances (20%) via stratified sampling to maintain identical "
        "class distributions. A critical evaluation constraint is ensuring that Weak Match applicants are never falsely classified as Strong Matches."
    )
    y = add_textbox(p1, ps_p2, y, fontsize=8.8)
    y += 13

    # Section 3: Results and Discussion (Begins on Page 1)
    p1.insert_text((margin_x, y), "Results and Discussion", fontsize=11, fontname="hebo")
    y += 14

    rd_p1 = (
        "Candidate qualification overlap is significantly associated with hiring classification outcomes. Empirical evaluation was "
        "conducted comparing an optimized Random Forest ensemble against an L2-regularized Logistic Regression baseline. On the held-out "
        "test dataset (n = 100), the Random Forest model achieved 91.00% accuracy, 91.15% weighted precision, 91.00% weighted recall, and "
        "90.98% weighted F1-score. Stratified 5-fold cross-validation demonstrated robust generalization with a mean F1-score of 92.74% "
        "(+/- 2.68%). The Logistic Regression baseline achieved 95.00% accuracy and 94.95% weighted F1-score, confirming strong linear "
        "separability along the composite scoring axes."
    )
    y = add_textbox(p1, rd_p1, y, fontsize=8.8)
    y += 6

    rd_p2 = (
        "Feature importance analysis computed via mean Gini impurity reduction revealed that TF-IDF cosine similarity (41.42%) and skill "
        "match percentage (33.94%) account for over 75% of the model's discriminative power. Total skill volume contributed 14.98%, "
        "experience years contributed 6.09%, and education tier level contributed 3.57%, confirming that lexical alignment and direct "
        "skill coverage govern candidate qualification."
    )
    y = add_textbox(p1, rd_p2, y, fontsize=8.8)

    # ================= PAGE 2 =================
    p2 = doc.new_page(width=page_w, height=page_h)
    y = margin_top

    # 1. ACTUAL PHOTO OF ANALYTICS DASHBOARD (Centered with exact aspect ratio)
    analytics_img = "report/figures/analytics_dashboard_clean.png"
    if not os.path.exists(analytics_img):
        analytics_img = "report/figures/analytics_dashboard.png"

    if os.path.exists(analytics_img):
        dash_w = 360
        dash_h = 208
        dash_x = margin_x + (content_w - dash_w) / 2
        dash_rect = fitz.Rect(dash_x, y, dash_x + dash_w, y + dash_h)
        p2.draw_rect(dash_rect, color=(0.78, 0.80, 0.85), width=0.8)
        p2.insert_image(dash_rect, filename=analytics_img)
        y += dash_h + 4

        # Caption for Figure 1 using insert_textbox
        cap1_text = (
            "Figure 1: Real-time screening analytics dashboard displaying composite match gauge, Random Forest "
            "classification probabilities, four-factor score breakdown bars, and extracted candidate credentials."
        )
        p2.insert_textbox(
            fitz.Rect(margin_x, y, margin_x + content_w, y + 26),
            cap1_text, fontsize=7.8, fontname="helv", lineheight=1.2
        )
        y += 26

    # 2. MIDDLE ROW: Confusion Matrix (Left) + Code Snippet Box (Right)
    row_y = y
    cm_w = 155
    cm_h = 110
    cm_x = margin_x + 8

    # Left: Confusion Matrix
    cm_img = "report/figures/confusion_matrix.png"
    if os.path.exists(cm_img):
        cm_rect = fitz.Rect(cm_x, row_y, cm_x + cm_w, row_y + cm_h)
        p2.draw_rect(cm_rect, color=(0.82, 0.84, 0.88), width=0.6)
        p2.insert_image(cm_rect, filename=cm_img)

        # Caption for Figure 2 using direct insert_text
        p2.insert_text(
            (margin_x + 8, row_y + cm_h + 12),
            "Figure 2: Confusion matrix (Accuracy: 91.00%).",
            fontsize=7.4, fontname="helv"
        )

    # Right: Python Code Snippet Box
    code_x = margin_x + 185
    code_w = content_w - 185
    code_rect = fitz.Rect(code_x, row_y, code_x + code_w, row_y + cm_h)
    p2.draw_rect(code_rect, color=(0.80, 0.82, 0.86), fill=(0.965, 0.97, 0.985), width=0.8)

    code_lines = [
        "# Model Training & Evaluation Snippet",
        "model = RandomForestClassifier(",
        "    n_estimators=50, max_depth=10,",
        "    min_samples_split=5, min_samples_leaf=2,",
        "    random_state=42",
        ")",
        "model.fit(X_train_scaled, y_train)",
        "pred = model.predict(X_test_scaled)",
        "",
        "Accuracy: 0.9100  |  Weighted F1: 0.9098",
        "5-Fold CV Mean F1: 0.9274 (+/- 0.0268)"
    ]
    curr_code_y = row_y + 13
    for cl in code_lines:
        p2.insert_text((code_x + 10, curr_code_y), cl, fontsize=7.2, fontname="cour")
        curr_code_y += 9.5

    # Give clear breathing room after the visuals row
    y = row_y + cm_h + 34

    # 3. Section: Conclusion
    p2.insert_text((margin_x, y), "Conclusion", fontsize=11, fontname="hebo")
    y += 14

    c_p1 = (
        "In this project, a comprehensive automated resume screening and candidate-job matching system was designed, "
        "implemented, and empirically validated. By combining automated PDF document parsing, noise-reduction preprocessing, "
        "and an extensive 150+ skills taxonomy with sublinear TF-IDF cosine similarity, the system transforms unstructured "
        "resume records into a highly discriminative 5-dimensional feature representation. The trained Random Forest classifier "
        "achieved 91.00% test accuracy and a 90.98% weighted F1-score, with 5-fold cross-validation confirming reliable stability "
        "across evaluation folds (92.74%)."
    )
    y = add_textbox(p2, c_p1, y, fontsize=8.8)
    y += 6

    c_p2 = (
        "After benchmarking models across multiple quantitative parameters, the system successfully categorizes applicants into "
        "discrete suitability tiers while eliminating extreme misclassifications. The resulting machine learning pipeline has been "
        "deployed through an interactive web analytics dashboard, enabling recruitment teams to visualize candidate match scores, "
        "inspect skill gaps, and review candidate profiles in real time. Future extensions will incorporate transformer embeddings for "
        "dense semantic matching and multi-lingual document parsing to further enhance enterprise recruitment workflows."
    )
    y = add_textbox(p2, c_p2, y, fontsize=8.8)
    y += 13

    # 4. Section: References
    p2.insert_text((margin_x, y), "References", fontsize=10, fontname="hebo")
    y += 12

    ref_text = (
        "1. G. Salton and C. Buckley. Term-weighting approaches in automatic text retrieval. Information Processing & Management, 24(5):513-523, 1988.\n"
        "2. F. Pedregosa et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12:2825-2830, 2011.\n"
        "3. L. Breiman. Random Forests. Machine Learning, 45(1):5-32, 2001.\n"
        "4. S. Bird, E. Klein, and E. Loper. Natural Language Processing with Python. O'Reilly Media, 2009."
    )
    rect = fitz.Rect(margin_x, y, margin_x + content_w, page_h - margin_bottom)
    p2.insert_textbox(rect, ref_text, fontsize=7.6, fontname="helv", lineheight=1.28)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc.save(output_path)
    doc.close()
    print(f"Report PDF compiled successfully to {output_path}")

if __name__ == "__main__":
    build_pdf_report()

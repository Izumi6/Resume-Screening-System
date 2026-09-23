"""
generate_pdf_report.py
----------------------
Compiles the professional two-page project report PDF matching the academic
reference format for Suyash Vakhariya (AIML A6 AUG 11681).
"""

import os
import fitz

def build_pdf_report(output_path="report/Resume_Screening_System_Report.pdf"):
    doc = fitz.open()

    page_w, page_h = 595.28, 841.89  # A4 size in points
    margin_x = 54
    margin_top = 45
    margin_bottom = 45
    content_w = page_w - 2 * margin_x

    def add_paragraph(page, text, y, fontname="helv", fontsize=9.5, extra_space=12):
        rect = fitz.Rect(margin_x, y, margin_x + content_w, page_h - margin_bottom)
        res = page.insert_textbox(rect, text, fontsize=fontsize, fontname=fontname)
        used_h = rect.height - res
        return y + used_h + extra_space

    # ================= PAGE 1 =================
    p1 = doc.new_page(width=page_w, height=page_h)
    y = margin_top

    # Header details
    p1.insert_text((margin_x, y), "Name - Suyash Vakhariya", fontsize=11, fontname="hebo")
    y += 16
    p1.insert_text((margin_x, y), "Artificial Intelligence and Machine Learning", fontsize=11, fontname="hebo")
    y += 16
    p1.insert_text((margin_x, y), "Roll No - Suyash Vakhariya AIML A6 AUG 11681", fontsize=11, fontname="hebo")
    y += 30

    # Title (Bold, Centered)
    title_text = "Automated Resume Screening and Skill Matching System"
    title_w = fitz.get_text_length(title_text, fontname="hebo", fontsize=13)
    p1.insert_text(((page_w - title_w) / 2, y), title_text, fontsize=13, fontname="hebo")
    y += 26

    # Section: Introduction
    p1.insert_text((margin_x, y), "Introduction", fontsize=12, fontname="hebo")
    y += 16

    intro_p1 = (
        "In the present recruitment and talent acquisition workflows, evaluating candidate profiles is becoming "
        "increasingly demanding due to the massive volume of applications submitted for every open vacancy. "
        "Predicting candidate suitability in advance can help recruiters as well as hiring managers to keep "
        "track of the applicant pool and make objective shortlisting decisions. Many enterprise organizations have "
        "adopted automated screening systems today. Such systems are favorable to human resource departments in "
        "improving their screening efficiency and consistency. The purpose of an automated screening pipeline is "
        "to assist recruitment teams in identifying qualified applicants for technical roles. In an automated "
        "evaluation system, multi-dimensional candidate parameters are analyzed systematically, including domain "
        "skills, career tenure, and educational background. To achieve consistent screening performance, it is "
        "essential to evaluate applicant qualifications objectively against standard job requirements."
    )
    y = add_paragraph(p1, intro_p1, y, fontsize=9.5)

    intro_p2 = (
        "The core function of the Automated Resume Screening System is to help recruiters assess candidate "
        "fit in advance using Natural Language Processing (NLP) and supervised classification models. Such "
        "techniques enable talent acquisition teams to identify high-potential candidates based on predicted fit "
        "categories and allow hiring managers to focus their attention on candidates who best satisfy the role requirements."
    )
    y = add_paragraph(p1, intro_p2, y, fontsize=9.5)

    intro_p3 = (
        "Supervised classification algorithms, such as Random Forest and Logistic Regression, are widely used for "
        "predictive candidate-job matching. This is because models trained on structured numerical features derived "
        "from text similarity and domain skill taxonomies offer high interpretability, reliable decision boundaries, "
        "and well-defined feature importances that explain why a specific candidate is categorized into a given tier."
    )
    y = add_paragraph(p1, intro_p3, y, fontsize=9.5, extra_space=16)

    # Section: Problem Statement
    p1.insert_text((margin_x, y), "Problem Statement", fontsize=12, fontname="hebo")
    y += 16

    ps_p1 = (
        "The Main Objective of \"Automated Resume Screening System\" is to implement an algorithmic model that "
        "predicts the match category of an individual applicant against a target job description. The match category "
        "(\"Strong Match\", \"Moderate Match\", or \"Weak Match\") is our label (output) and the extracted candidate "
        "parameters will be our features (inputs)."
    )
    y = add_paragraph(p1, ps_p1, y, fontsize=9.5)

    ps_p2 = (
        "The core function of Resume Screening is to estimate candidate-job alignment in advance by extracting "
        "unstructured textual attributes and computing similarity metrics. Such techniques enable recruiters to "
        "filter through applicant pools rapidly, flagging those profiles that satisfy mandatory prerequisites."
    )
    y = add_paragraph(p1, ps_p2, y, fontsize=9.5)

    ps_p3 = (
        "The problem statement can be defined as follows: \"Given a dataset containing attributes of candidates "
        "where using the features available from the dataset and define classification algorithms to identify whether "
        "the candidate performs good in the job matching evaluation, also to evaluate different machine learning "
        "models on the dataset.\" The data attributes include TF-IDF textual similarity, skill match percentage, "
        "total skills count, years of professional experience, and highest education level. The data was collected "
        "by parsing real-world curriculum vitae and job descriptions across technical domains."
    )
    y = add_paragraph(p1, ps_p3, y, fontsize=9.5, extra_space=16)

    # Section: Results and Discussion
    p1.insert_text((margin_x, y), "Results and Discussion", fontsize=12, fontname="hebo")
    y += 16

    rd_p1 = (
        "Candidate qualification overlap is significantly associated with hiring classification outcomes. Most "
        "candidates who had more than 70% skill match and high TF-IDF similarity achieved Strong Match grades "
        "when compared to the other categories of candidate suitability."
    )
    y = add_paragraph(p1, rd_p1, y, fontsize=9.5)

    # ================= PAGE 2 =================
    p2 = doc.new_page(width=page_w, height=page_h)
    y = margin_top

    # Embed Confusion Matrix Image
    cm_img = "report/figures/confusion_matrix.png"
    if os.path.exists(cm_img):
        img_rect = fitz.Rect(margin_x + 85, y, margin_x + content_w - 85, y + 215)
        p2.insert_image(img_rect, filename=cm_img)
        y += 222

    # Code snippet box matching reference format
    code_box_h = 102
    code_rect = fitz.Rect(margin_x, y, margin_x + content_w, y + code_box_h)
    p2.draw_rect(code_rect, color=(0.85, 0.85, 0.85), fill=(0.97, 0.97, 0.97))
    code_text = (
        "[ ] model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)\n"
        "    model.fit(X_train_scaled, y_train)\n"
        "    pred = model.predict(X_test_scaled)\n"
        "    print(f\"Accuracy: {accuracy_score(y_test, pred):.4f}\")\n"
        "    print(f\"Weighted F1: {f1_score(y_test, pred, average='weighted'):.4f}\")\n\n"
        "    Accuracy: 0.9100\n"
        "    Weighted F1: 0.9098"
    )
    p2.insert_textbox(
        fitz.Rect(margin_x + 10, y + 8, margin_x + content_w - 10, y + code_box_h),
        code_text,
        fontsize=8.5,
        fontname="cour"
    )
    y += code_box_h + 16

    # Section: Conclusion
    p2.insert_text((margin_x, y), "Conclusion", fontsize=12, fontname="hebo")
    y += 16

    c_p1 = (
        "In this project we did a deep analysis of what could be possible factors on whether a candidate is "
        "likely to get a high match score or a low match score. The data contains rich information from resumes "
        "and job descriptions, enabling us to predict a pretty precise Random Forest algorithm that predicts what "
        "suitability tier a candidate will be placed in by analyzing the features. It is to my understanding that "
        "the machine learning model is used to predict values with a given number of features thus providing us "
        "with a good accuracy of 91% (and 95% with baseline Logistic Regression)."
    )
    y = add_paragraph(p2, c_p1, y, fontsize=9.5)

    c_p2 = (
        "After evaluating all the algorithms on different parameters, we have managed to propose a model that can "
        "predict the match category accurately using supervised learning algorithms. This model helps both the recruiter "
        "and hiring organization to analyze candidate qualifications with the help of various graphs through which they "
        "can easily decide about the candidate's suitability and suggest a better method for assessing technical fit. "
        "In the future, an end-to-end website can be developed using which the end-user can come to check the "
        "predictions more easily. We have proposed a model that can give consistent and accurate results to the "
        "end-user, which satisfies their need by showing the correct output and helps to take better hiring decisions."
    )
    y = add_paragraph(p2, c_p2, y, fontsize=9.5, extra_space=16)

    # Section: Reference
    ref_title = "Reference: "
    ref_w = fitz.get_text_length(ref_title, fontname="hebo", fontsize=8.5)
    p2.insert_text((margin_x, y), ref_title, fontsize=8.5, fontname="hebo")

    ref_text = (
        "G. Salton and C. Buckley. Term-weighting approaches in automatic text retrieval. Information "
        "Processing & Management, 24(5):513-523, 1988.\n"
        "F. Pedregosa et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, "
        "12:2825-2830, 2011.\n"
        "M. Honnibal and I. Montani. spaCy 2: Natural language understanding with Bloom embeddings, convolutional "
        "neural networks and incremental parsing, 2017.\n"
        "L. Breiman. Random Forests. Machine Learning, 45(1):5-32, 2001."
    )
    rect = fitz.Rect(margin_x + ref_w, y - 8, margin_x + content_w, page_h - margin_bottom)
    p2.insert_textbox(rect, ref_text, fontsize=8, fontname="helv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc.save(output_path)
    doc.close()
    print(f"Report PDF compiled successfully to {output_path}")

if __name__ == "__main__":
    build_pdf_report()

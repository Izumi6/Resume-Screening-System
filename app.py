"""
Resume Screening System
-----------------------
Main Streamlit application.

Allows users to upload a resume PDF and enter a job description,
then analyzes the match using NLP and machine learning.

Run with: streamlit run app.py
"""

import streamlit as st
import os
import sys
import logging

# make sure project root is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pdf_extractor import extract_text_from_pdf
from src.text_preprocessor import TextPreprocessor
from src.skill_extractor import SkillExtractor
from src.similarity_engine import SimilarityEngine
from src.ml_classifier import ResumeClassifier
from src.utils import validate_pdf, get_match_category, setup_logging
from visualizations.charts import (
    create_match_gauge, create_score_breakdown,
    create_skill_comparison, create_skill_radar,
    create_confusion_matrix_chart, create_feature_importance_chart,
    create_probability_chart,
)

setup_logging()
logger = logging.getLogger(__name__)


# --- Page config ---
st.set_page_config(
    page_title="Resume Screening System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Custom CSS ---
def load_custom_css():
    st.markdown("""
    <style>
        /* import Inter font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* global overrides */
        .stApp {
            font-family: 'Inter', sans-serif;
        }
        
        /* hide the default streamlit header and footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* card styling */
        .metric-card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 20px;
            margin: 8px 0;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: #6366f1;
        }
        .metric-card h3 {
            color: #94a3b8;
            font-size: 13px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }
        .metric-card .value {
            color: #e2e8f0;
            font-size: 28px;
            font-weight: 700;
        }
        
        /* badge styles for skills */
        .skill-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 500;
            margin: 3px;
        }
        .skill-matched {
            background-color: rgba(34, 197, 94, 0.15);
            color: #22c55e;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }
        .skill-missing {
            background-color: rgba(239, 68, 68, 0.15);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        .skill-extra {
            background-color: rgba(99, 102, 241, 0.15);
            color: #818cf8;
            border: 1px solid rgba(99, 102, 241, 0.3);
        }
        
        /* section headers */
        .section-header {
            color: #e2e8f0;
            font-size: 20px;
            font-weight: 600;
            margin: 20px 0 10px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid #334155;
        }
        
        /* recommendation badge */
        .recommendation {
            display: inline-block;
            padding: 8px 24px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            letter-spacing: 0.3px;
        }
        .rec-strong {
            background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(34,197,94,0.05));
            color: #22c55e;
            border: 1px solid rgba(34,197,94,0.4);
        }
        .rec-moderate {
            background: linear-gradient(135deg, rgba(245,158,11,0.2), rgba(245,158,11,0.05));
            color: #f59e0b;
            border: 1px solid rgba(245,158,11,0.4);
        }
        .rec-weak {
            background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(239,68,68,0.05));
            color: #ef4444;
            border: 1px solid rgba(239,68,68,0.4);
        }
        
        /* info card for extracted details */
        .info-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 16px 20px;
            margin: 8px 0;
        }
        .info-card .label {
            color: #94a3b8;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .info-card .content {
            color: #e2e8f0;
            font-size: 15px;
            margin-top: 4px;
        }
        
        /* sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid #1e293b;
        }
        
        /* tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 20px;
        }
    </style>
    """, unsafe_allow_html=True)


def render_metric_card(title, value, suffix=""):
    """Render a styled metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <div class="value">{value}{suffix}</div>
    </div>
    """, unsafe_allow_html=True)


def render_skill_badges(skills, badge_class):
    """Render a list of skills as colored badges."""
    if not skills:
        st.markdown("*None detected*")
        return
    
    badges_html = ""
    for skill in skills:
        badges_html += f'<span class="skill-badge {badge_class}">{skill}</span>'
    st.markdown(badges_html, unsafe_allow_html=True)


def get_sample_job_description():
    """Return a sample JD for quick testing."""
    return """Senior Data Scientist

We are looking for an experienced Data Scientist to join our analytics team.

Requirements:
- 3+ years of experience in data science or machine learning
- Strong proficiency in Python and SQL
- Experience with machine learning frameworks (Scikit-learn, TensorFlow, or PyTorch)
- Knowledge of statistical modeling and data analysis
- Familiarity with NLP and deep learning techniques
- Experience with data visualization tools (Matplotlib, Seaborn, Plotly, or Tableau)
- Understanding of cloud platforms (AWS, GCP, or Azure)
- Proficiency in Pandas and NumPy for data manipulation
- Experience with Git version control
- Strong problem solving and communication skills

Nice to have:
- Experience with Docker and CI/CD pipelines
- Knowledge of big data technologies (Spark, Hadoop)
- Master's degree or PhD in Computer Science, Statistics, or related field
- Experience with A/B testing and recommendation systems
"""


@st.cache_resource
def load_components():
    """Initialize and cache all the heavy components."""
    preprocessor = TextPreprocessor()
    skill_extractor = SkillExtractor()
    similarity_engine = SimilarityEngine()
    classifier = ResumeClassifier()
    
    # try to load a pre-trained model
    classifier.load_model()
    
    return preprocessor, skill_extractor, similarity_engine, classifier


def run_analysis(resume_file, jd_text, preprocessor, skill_extractor,
                 similarity_engine, classifier):
    """
    Run the complete analysis pipeline on a resume against a job description.
    
    Returns a dict with all the analysis results, or None if something fails.
    """
    results = {}
    
    # step 1: extract text from PDF
    raw_text = extract_text_from_pdf(resume_file)
    if not raw_text.strip():
        st.error("Could not extract text from the PDF. The file might be scanned or corrupted.")
        return None
    
    results["raw_text"] = raw_text
    
    # step 2: extract contact info (before preprocessing strips them)
    contact_info = preprocessor.extract_contact_info(raw_text)
    results["contact_info"] = contact_info
    
    # step 3: preprocess both texts
    clean_resume = preprocessor.preprocess(raw_text)
    clean_jd = preprocessor.preprocess(jd_text)
    
    if not clean_resume:
        st.error("Resume text was empty after preprocessing.")
        return None
    
    results["clean_resume"] = clean_resume
    results["clean_jd"] = clean_jd
    
    # step 4: extract skills from both
    resume_skills = skill_extractor.extract_skills(raw_text)
    jd_skills = skill_extractor.extract_skills(jd_text)
    resume_skills_flat = skill_extractor.extract_skills_flat(raw_text)
    jd_skills_flat = skill_extractor.extract_skills_flat(jd_text)
    
    results["resume_skills"] = resume_skills
    results["jd_skills"] = jd_skills
    results["resume_skills_flat"] = resume_skills_flat
    results["jd_skills_flat"] = jd_skills_flat
    
    # step 5: extract education and experience
    education = skill_extractor.extract_education(raw_text)
    experience_years = skill_extractor.extract_experience_years(raw_text)
    education_level = skill_extractor.get_highest_education_level(raw_text)
    
    results["education"] = education
    results["experience_years"] = experience_years
    results["education_level"] = education_level
    
    # step 6: compute TF-IDF similarity
    tfidf_score = similarity_engine.compute_tfidf_similarity(clean_resume, clean_jd)
    results["tfidf_score"] = tfidf_score
    
    # step 7: compute skill match
    skill_match = similarity_engine.compute_skill_match(resume_skills_flat, jd_skills_flat)
    results["skill_match"] = skill_match
    
    # step 8: compute composite score
    composite = similarity_engine.compute_composite_score(
        tfidf_score=tfidf_score,
        skill_match_pct=skill_match["match_percentage"],
        experience_years=experience_years,
        education_level=education_level,
    )
    results["composite"] = composite
    
    # step 9: ML classification
    features = {
        "tfidf_similarity": tfidf_score,
        "skill_match_pct": skill_match["match_percentage"],
        "experience_years": experience_years,
        "education_level": education_level,
        "total_skills_count": len(resume_skills_flat),
    }
    prediction = classifier.predict(features)
    results["prediction"] = prediction
    results["features"] = features
    
    # step 10: get top TF-IDF terms
    results["top_resume_terms"] = similarity_engine.get_top_tfidf_terms(clean_resume, top_n=10)
    results["top_jd_terms"] = similarity_engine.get_top_tfidf_terms(clean_jd, top_n=10)
    
    # step 11: get match category
    label, color = get_match_category(composite["overall_score"])
    results["match_label"] = label
    results["match_color"] = color
    
    return results


def render_dashboard_tab(results, classifier):
    """Render the main dashboard overview tab."""
    
    # -- match score gauge and recommendation --
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = create_match_gauge(results["composite"]["overall_score"])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        label = results["match_label"]
        if "Strong" in label:
            css_class = "rec-strong"
        elif "Moderate" in label:
            css_class = "rec-moderate"
        else:
            css_class = "rec-weak"
        
        st.markdown(f"""
        <div style="text-align: center;">
            <div class="recommendation {css_class}">{label}</div>
            <br>
            <p style="color: #94a3b8; font-size: 13px;">
                Classified by: {results['prediction']['model_used']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # prediction probabilities
        if results["prediction"]["probabilities"]:
            fig = create_probability_chart(results["prediction"]["probabilities"])
            st.plotly_chart(fig, use_container_width=True)
    
    # -- score component breakdown --
    st.markdown('<div class="section-header">Score Breakdown</div>', unsafe_allow_html=True)
    
    components = results["composite"]["components"]
    cols = st.columns(4)
    
    for i, (name, value) in enumerate(components.items()):
        with cols[i]:
            render_metric_card(name, f"{value:.1f}", "%")
    
    fig = create_score_breakdown(components)
    st.plotly_chart(fig, use_container_width=True)
    
    # -- quick stats row --
    st.markdown('<div class="section-header">Candidate Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Skills Found", len(results["resume_skills_flat"]))
    with c2:
        render_metric_card("Skills Matched",
                          f"{len(results['skill_match']['matched'])}/{len(results['jd_skills_flat'])}")
    with c3:
        render_metric_card("Experience", f"{results['experience_years']}", " yrs")
    with c4:
        edu = results["education"]
        degree = edu[0]["degree"] if edu else "Not detected"
        render_metric_card("Education", degree)


def render_skills_tab(results):
    """Render the detailed skill analysis tab."""
    
    skill_match = results["skill_match"]
    
    # -- match percentage header --
    pct = skill_match["match_percentage"]
    st.markdown(f"""
    <div class="metric-card" style="text-align: center;">
        <h3>Skill Match Rate</h3>
        <div class="value" style="font-size: 36px;">{pct:.1f}%</div>
        <p style="color: #94a3b8; margin-top: 4px;">
            {len(skill_match['matched'])} of {len(skill_match['matched']) + len(skill_match['missing'])} required skills found
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # -- matched vs missing skills --
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">Matched Skills</div>', unsafe_allow_html=True)
        render_skill_badges(skill_match["matched"], "skill-matched")
    
    with col2:
        st.markdown('<div class="section-header">Missing Skills</div>', unsafe_allow_html=True)
        render_skill_badges(skill_match["missing"], "skill-missing")
    
    # -- extra skills the candidate has --
    if skill_match["extra"]:
        st.markdown('<div class="section-header">Additional Skills (not in JD)</div>',
                   unsafe_allow_html=True)
        render_skill_badges(skill_match["extra"], "skill-extra")
    
    # -- skill comparison chart --
    st.markdown('<div class="section-header">Skill Comparison</div>', unsafe_allow_html=True)
    fig = create_skill_comparison(skill_match["matched"], skill_match["missing"])
    st.plotly_chart(fig, use_container_width=True)
    
    # -- radar chart by category --
    st.markdown('<div class="section-header">Coverage by Category</div>', unsafe_allow_html=True)
    fig = create_skill_radar(results["resume_skills"], results["jd_skills"])
    st.plotly_chart(fig, use_container_width=True)


def render_resume_tab(results):
    """Render the resume details tab."""
    
    # -- contact information --
    contact = results["contact_info"]
    st.markdown('<div class="section-header">Contact Information</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        name = contact.get("name", "Not detected")
        st.markdown(f"""
        <div class="info-card">
            <div class="label">Name</div>
            <div class="content">{name}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        email = contact["emails"][0] if contact["emails"] else "Not detected"
        st.markdown(f"""
        <div class="info-card">
            <div class="label">Email</div>
            <div class="content">{email}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        phone = contact["phones"][0] if contact["phones"] else "Not detected"
        st.markdown(f"""
        <div class="info-card">
            <div class="label">Phone</div>
            <div class="content">{phone}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # -- education --
    st.markdown('<div class="section-header">Education</div>', unsafe_allow_html=True)
    education = results["education"]
    if education:
        for edu in education:
            level_labels = {1: "High School", 2: "Diploma", 3: "Undergraduate",
                          4: "Postgraduate", 5: "Doctorate"}
            level_text = level_labels.get(edu["level"], "")
            st.markdown(f"""
            <div class="info-card">
                <div class="label">{level_text} (Level {edu['level']})</div>
                <div class="content">{edu['degree']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No education qualifications detected in the resume.")
    
    # -- experience --
    st.markdown('<div class="section-header">Experience</div>', unsafe_allow_html=True)
    exp = results["experience_years"]
    st.markdown(f"""
    <div class="info-card">
        <div class="label">Estimated Experience</div>
        <div class="content">{exp} year{'s' if exp != 1 else ''}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # -- all skills by category --
    st.markdown('<div class="section-header">All Detected Skills</div>', unsafe_allow_html=True)
    resume_skills = results["resume_skills"]
    if resume_skills:
        for category, skills in resume_skills.items():
            st.markdown(f"**{category}**")
            render_skill_badges(skills, "skill-extra")
            st.markdown("")
    else:
        st.info("No skills were detected.")
    
    # -- top TF-IDF terms --
    st.markdown('<div class="section-header">Top Keywords (by TF-IDF weight)</div>',
               unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Resume Keywords**")
        for term in results.get("top_resume_terms", []):
            st.markdown(f"- `{term['term']}` ({term['score']:.4f})")
    with col2:
        st.markdown("**Job Description Keywords**")
        for term in results.get("top_jd_terms", []):
            st.markdown(f"- `{term['term']}` ({term['score']:.4f})")
    
    # -- raw text (collapsed) --
    with st.expander("View Raw Extracted Text"):
        st.text(results["raw_text"][:5000])
        if len(results["raw_text"]) > 5000:
            st.caption(f"(Showing first 5000 of {len(results['raw_text'])} characters)")


def render_model_tab(classifier, results):
    """Render the model insights tab."""
    
    # -- classification result --
    st.markdown('<div class="section-header">Classification Result</div>', unsafe_allow_html=True)
    
    pred = results["prediction"]
    st.markdown(f"""
    <div class="info-card">
        <div class="label">Predicted Category</div>
        <div class="content" style="font-size: 20px; font-weight: 600;">{pred['prediction']}</div>
        <div class="label" style="margin-top: 8px;">Model Used</div>
        <div class="content">{pred['model_used']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # -- feature values used --
    st.markdown('<div class="section-header">Input Features</div>', unsafe_allow_html=True)
    features = results["features"]
    cols = st.columns(len(features))
    feature_labels = {
        "tfidf_similarity": "TF-IDF Score",
        "skill_match_pct": "Skill Match %",
        "experience_years": "Experience (yrs)",
        "education_level": "Education Level",
        "total_skills_count": "Total Skills",
    }
    for i, (key, val) in enumerate(features.items()):
        with cols[i]:
            display_val = f"{val:.3f}" if isinstance(val, float) else str(val)
            render_metric_card(feature_labels.get(key, key), display_val)
    
    # -- feature importance --
    if classifier.is_trained:
        st.markdown('<div class="section-header">Feature Importance</div>',
                   unsafe_allow_html=True)
        importance = classifier.get_feature_importance()
        if importance:
            fig = create_feature_importance_chart(importance)
            st.plotly_chart(fig, use_container_width=True)
    
    # -- training metrics --
    if classifier.training_metrics:
        st.markdown('<div class="section-header">Model Training Metrics</div>',
                   unsafe_allow_html=True)
        
        metrics = classifier.training_metrics
        
        # show RF metrics
        if "random_forest" in metrics:
            rf = metrics["random_forest"]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                render_metric_card("Accuracy", f"{rf['accuracy']:.2%}")
            with col2:
                render_metric_card("Precision", f"{rf['precision_weighted']:.2%}")
            with col3:
                render_metric_card("Recall", f"{rf['recall_weighted']:.2%}")
            with col4:
                render_metric_card("F1 Score", f"{rf['f1_weighted']:.2%}")
            
            # confusion matrix
            if "confusion_matrix" in rf and "labels" in rf:
                st.markdown("**Confusion Matrix (Random Forest)**")
                fig = create_confusion_matrix_chart(rf["confusion_matrix"], rf["labels"])
                st.plotly_chart(fig, use_container_width=True)
            
            # classification report
            with st.expander("Full Classification Report"):
                st.text(rf.get("classification_report", "Not available"))
        
        # CV scores
        if "cv_scores" in metrics:
            cv = metrics["cv_scores"]
            st.markdown(f"**Cross-Validation:** {cv['mean']:.4f} (+/- {cv['std']:.4f})")
    else:
        st.info(
            "No trained model found. Run 'python notebooks/train_model.py' to train the "
            "classifier and see detailed metrics here."
        )


def main():
    load_custom_css()
    
    # -- sidebar --
    with st.sidebar:
        st.markdown("""
        <h2 style="color: #e2e8f0; margin-bottom: 4px;">Resume Screening System</h2>
        <p style="color: #64748b; font-size: 13px; margin-top: 0;">
            NLP and ML-powered candidate matching
        </p>
        <hr style="border-color: #334155;">
        """, unsafe_allow_html=True)
        
        # file upload
        st.markdown("**Upload Resume (PDF)**")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload the candidate's resume in PDF format. Max 10MB.",
            label_visibility="collapsed",
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # job description input
        st.markdown("**Job Description**")
        
        # sample JD button
        if st.button("Load Sample JD", use_container_width=True):
            st.session_state["jd_text"] = get_sample_job_description()
        
        jd_text = st.text_area(
            "Enter the job description",
            value=st.session_state.get("jd_text", ""),
            height=250,
            placeholder="Paste the job description here...",
            label_visibility="collapsed",
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # analyze button
        analyze_clicked = st.button(
            "Analyze Resume",
            type="primary",
            use_container_width=True,
            disabled=(uploaded_file is None or not jd_text.strip()),
        )
        
        # version info at bottom
        st.markdown("""
        <hr style="border-color: #334155;">
        <p style="color: #475569; font-size: 11px; text-align: center;">
            Built with Python, spaCy, Scikit-learn, and Streamlit<br>
            v1.0.0
        </p>
        """, unsafe_allow_html=True)
    
    # -- main content area --
    
    # show welcome screen if no analysis has been done yet
    if "analysis_results" not in st.session_state and not analyze_clicked:
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px;">
            <h1 style="color: #e2e8f0; font-size: 36px; font-weight: 700;">
                Resume Screening System
            </h1>
            <p style="color: #94a3b8; font-size: 18px; max-width: 600px; margin: 20px auto;">
                Upload a resume and provide a job description to get an instant 
                match analysis with skill gap identification and scoring.
            </p>
            <div style="display: flex; justify-content: center; gap: 40px; margin-top: 40px;">
                <div style="text-align: center;">
                    <div style="font-size: 32px; color: #6366f1; font-weight: 700;">1</div>
                    <p style="color: #94a3b8; font-size: 14px;">Upload a PDF resume</p>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 32px; color: #8b5cf6; font-weight: 700;">2</div>
                    <p style="color: #94a3b8; font-size: 14px;">Enter a job description</p>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 32px; color: #a78bfa; font-weight: 700;">3</div>
                    <p style="color: #94a3b8; font-size: 14px;">Get the match analysis</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # run analysis if button was clicked
    if analyze_clicked:
        # validate the PDF first
        is_valid, error_msg = validate_pdf(uploaded_file)
        if not is_valid:
            st.error(error_msg)
            return
        
        # load components
        with st.spinner("Loading NLP models..."):
            preprocessor, skill_extractor, similarity_engine, classifier = load_components()
        
        # run the pipeline
        with st.spinner("Analyzing resume..."):
            results = run_analysis(
                uploaded_file, jd_text,
                preprocessor, skill_extractor, similarity_engine, classifier
            )
        
        if results:
            st.session_state["analysis_results"] = results
            st.session_state["classifier"] = classifier
        else:
            return
    
    # display results
    if "analysis_results" in st.session_state:
        results = st.session_state["analysis_results"]
        classifier = st.session_state.get("classifier")
        
        if classifier is None:
            _, _, _, classifier = load_components()
        
        # create tabs (no emojis)
        tab1, tab2, tab3, tab4 = st.tabs([
            "Dashboard", "Skills Analysis", "Resume Details", "Model Insights"
        ])
        
        with tab1:
            render_dashboard_tab(results, classifier)
        
        with tab2:
            render_skills_tab(results)
        
        with tab3:
            render_resume_tab(results)
        
        with tab4:
            render_model_tab(classifier, results)


if __name__ == "__main__":
    main()

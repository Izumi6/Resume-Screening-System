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
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Custom CSS ---
def load_custom_css():
    st.markdown("""
    <style>
        /* ===== FONT IMPORTS ===== */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* ===== ROOT VARIABLES ===== */
        :root {
            --bg-primary: #0a0e1a;
            --bg-secondary: #111827;
            --bg-card: rgba(17, 24, 39, 0.7);
            --bg-card-hover: rgba(30, 41, 59, 0.85);
            --border-subtle: rgba(99, 102, 241, 0.15);
            --border-hover: rgba(99, 102, 241, 0.5);
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-indigo: #6366f1;
            --accent-violet: #8b5cf6;
            --accent-purple: #a78bfa;
            --accent-cyan: #06b6d4;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --glow-indigo: rgba(99, 102, 241, 0.4);
            --glow-violet: rgba(139, 92, 246, 0.3);
        }

        /* ===== GLOBAL OVERRIDES ===== */
        .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
        }

        /* reset p margins inside our custom components */
        .metric-card p, .info-card p, .skill-match-hero p,
        .step-card p, .stat-item p, .results-header p,
        .hero-container p, .hero-container h1 {
            margin: 0;
            padding: 0;
        }

        /* hide default streamlit chrome */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display: none;}

        /* ===== ANIMATED BACKGROUND MESH ===== */
        .stApp::before {
            content: '';
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background:
                radial-gradient(ellipse 80% 60% at 10% 20%, rgba(99,102,241,0.08), transparent 50%),
                radial-gradient(ellipse 60% 80% at 90% 80%, rgba(139,92,246,0.06), transparent 50%),
                radial-gradient(ellipse 70% 50% at 50% 50%, rgba(6,182,212,0.04), transparent 50%);
            pointer-events: none;
            z-index: 0;
            animation: meshShift 20s ease-in-out infinite alternate;
        }

        @keyframes meshShift {
            0% { opacity: 0.8; }
            50% { opacity: 1; }
            100% { opacity: 0.8; }
        }

        /* ===== SCROLLBAR STYLING ===== */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: var(--bg-primary);
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(99, 102, 241, 0.3);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(99, 102, 241, 0.6);
        }

        /* ===== SIDEBAR ===== */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0c1120 0%, #0a0e1a 100%);
            border-right: 1px solid var(--border-subtle);
        }
        section[data-testid="stSidebar"]::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-indigo), var(--accent-violet), var(--accent-cyan));
            z-index: 10;
        }

        /* ===== GLASSMORPHISM METRIC CARD ===== */
        .metric-card {
            background: linear-gradient(145deg, rgba(17,24,39,0.8) 0%, rgba(15,23,42,0.9) 100%);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 24px;
            margin: 8px 0;
            position: relative;
            overflow: hidden;
            transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--accent-indigo), var(--accent-violet), transparent);
            opacity: 0;
            transition: opacity 0.35s ease;
        }
        .metric-card:hover {
            transform: translateY(-4px);
            border-color: var(--border-hover);
            box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15), 0 2px 8px rgba(0,0,0,0.3);
        }
        .metric-card:hover::before {
            opacity: 1;
        }
        .metric-card h3 {
            color: var(--text-secondary);
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            margin-bottom: 10px;
        }
        .metric-card .value {
            color: var(--text-primary);
            font-size: 30px;
            font-weight: 800;
            background: linear-gradient(135deg, #f1f5f9, #cbd5e1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        /* ===== SKILL BADGES ===== */
        .skill-badge {
            display: inline-flex;
            align-items: center;
            padding: 6px 14px;
            border-radius: 24px;
            font-size: 12.5px;
            font-weight: 600;
            margin: 3px;
            letter-spacing: 0.3px;
            transition: all 0.25s ease;
        }
        .skill-badge:hover {
            transform: translateY(-1px) scale(1.03);
        }
        .skill-matched {
            background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.05));
            color: #34d399;
            border: 1px solid rgba(16,185,129,0.3);
            box-shadow: 0 0 12px rgba(16,185,129,0.08);
        }
        .skill-missing {
            background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
            color: #f87171;
            border: 1px solid rgba(239,68,68,0.3);
            box-shadow: 0 0 12px rgba(239,68,68,0.08);
        }
        .skill-extra {
            background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.08));
            color: #a5b4fc;
            border: 1px solid rgba(99,102,241,0.3);
            box-shadow: 0 0 12px rgba(99,102,241,0.08);
        }

        /* ===== SECTION HEADERS ===== */
        .section-header {
            color: var(--text-primary);
            font-size: 18px;
            font-weight: 700;
            margin: 28px 0 14px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid transparent;
            border-image: linear-gradient(90deg, var(--accent-indigo), transparent) 1;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        /* ===== RECOMMENDATION BADGES ===== */
        .recommendation {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 12px 32px;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 700;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
        }
        .rec-strong {
            background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(16,185,129,0.05));
            color: #34d399;
            border: 1px solid rgba(16,185,129,0.4);
            box-shadow: 0 0 24px rgba(16,185,129,0.12);
        }
        .rec-moderate {
            background: linear-gradient(135deg, rgba(245,158,11,0.2), rgba(245,158,11,0.05));
            color: #fbbf24;
            border: 1px solid rgba(245,158,11,0.4);
            box-shadow: 0 0 24px rgba(245,158,11,0.12);
        }
        .rec-weak {
            background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(239,68,68,0.05));
            color: #f87171;
            border: 1px solid rgba(239,68,68,0.4);
            box-shadow: 0 0 24px rgba(239,68,68,0.12);
        }

        /* ===== INFO CARDS ===== */
        .info-card {
            background: linear-gradient(145deg, rgba(17,24,39,0.75), rgba(15,23,42,0.85));
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 18px 22px;
            margin: 8px 0;
            transition: all 0.3s ease;
        }
        .info-card:hover {
            border-color: rgba(99, 102, 241, 0.3);
            box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        }
        .info-card .label {
            color: var(--text-muted);
            font-size: 10.5px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .info-card .content {
            color: var(--text-primary);
            font-size: 15px;
            font-weight: 500;
            margin-top: 6px;
        }

        /* ===== TAB STYLING ===== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: rgba(17,24,39,0.5);
            border-radius: 12px;
            padding: 4px;
            border: 1px solid var(--border-subtle);
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 24px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 13.5px;
            letter-spacing: 0.3px;
            transition: all 0.2s ease;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, var(--accent-indigo), var(--accent-violet)) !important;
            color: white !important;
        }

        /* ===== FILE UPLOADER ===== */
        .stFileUploader > div {
            border-radius: 12px !important;
            border: 2px dashed var(--border-subtle) !important;
            background: rgba(17,24,39,0.5) !important;
            transition: all 0.3s ease;
        }
        .stFileUploader > div:hover {
            border-color: var(--accent-indigo) !important;
            background: rgba(99,102,241,0.05) !important;
        }

        /* ===== TEXT AREA ===== */
        .stTextArea textarea {
            border-radius: 12px !important;
            border: 1px solid var(--border-subtle) !important;
            background: rgba(17,24,39,0.6) !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 13.5px !important;
            transition: all 0.3s ease;
        }
        .stTextArea textarea:focus {
            border-color: var(--accent-indigo) !important;
            box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
        }

        /* ===== BUTTONS ===== */
        .stButton > button {
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            letter-spacing: 0.3px !important;
            padding: 10px 24px !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--accent-indigo), var(--accent-violet)) !important;
            border: none !important;
            box-shadow: 0 4px 16px rgba(99,102,241,0.3) !important;
        }
        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 6px 24px rgba(99,102,241,0.5) !important;
            transform: translateY(-1px) !important;
        }
        .stButton > button[kind="secondary"] {
            background: rgba(17,24,39,0.6) !important;
            border: 1px solid var(--border-subtle) !important;
        }
        .stButton > button[kind="secondary"]:hover {
            border-color: var(--accent-indigo) !important;
            background: rgba(99,102,241,0.1) !important;
        }

        /* ===== EXPANDER ===== */
        .streamlit-expanderHeader {
            border-radius: 10px !important;
            background: rgba(17,24,39,0.5) !important;
            border: 1px solid var(--border-subtle) !important;
        }

        /* ===== SPINNER ===== */
        .stSpinner > div {
            border-color: var(--accent-indigo) transparent transparent transparent !important;
        }

        /* ===== LANDING PAGE ===== */
        .hero-container {
            text-align: center;
            padding: 40px 20px 30px 20px;
            position: relative;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.08));
            border: 1px solid rgba(99,102,241,0.25);
            border-radius: 100px;
            padding: 8px 20px;
            font-size: 12px;
            font-weight: 600;
            color: var(--accent-purple);
            letter-spacing: 0.5px;
            margin-bottom: 24px;
            animation: fadeInDown 0.6s ease;
        }

        .hero-title {
            font-size: 52px;
            font-weight: 900;
            line-height: 1.1;
            margin-bottom: 16px;
            background: linear-gradient(135deg, #f1f5f9 0%, #6366f1 50%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: fadeInUp 0.7s ease;
        }

        .hero-subtitle {
            color: var(--text-secondary);
            font-size: 17px;
            line-height: 1.6;
            max-width: 600px;
            margin: 0 auto 48px auto;
            animation: fadeInUp 0.8s ease;
        }

        /* ===== STEP CARDS ===== */
        .steps-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 24px;
            max-width: 900px;
            margin: 0 auto 50px auto;
            animation: fadeInUp 0.9s ease;
        }
        .step-card {
            background: linear-gradient(145deg, rgba(17,24,39,0.8), rgba(15,23,42,0.9));
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 32px 24px;
            text-align: center;
            position: relative;
            overflow: hidden;
            transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .step-card:hover {
            transform: translateY(-6px);
            border-color: var(--border-hover);
            box-shadow: 0 16px 48px rgba(99,102,241,0.15);
        }
        .step-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-indigo), var(--accent-violet));
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        .step-card:hover::before {
            opacity: 1;
        }
        .step-num {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 48px;
            height: 48px;
            border-radius: 14px;
            font-size: 20px;
            font-weight: 800;
            margin-bottom: 16px;
        }
        .step-num-1 {
            background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(99,102,241,0.08));
            color: var(--accent-indigo);
            border: 1px solid rgba(99,102,241,0.3);
        }
        .step-num-2 {
            background: linear-gradient(135deg, rgba(139,92,246,0.2), rgba(139,92,246,0.08));
            color: var(--accent-violet);
            border: 1px solid rgba(139,92,246,0.3);
        }
        .step-num-3 {
            background: linear-gradient(135deg, rgba(6,182,212,0.2), rgba(6,182,212,0.08));
            color: var(--accent-cyan);
            border: 1px solid rgba(6,182,212,0.3);
        }
        .step-title {
            color: var(--text-primary);
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .step-desc {
            color: var(--text-muted);
            font-size: 13px;
            line-height: 1.5;
        }

        /* ===== FEATURE PILLS ===== */
        .feature-row {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 12px;
            max-width: 800px;
            margin: 0 auto 40px auto;
            animation: fadeInUp 1s ease;
        }
        .feature-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(17,24,39,0.6);
            border: 1px solid var(--border-subtle);
            border-radius: 100px;
            padding: 8px 16px;
            font-size: 12px;
            font-weight: 500;
            color: var(--text-secondary);
            transition: all 0.25s ease;
        }
        .feature-pill:hover {
            border-color: var(--accent-indigo);
            color: var(--accent-purple);
            background: rgba(99,102,241,0.06);
        }
        .feature-pill .dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--accent-indigo);
        }

        /* ===== ANIMATED STATS ROW ===== */
        .stats-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            max-width: 820px;
            margin: 0 auto;
            animation: fadeInUp 1.1s ease;
        }
        .stat-item {
            text-align: center;
            padding: 20px 12px;
            background: rgba(17,24,39,0.5);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            transition: all 0.3s ease;
        }
        .stat-item:hover {
            border-color: rgba(99,102,241,0.3);
            transform: translateY(-2px);
        }
        .stat-value {
            font-size: 28px;
            font-weight: 800;
            background: linear-gradient(135deg, var(--accent-indigo), var(--accent-violet));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .stat-label {
            color: var(--text-muted);
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-top: 4px;
        }

        /* ===== ANIMATIONS ===== */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }

        /* ===== ANALYSIS RUNNING OVERLAY ===== */
        .analysis-status {
            background: linear-gradient(145deg, rgba(17,24,39,0.9), rgba(15,23,42,0.95));
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            max-width: 500px;
            margin: 60px auto;
        }
        .analysis-status h3 {
            color: var(--text-primary);
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 12px;
        }
        .analysis-status p {
            color: var(--text-secondary);
            font-size: 14px;
        }

        /* ===== RESULTS HEADER ===== */
        .results-header {
            background: linear-gradient(145deg, rgba(17,24,39,0.7), rgba(15,23,42,0.85));
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 24px 32px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            animation: fadeInUp 0.5s ease;
        }
        .results-header-title {
            font-size: 22px;
            font-weight: 800;
            background: linear-gradient(135deg, #f1f5f9, #cbd5e1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .results-header-sub {
            color: var(--text-muted);
            font-size: 13px;
            font-weight: 500;
            margin-top: 2px;
        }

        /* ===== SCORE RING ===== */
        .score-ring-container {
            text-align: center;
            padding: 20px;
            animation: fadeInUp 0.6s ease;
        }

        /* ===== PROGRESS BAR ===== */
        .custom-progress {
            width: 100%;
            height: 8px;
            background: rgba(17,24,39,0.8);
            border-radius: 4px;
            overflow: hidden;
            margin: 8px 0;
        }
        .custom-progress-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* ===== SKILL MATCH HEADER ===== */
        .skill-match-hero {
            background: linear-gradient(145deg, rgba(17,24,39,0.8), rgba(15,23,42,0.9));
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 36px 24px;
            text-align: center;
            position: relative;
            overflow: hidden;
            animation: fadeInUp 0.5s ease;
        }
        .skill-match-hero::after {
            content: '';
            position: absolute;
            top: -50%; left: -50%;
            width: 200%; height: 200%;
            background: radial-gradient(circle, rgba(99,102,241,0.05) 0%, transparent 70%);
            pointer-events: none;
        }
        .skill-match-hero h3 {
            color: var(--text-secondary);
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 10px;
        }
        .skill-match-hero .big-number {
            font-size: 56px;
            font-weight: 900;
            background: linear-gradient(135deg, var(--accent-indigo), var(--accent-violet));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1;
        }
        .skill-match-hero .sub-text {
            color: var(--text-muted);
            font-size: 13px;
            margin-top: 8px;
        }

        /* ===== SIDEBAR LOGO ===== */
        .sidebar-logo {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px;
        }
        .sidebar-logo-icon {
            width: 40px;
            height: 40px;
            border-radius: 12px;
            background: linear-gradient(135deg, var(--accent-indigo), var(--accent-violet));
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            box-shadow: 0 4px 12px rgba(99,102,241,0.3);
        }
        .sidebar-logo-text {
            color: var(--text-primary);
            font-size: 18px;
            font-weight: 800;
            letter-spacing: -0.3px;
        }
        .sidebar-logo-sub {
            color: var(--text-muted);
            font-size: 11px;
            font-weight: 500;
            letter-spacing: 0.5px;
        }

        /* ===== VERSION FOOTER ===== */
        .version-footer {
            text-align: center;
            padding: 16px 0;
            color: var(--text-muted);
            font-size: 10.5px;
            letter-spacing: 0.3px;
        }
        .version-footer a {
            color: var(--accent-indigo);
            text-decoration: none;
        }

        /* ===== RESPONSIVE ===== */
        @media (max-width: 768px) {
            .steps-grid {
                grid-template-columns: 1fr;
            }
            .stats-row {
                grid-template-columns: repeat(2, 1fr);
            }
            .hero-title {
                font-size: 36px;
            }
        }
    </style>
    """, unsafe_allow_html=True)


def render_metric_card(title, value, suffix=""):
    """Render a styled metric card."""
    st.markdown(f"""<div class="metric-card">
<h3>{title}</h3>
<p class="value">{value}{suffix}</p>
</div>""", unsafe_allow_html=True)


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

    # -- results header --
    candidate_name = results["contact_info"].get("name", "Candidate")
    if not candidate_name:
        candidate_name = "Candidate"
    score = results["composite"]["overall_score"]

    rec_class = 'rec-strong' if 'Strong' in results['match_label'] else 'rec-moderate' if 'Moderate' in results['match_label'] else 'rec-weak'
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown(f"""<div class="results-header-title">Analysis Results — {candidate_name}</div>
<div class="results-header-sub">Composite score computed from {len(results['features'])} features</div>""", unsafe_allow_html=True)
    with header_col2:
        st.markdown(f'<div style="text-align:right;padding-top:8px;"><span class="recommendation {rec_class}">{results["match_label"]}</span></div>', unsafe_allow_html=True)

    # -- match score gauge and recommendation --
    col1, col2 = st.columns([2, 1])

    with col1:
        fig = create_match_gauge(results["composite"]["overall_score"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)

        # Model info
        st.markdown(f"""<div class="info-card" style="text-align: center;">
<p class="label">Classified By</p>
<p class="content" style="font-weight: 700; color: var(--accent-purple);">{results['prediction']['model_used']}</p>
</div>""", unsafe_allow_html=True)

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
            # color based on value
            if value >= 70:
                bar_color = "var(--success)"
            elif value >= 45:
                bar_color = "var(--warning)"
            else:
                bar_color = "var(--danger)"

            st.markdown(f"""<div class="metric-card">
<h3>{name}</h3>
<p class="value">{value:.1f}%</p>
<span class="custom-progress" style="display:block;"><span class="custom-progress-fill" style="display:block;width:{value}%;background:{bar_color};"></span></span>
</div>""", unsafe_allow_html=True)

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
    matched_count = len(skill_match['matched'])
    total_required = matched_count + len(skill_match['missing'])

    st.markdown(f"""<div class="skill-match-hero">
<h3>Skill Match Rate</h3>
<p class="big-number">{pct:.1f}%</p>
<p class="sub-text">{matched_count} of {total_required} required skills found</p>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

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
        st.markdown(f"""<div class="info-card">
<p class="label">Name</p>
<p class="content">{name}</p>
</div>""", unsafe_allow_html=True)
    with col2:
        email = contact["emails"][0] if contact["emails"] else "Not detected"
        st.markdown(f"""<div class="info-card">
<p class="label">Email</p>
<p class="content">{email}</p>
</div>""", unsafe_allow_html=True)
    with col3:
        phone = contact["phones"][0] if contact["phones"] else "Not detected"
        st.markdown(f"""<div class="info-card">
<p class="label">Phone</p>
<p class="content">{phone}</p>
</div>""", unsafe_allow_html=True)

    # -- education --
    st.markdown('<div class="section-header">Education</div>', unsafe_allow_html=True)
    education = results["education"]
    if education:
        for edu in education:
            level_labels = {1: "High School", 2: "Diploma", 3: "Undergraduate",
                          4: "Postgraduate", 5: "Doctorate"}
            level_text = level_labels.get(edu["level"], "")
            st.markdown(f"""<div class="info-card">
<p class="label">{level_text} (Level {edu['level']})</p>
<p class="content">{edu['degree']}</p>
</div>""", unsafe_allow_html=True)
    else:
        st.info("No education qualifications detected in the resume.")

    # -- experience --
    st.markdown('<div class="section-header">Experience</div>', unsafe_allow_html=True)
    exp = results["experience_years"]
    st.markdown(f"""<div class="info-card">
<p class="label">Estimated Experience</p>
<p class="content">{exp} year{'s' if exp != 1 else ''}</p>
</div>""", unsafe_allow_html=True)

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
    pred_label = pred['prediction']
    pred_class = "rec-strong" if "Strong" in pred_label else "rec-moderate" if "Moderate" in pred_label else "rec-weak"

    st.markdown(f"""<div class="info-card" style="text-align: center; padding: 30px;">
<p class="label">Predicted Category</p>
<p style="margin: 12px 0;"><span class="recommendation {pred_class}">{pred_label}</span></p>
<p class="label" style="margin-top: 16px;">Model Used</p>
<p class="content" style="color: var(--accent-purple); font-weight: 600;">{pred['model_used']}</p>
</div>""", unsafe_allow_html=True)

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
        st.markdown("""<div class="sidebar-logo">
<span class="sidebar-logo-icon" style="font-weight: 700; color: #fff; font-size: 18px;">R</span>
<span><span class="sidebar-logo-text">ResumeAI</span><br><span class="sidebar-logo-sub">Smart Screening Engine</span></span>
</div>
<hr style="border-color: rgba(99,102,241,0.15); margin: 16px 0;">""", unsafe_allow_html=True)

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
        <hr style="border-color: rgba(99,102,241,0.15); margin: 24px 0 12px 0;">
        <div class="version-footer">
            Built with Python · spaCy · Scikit-learn · Streamlit<br>
            <span style="color: var(--accent-indigo);">v2.0.0</span>
        </div>
        """, unsafe_allow_html=True)

    # -- main content area --

    # show welcome screen if no analysis has been done yet
    if "analysis_results" not in st.session_state and not analyze_clicked:
        # Hero header section
        st.markdown("""<div class="hero-container">
<p class="hero-badge">AI-POWERED RESUME ANALYSIS</p>
<h1 class="hero-title">Resume Screening System</h1>
<p class="hero-subtitle">Upload a resume and provide a job description to get an instant match analysis with skill gap identification, NLP-powered scoring, and ML classification.</p>
</div>""", unsafe_allow_html=True)

        # Step cards - use streamlit columns for reliable rendering
        cols = st.columns(3)
        with cols[0]:
            st.markdown("""<div class="step-card">
<p class="step-num step-num-1">1</p>
<p class="step-title">Upload Resume</p>
<p class="step-desc">Drop a PDF resume file in the sidebar to extract text and skills automatically</p>
</div>""", unsafe_allow_html=True)
        with cols[1]:
            st.markdown("""<div class="step-card">
<p class="step-num step-num-2">2</p>
<p class="step-title">Add Job Description</p>
<p class="step-desc">Paste the target job description or use our sample JD to test the system</p>
</div>""", unsafe_allow_html=True)
        with cols[2]:
            st.markdown("""<div class="step-card">
<p class="step-num step-num-3">3</p>
<p class="step-title">Get Analysis</p>
<p class="step-desc">Receive a detailed match report with scores, skill gaps, and recommendations</p>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Feature pills row
        st.markdown("""<div class="feature-row">
<span class="feature-pill"><span class="dot"></span>TF-IDF Similarity</span>
<span class="feature-pill"><span class="dot"></span>Skill Gap Analysis</span>
<span class="feature-pill"><span class="dot"></span>ML Classification</span>
<span class="feature-pill"><span class="dot"></span>Education Detection</span>
<span class="feature-pill"><span class="dot"></span>Experience Parsing</span>
<span class="feature-pill"><span class="dot"></span>Contact Extraction</span>
</div>""", unsafe_allow_html=True)

        # Stats row - use streamlit columns
        st.markdown("<br>", unsafe_allow_html=True)
        stat_cols = st.columns(4)
        stats_data = [
            ("150+", "Skills Tracked"),
            ("7", "Skill Categories"),
            ("5", "ML Features"),
            ("91%", "Model Accuracy"),
        ]
        for col, (value, label) in zip(stat_cols, stats_data):
            with col:
                st.markdown(f"""<div class="stat-item">
<p class="stat-value">{value}</p>
<p class="stat-label">{label}</p>
</div>""", unsafe_allow_html=True)

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

        # create tabs
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

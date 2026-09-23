"""
combine_reports.py
------------------
Combines the two academic project reports for Suyash Vakhariya:
1. Project 1: Automated Resume Screening and Skill Matching System
2. Project 2: Movie Recommendation Engine

Produces:
- `report/Combined_Projects_Report.pdf` (5 pages: Project 1 [pages 1-2],
   dedicated Section Divider page [page 3], and Project 2 [pages 4-5]).
- `report/Combined_4Page_Report.pdf` (4 pages: direct sequential stitch).

Both PDFs feature:
- Clearly identifiable project labels and running footers on every page.
- Embedded PDF Bookmarks / Table of Contents (TOC) hierarchy.
- Zero emojis, professional academic formatting, and exact student credentials.
"""

import os
import fitz

def create_divider_page(doc, page_w, page_h):
    page = doc.new_page(width=page_w, height=page_h)
    margin_x = 44
    content_w = page_w - 2 * margin_x

    # Top Institutional Banner
    y = 50
    header_org = "ACADEMIC PROJECT REPORT PORTFOLIO"
    hw = fitz.get_text_length(header_org, fontname="hebo", fontsize=11)
    page.insert_text(((page_w - hw) / 2, y), header_org, fontsize=11, fontname="hebo", color=(0.25, 0.28, 0.35))
    y += 16

    sub_hdr = "Artificial Intelligence and Machine Learning  |  Suyash Vakhariya  |  Roll No: AIML A6 AUG 11681"
    shw = fitz.get_text_length(sub_hdr, fontname="helv", fontsize=9)
    page.insert_text(((page_w - shw) / 2, y), sub_hdr, fontsize=9, fontname="helv", color=(0.45, 0.48, 0.55))
    y += 18
    page.draw_line((margin_x + 20, y), (page_w - margin_x - 20, y), color=(0.80, 0.82, 0.88), width=0.8)
    y += 45

    # Center Section Divider Card
    card_h = 320
    card_rect = fitz.Rect(margin_x + 10, y, page_w - margin_x - 10, y + card_h)
    page.draw_rect(card_rect, color=(0.78, 0.80, 0.86), fill=(0.965, 0.972, 0.985), width=1.0)

    # Card Content
    cy = y + 26
    badge_text = "PART 2 OF 2"
    bw = fitz.get_text_length(badge_text, fontname="hebo", fontsize=10)
    page.insert_text(((page_w - bw) / 2, cy), badge_text, fontsize=10, fontname="hebo", color=(0.35, 0.40, 0.55))
    cy += 24

    title_text = "PROJECT 2: MOVIE RECOMMENDATION ENGINE"
    tw = fitz.get_text_length(title_text, fontname="hebo", fontsize=15)
    page.insert_text(((page_w - tw) / 2, cy), title_text, fontsize=15, fontname="hebo", color=(0.10, 0.12, 0.18))
    cy += 18

    subtitle = "A Hybrid Content-Based & Truncated SVD Collaborative Filtering Architecture"
    sw = fitz.get_text_length(subtitle, fontname="helv", fontsize=9.5)
    page.insert_text(((page_w - sw) / 2, cy), subtitle, fontsize=9.5, fontname="helv", color=(0.35, 0.38, 0.45))
    cy += 20
    page.draw_line((margin_x + 40, cy), (page_w - margin_x - 40, cy), color=(0.82, 0.84, 0.90), width=0.6)
    cy += 24

    # Technical Highlights Summary Box
    details = [
        ("Author & Roll No", "Suyash Vakhariya  |  AIML A6 AUG 11681"),
        ("Specialization", "Artificial Intelligence and Machine Learning"),
        ("Core Methodology", "Sublinear TF-IDF Vectorization + Truncated SVD Matrix Factorization (k = 20)"),
        ("Benchmark Dataset", "MovieLens 100k Benchmark (100,836 ratings across 9,742 movies by 610 users)"),
        ("Empirical Results", "Test RMSE: 2.1831 (target < 2.500), Precision@10: 0.274, Recall@10: 0.215, Diversity: 0.950"),
        ("Cloud Deployment", "Interactive Streamlit Engine + Live Vercel Production Environment"),
        ("Source Code Repo", "https://github.com/Izumi6/Movie-Recommendation-System")
    ]

    for label, val in details:
        page.insert_text((margin_x + 35, cy), f"{label}:", fontsize=8.2, fontname="hebo", color=(0.20, 0.22, 0.30))
        page.insert_text((margin_x + 145, cy), val, fontsize=8.2, fontname="helv", color=(0.28, 0.30, 0.38))
        cy += 16.5

    cy += 10
    notice = "The complete two-page academic report for Project 2 commences on the following page (Page 4)."
    nw = fitz.get_text_length(notice, fontname="helv", fontsize=8.4)
    page.insert_text(((page_w - nw) / 2, cy), notice, fontsize=8.4, fontname="helv", color=(0.35, 0.38, 0.48))

    # Bottom Document Portfolio Map
    y += card_h + 35
    map_title = "Document Portfolio Navigation Map:"
    page.insert_text((margin_x + 20, y), map_title, fontsize=9, fontname="hebo", color=(0.30, 0.32, 0.40))
    y += 16
    map_lines = [
        "- Pages 1 - 2: Project 1 — Automated Resume Screening and Skill Matching System",
        "- Page 3:      Section Divider & Transition Overview (Current Page)",
        "- Pages 4 - 5: Project 2 — Movie Recommendation Engine (Empirical Evaluation & Analytics)"
    ]
    for ml in map_lines:
        page.insert_text((margin_x + 30, y), ml, fontsize=8.2, fontname="cour", color=(0.35, 0.38, 0.45))
        y += 14

    return page


def build_combined_reports():
    pdf_resume = "report/Resume_Screening_System_Report.pdf"
    pdf_movie = "report/Movie_Recommendation_System_Report.pdf"

    if not os.path.exists(pdf_resume):
        raise FileNotFoundError(f"Missing {pdf_resume}")
    if not os.path.exists(pdf_movie):
        raise FileNotFoundError(f"Missing {pdf_movie}")

    doc_resume = fitz.open(pdf_resume)
    doc_movie = fitz.open(pdf_movie)

    page_w, page_h = 595.28, 841.89
    margin_x = 44

    # ================= 1. BUILD 5-PAGE PORTFOLIO WITH DIVIDER =================
    out_5page = fitz.open()

    # Pages 1 & 2: Resume Screening System
    out_5page.insert_pdf(doc_resume)

    # Page 3: Section Divider
    create_divider_page(out_5page, page_w, page_h)

    # Pages 4 & 5: Movie Recommendation Engine
    out_5page.insert_pdf(doc_movie)

    # Add Running Footer to all 5 pages
    footer_labels = [
        ("PROJECT 1: RESUME SCREENING SYSTEM", "Page 1 of 5"),
        ("PROJECT 1: RESUME SCREENING SYSTEM", "Page 2 of 5"),
        ("PORTFOLIO SECTION DIVIDER", "Page 3 of 5"),
        ("PROJECT 2: MOVIE RECOMMENDATION ENGINE", "Page 4 of 5"),
        ("PROJECT 2: MOVIE RECOMMENDATION ENGINE", "Page 5 of 5"),
    ]

    for p_idx, page in enumerate(out_5page):
        p_label, p_num = footer_labels[p_idx]
        fy = page_h - 22

        # Subtle separator line
        page.draw_line((margin_x, fy - 7), (page_w - margin_x, fy - 7), color=(0.85, 0.86, 0.90), width=0.5)

        # Left label: Project identification
        page.insert_text((margin_x, fy), p_label, fontsize=6.8, fontname="hebo", color=(0.40, 0.43, 0.50))

        # Center label: Student attribution
        mid_text = "Suyash Vakhariya (AIML A6 AUG 11681)"
        mw = fitz.get_text_length(mid_text, fontname="helv", fontsize=6.8)
        page.insert_text(((page_w - mw) / 2, fy), mid_text, fontsize=6.8, fontname="helv", color=(0.50, 0.53, 0.60))

        # Right label: Page number
        rw = fitz.get_text_length(p_num, fontname="hebo", fontsize=6.8)
        page.insert_text((page_w - margin_x - rw, fy), p_num, fontsize=6.8, fontname="hebo", color=(0.40, 0.43, 0.50))

    # Set PDF Bookmarks / Outline Hierarchy
    toc_5page = [
        [1, "Project 1: Automated Resume Screening System", 1],
        [2, "Page 1: Problem Formulation & Benchmark Tables", 1],
        [2, "Page 2: Screening Analytics Dashboard & Evaluation", 2],
        [1, "Section Divider: Transition to Project 2", 3],
        [1, "Project 2: Movie Recommendation Engine", 4],
        [2, "Page 1: Architecture & Collaborative Filtering", 4],
        [2, "Page 2: CineMatch Analytics Dashboard & Discussion", 5],
    ]
    out_5page.set_toc(toc_5page)

    combined_5p_path = "report/Combined_Projects_Report.pdf"
    out_5page.save(combined_5p_path)
    out_5page.close()
    print(f"Compiled 5-page report with section divider to {combined_5p_path}")

    # ================= 2. BUILD 4-PAGE DIRECT STITCH =================
    out_4page = fitz.open()
    out_4page.insert_pdf(doc_resume)
    out_4page.insert_pdf(doc_movie)

    footer_4p_labels = [
        ("PROJECT 1: RESUME SCREENING SYSTEM", "Page 1 of 4"),
        ("PROJECT 1: RESUME SCREENING SYSTEM", "Page 2 of 4"),
        ("PROJECT 2: MOVIE RECOMMENDATION ENGINE", "Page 3 of 4"),
        ("PROJECT 2: MOVIE RECOMMENDATION ENGINE", "Page 4 of 4"),
    ]

    for p_idx, page in enumerate(out_4page):
        p_label, p_num = footer_4p_labels[p_idx]
        fy = page_h - 22
        page.draw_line((margin_x, fy - 7), (page_w - margin_x, fy - 7), color=(0.85, 0.86, 0.90), width=0.5)
        page.insert_text((margin_x, fy), p_label, fontsize=6.8, fontname="hebo", color=(0.40, 0.43, 0.50))

        mid_text = "Suyash Vakhariya (AIML A6 AUG 11681)"
        mw = fitz.get_text_length(mid_text, fontname="helv", fontsize=6.8)
        page.insert_text(((page_w - mw) / 2, fy), mid_text, fontsize=6.8, fontname="helv", color=(0.50, 0.53, 0.60))

        rw = fitz.get_text_length(p_num, fontname="hebo", fontsize=6.8)
        page.insert_text((page_w - margin_x - rw, fy), p_num, fontsize=6.8, fontname="hebo", color=(0.40, 0.43, 0.50))

    toc_4page = [
        [1, "Project 1: Automated Resume Screening System", 1],
        [2, "Page 1: Problem Formulation & Benchmark Tables", 1],
        [2, "Page 2: Screening Analytics Dashboard & Evaluation", 2],
        [1, "Project 2: Movie Recommendation Engine", 3],
        [2, "Page 1: Architecture & Collaborative Filtering", 3],
        [2, "Page 2: CineMatch Analytics Dashboard & Discussion", 4],
    ]
    out_4page.set_toc(toc_4page)

    combined_4p_path = "report/Combined_4Page_Report.pdf"
    out_4page.save(combined_4p_path)
    out_4page.close()
    print(f"Compiled 4-page direct stitch report to {combined_4p_path}")

    doc_resume.close()
    doc_movie.close()

if __name__ == "__main__":
    build_combined_reports()

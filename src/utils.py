"""
utils.py
--------
Shared utility functions used across the project.
Nothing fancy, just common helpers to avoid repeating code.
"""

import os
import json
import logging


def setup_logging(level=logging.INFO):
    """Set up basic logging configuration for the project."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def validate_pdf(uploaded_file):
    """
    Check if the uploaded file is actually a PDF.
    Returns (is_valid, error_message).
    """
    if uploaded_file is None:
        return False, "No file uploaded."
    
    # check file extension
    name = getattr(uploaded_file, 'name', '')
    if not name.lower().endswith('.pdf'):
        return False, f"Expected a PDF file, got '{name}'."
    
    # check file size (limit to 10MB)
    size = getattr(uploaded_file, 'size', 0)
    if size > 10 * 1024 * 1024:
        return False, "File too large. Maximum size is 10MB."
    
    if size == 0:
        return False, "File is empty."
    
    return True, ""


def load_json(filepath):
    """Load a JSON file and return the data. Returns empty dict on error."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Failed to load JSON from {filepath}: {e}")
        return {}


def save_json(data, filepath):
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def truncate_text(text, max_length=500):
    """Truncate text to max_length characters, adding ellipsis if needed."""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(' ', 1)[0] + "..."


def get_match_category(score):
    """
    Determine match category based on score.
    Returns (label, color) tuple.
    """
    from config import MATCH_CATEGORIES
    
    for key in ["strong", "moderate", "weak"]:
        cat = MATCH_CATEGORIES[key]
        if score >= cat["min_score"]:
            return cat["label"], cat["color"]
    
    return "Weak Match", "#ef4444"

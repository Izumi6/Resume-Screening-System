"""
text_preprocessor.py
--------------------
NLP preprocessing pipeline for cleaning and normalizing resume text.

The idea is to take the raw messy text we get from PDF extraction and
turn it into something clean enough for TF-IDF and skill matching.
We use NLTK for tokenization, lemmatization, and stop word removal,
plus regex patterns to pull out contact details before we strip them.
"""

import re
import logging

logger = logging.getLogger(__name__)

# pre-compile regex patterns so we're not recompiling every call
EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
PHONE_PATTERN = re.compile(r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\./0-9]{7,15}')
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
SPECIAL_CHARS = re.compile(r'[^a-zA-Z0-9\s\.\,\-\/\+\#]')
EXTRA_WHITESPACE = re.compile(r'\s+')

# Standard English stop words fallback in case NLTK corpus isn't cached
DEFAULT_STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'can', 'shall',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'as', 'into', 'through', 'during', 'before', 'after', 'and',
    'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
    'neither', 'each', 'every', 'all', 'any', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same',
    'than', 'too', 'very', 'just', 'because', 'if', 'then',
    'this', 'that', 'these', 'those', 'i', 'me', 'my', 'we',
    'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
    'it', 'its', 'they', 'them', 'their', 'what', 'which',
    'who', 'whom', 'when', 'where', 'why', 'how',
}


class TextPreprocessor:
    """
    Handles all text cleaning and normalization steps.
    
    Uses NLTK under the hood for lemmatization and stop-word filtering.
    Falls back gracefully to built-in rules if external resources are unavailable.
    """
    
    def __init__(self):
        self.lemmatizer = None
        self.stop_words = set(DEFAULT_STOP_WORDS)
        self._init_nlp()
    
    def _init_nlp(self):
        """Initialize NLTK lemmatizer and stop words."""
        try:
            import nltk
            from nltk.stem import WordNetLemmatizer
            from nltk.corpus import stopwords
            
            try:
                self.stop_words = set(stopwords.words('english'))
            except Exception:
                pass
            
            try:
                self.lemmatizer = WordNetLemmatizer()
                # test lemmatizer
                self.lemmatizer.lemmatize("running", pos="v")
            except Exception:
                self.lemmatizer = None
        except ImportError:
            logger.warning("NLTK not found. Falling back to built-in stop words and rules.")
            self.lemmatizer = None
    
    def preprocess(self, text):
        """
        Run the full preprocessing pipeline on the given text.
        
        Steps:
        1. Convert to lowercase
        2. Remove URLs, emails, phone numbers
        3. Remove special characters
        4. Tokenize and lemmatize
        5. Remove stop words
        6. Clean up whitespace
        
        Returns the cleaned text as a single string.
        """
        if not text or not text.strip():
            return ""
        
        # lowercase everything
        text = text.lower()
        
        # strip out URLs and contact info (we extract these separately)
        text = URL_PATTERN.sub(' ', text)
        text = EMAIL_PATTERN.sub(' ', text)
        text = PHONE_PATTERN.sub(' ', text)
        
        # remove special characters but keep basics like periods, commas, +, #
        text = SPECIAL_CHARS.sub(' ', text)
        
        # tokenize and lemmatize
        words = text.split()
        tokens = []
        for w in words:
            # strip surrounding punctuation like commas and periods
            clean_w = w.strip('.,;:')
            if len(clean_w) >= 2 and clean_w not in self.stop_words:
                if self.lemmatizer:
                    # apply verb and noun lemmatization
                    lemma = self.lemmatizer.lemmatize(clean_w, pos='v')
                    lemma = self.lemmatizer.lemmatize(lemma, pos='n')
                    tokens.append(lemma)
                else:
                    tokens.append(clean_w)
        
        # collapse multiple spaces
        cleaned = ' '.join(tokens)
        cleaned = EXTRA_WHITESPACE.sub(' ', cleaned).strip()
        
        return cleaned
    
    def extract_contact_info(self, text):
        """
        Pull out contact details from the raw text.
        
        Returns a dict with 'emails', 'phones', and 'urls' lists.
        We run this on the ORIGINAL text before preprocessing,
        since preprocessing would strip these out.
        """
        info = {
            "emails": [],
            "phones": [],
            "urls": [],
            "name": "",
        }
        
        if not text:
            return info
        
        # find emails
        emails = EMAIL_PATTERN.findall(text)
        info["emails"] = list(set(emails))
        
        # find phone numbers
        phones = PHONE_PATTERN.findall(text)
        cleaned_phones = []
        for p in phones:
            p = p.strip()
            # only keep if it looks like a real phone number (7+ digits)
            digit_count = sum(1 for c in p if c.isdigit())
            if digit_count >= 7:
                cleaned_phones.append(p)
        info["phones"] = list(set(cleaned_phones))
        
        # find URLs
        urls = URL_PATTERN.findall(text)
        info["urls"] = list(set(urls))
        
        # extract candidate name using header heuristics
        info["name"] = self._extract_name(text)
        
        return info
    
    def _extract_name(self, text):
        """
        Try to figure out the candidate's name from the resume text.
        Resumes typically have the candidate name on the first non-empty line.
        """
        ignore_keywords = {
            'resume', 'curriculum', 'vitae', 'cv', 'profile', 'contact',
            'summary', 'experience', 'education', 'skills', 'objective',
            'phone', 'email', 'address', 'page', 'portfolio'
        }
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines[:6]:
            words = line.split()
            # Most human names on resumes are 2 to 4 words and don't contain common section titles
            if 2 <= len(words) <= 4:
                lower_line = line.lower()
                if not any(kw in lower_line for kw in ignore_keywords):
                    if not any(c in line for c in ['@', 'http', 'www', '/', '\\']):
                        return line
        
        # fallback: first short line if available
        if lines and len(lines[0].split()) <= 4:
            return lines[0]
        
        return ""

"""
Tests for the text preprocessing module.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from src.text_preprocessor import TextPreprocessor


class TestTextPreprocessor(unittest.TestCase):
    
    def setUp(self):
        self.preprocessor = TextPreprocessor()
    
    def test_empty_text(self):
        result = self.preprocessor.preprocess("")
        self.assertEqual(result, "")
    
    def test_none_text(self):
        result = self.preprocessor.preprocess(None)
        self.assertEqual(result, "")
    
    def test_whitespace_only(self):
        result = self.preprocessor.preprocess("   \n\t  ")
        self.assertEqual(result, "")
    
    def test_basic_preprocessing(self):
        text = "I have experience with Python and Machine Learning."
        result = self.preprocessor.preprocess(text)
        # should be lowercased and cleaned
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
    
    def test_url_removal(self):
        text = "Check my portfolio at https://github.com/johndoe and my work"
        result = self.preprocessor.preprocess(text)
        self.assertNotIn("https://", result)
        self.assertNotIn("github.com", result)
    
    def test_email_removal(self):
        text = "Contact me at john.doe@example.com for more info"
        result = self.preprocessor.preprocess(text)
        self.assertNotIn("@", result)
    
    def test_contact_extraction_email(self):
        text = "Name: John Doe\nEmail: john@example.com\nPhone: +91 9876543210"
        info = self.preprocessor.extract_contact_info(text)
        self.assertIn("john@example.com", info["emails"])
    
    def test_contact_extraction_phone(self):
        text = "Phone: +91 9876543210"
        info = self.preprocessor.extract_contact_info(text)
        self.assertTrue(len(info["phones"]) > 0)
    
    def test_contact_extraction_empty(self):
        info = self.preprocessor.extract_contact_info("")
        self.assertEqual(info["emails"], [])
        self.assertEqual(info["phones"], [])
    
    def test_special_chars_cleaned(self):
        text = "Skills: Python!!! Java@@@ C++### data$$$"
        result = self.preprocessor.preprocess(text)
        self.assertNotIn("!!!", result)
        self.assertNotIn("@@@", result)
        self.assertNotIn("$$$", result)
    
    def test_preserves_meaningful_content(self):
        text = "Experienced software developer with Python and Django knowledge"
        result = self.preprocessor.preprocess(text)
        # should still contain some meaningful words (lemmatized)
        self.assertTrue(len(result.split()) >= 2)


if __name__ == "__main__":
    unittest.main()

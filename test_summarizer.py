"""
Unit tests for the NLP Document Summarizer.
Tests the preprocessing, chunking, summarization, and API endpoints.
"""

import os
import json
import pytest
import tempfile

from preprocessor import TextPreprocessor
from config import CONFIG


class TestTextPreprocessor:
    """Tests for the text preprocessing utilities."""

    def setup_method(self):
        self.preprocessor = TextPreprocessor()

    def test_clean_basic_text(self):
        text = "  Hello   world.  This  is   a   test.  "
        cleaned = self.preprocessor.clean(text)
        assert cleaned == "Hello world. This is a test."

    def test_clean_html_tags(self):
        text = "<p>Hello <strong>world</strong></p>"
        cleaned = self.preprocessor.clean(text)
        assert "<" not in cleaned
        assert "Hello" in cleaned
        assert "world" in cleaned

    def test_clean_html_entities(self):
        text = "Hello &amp; world &lt;test&gt;"
        cleaned = self.preprocessor.clean(text)
        assert "&amp;" not in cleaned
        assert "&lt;" not in cleaned

    def test_clean_urls(self):
        text = "Visit https://example.com for more info."
        cleaned = self.preprocessor.clean(text)
        assert "https://example.com" not in cleaned
        assert "Visit" in cleaned

    def test_clean_empty_input(self):
        assert self.preprocessor.clean("") == ""
        assert self.preprocessor.clean("   ") == ""
        assert self.preprocessor.clean(None) == ""

    def test_split_sentences(self):
        text = "This is the first sentence. This is the second. And here is the third one."
        sentences = self.preprocessor.split_sentences(text)
        assert len(sentences) == 3

    def test_split_sentences_filters_short(self):
        text = "Good. OK. This is a longer sentence that should be kept."
        sentences = self.preprocessor.split_sentences(text)
        # Short sentences like "Good." and "OK." should be filtered
        assert len(sentences) == 1

    def test_count_tokens(self):
        text = "Hello world, this is a test sentence."
        count = self.preprocessor.count_tokens(text)
        assert count > 0
        assert isinstance(count, int)

    def test_count_tokens_empty(self):
        assert self.preprocessor.count_tokens("") == 0
        assert self.preprocessor.count_tokens(None) == 0

    def test_word_count(self):
        text = "one two three four five"
        assert self.preprocessor.get_word_count(text) == 5

    def test_extract_key_sentences(self):
        text = (
            "Machine learning is a branch of artificial intelligence. "
            "It allows computers to learn from data. "
            "Deep learning is a subset of machine learning. "
            "Neural networks are used in deep learning. "
            "Data is essential for training machine learning models. "
            "Python is a popular language for machine learning. "
            "TensorFlow and PyTorch are common frameworks. "
            "Machine learning has many real-world applications."
        )
        key_sentences = self.preprocessor.extract_key_sentences(text, num_sentences=3)
        assert len(key_sentences) == 3
        # All returned items should be from the original text
        for sent in key_sentences:
            assert sent in text

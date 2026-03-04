"""
Text preprocessing utilities for the document summarizer.
Handles text cleaning, sentence splitting, and token counting.
"""

import re
import html
import nltk
from transformers import BartTokenizer

from config import CONFIG

# Download NLTK data for sentence tokenization
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


class TextPreprocessor:
    """Cleans and prepares text for summarization."""

    def __init__(self):
        self.tokenizer = BartTokenizer.from_pretrained(CONFIG["model_name"])

    def clean(self, text):
        """
        Clean raw text for summarization.
        Handles HTML, extra whitespace, encoding issues, etc.
        """
        if not text or not text.strip():
            return ""

        # Decode HTML entities
        text = html.unescape(text)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+\.\S+", "", text)

        # Normalize unicode characters
        text = text.encode("ascii", "ignore").decode("ascii")

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def split_sentences(self, text):
        """Split text into sentences using NLTK."""
        if not text or not text.strip():
            return []

        sentences = nltk.sent_tokenize(text)

        # Filter out very short sentences (likely artifacts)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        return sentences

    def count_tokens(self, text):
        """Count the number of BART tokens in a text."""
        if not text:
            return 0
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)

    def truncate_to_tokens(self, text, max_tokens):
        """Truncate text to a maximum number of tokens."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)

    def get_word_count(self, text):
        """Simple word count."""
        if not text:
            return 0
        return len(text.split())

    def extract_key_sentences(self, text, num_sentences=5):
        """
        Extract the most important sentences based on word frequency.
        This is a simple extractive approach used as a fallback.
        """
        sentences = self.split_sentences(text)
        if len(sentences) <= num_sentences:
            return sentences

        # Calculate word frequencies
        words = text.lower().split()
        word_freq = {}
        for word in words:
            word = re.sub(r"[^\w]", "", word)
            if word and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Score sentences by sum of word frequencies
        sentence_scores = []
        for i, sent in enumerate(sentences):
            score = 0
            sent_words = sent.lower().split()
            for word in sent_words:
                word = re.sub(r"[^\w]", "", word)
                score += word_freq.get(word, 0)
            # Normalize by sentence length to avoid bias toward long sentences
            if len(sent_words) > 0:
                score = score / len(sent_words)
            # Slight boost for sentences near the beginning (they're usually important)
            position_boost = 1.0 + (0.1 * max(0, 5 - i))
            score *= position_boost
            sentence_scores.append((i, score, sent))

        # Sort by score and return top sentences in original order
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = sorted(sentence_scores[:num_sentences], key=lambda x: x[0])

        return [sent for _, _, sent in top_sentences]

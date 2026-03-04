"""
NLP Document Summarizer - Core Engine
Handles text summarization using BART with intelligent chunking
for long documents.
"""

import time
import torch
from transformers import BartForConditionalGeneration, BartTokenizer

from preprocessor import TextPreprocessor
from config import CONFIG


class DocumentSummarizer:
    """
    Summarizes documents using facebook/bart-large-cnn.
    Handles long documents by splitting them into overlapping chunks
    and recursively summarizing until the output fits the target length.
    """

    def __init__(self, model_name=None):
        self.model_name = model_name or CONFIG["model_name"]
        self.preprocessor = TextPreprocessor()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading model: {self.model_name}")
        print(f"Using device: {self.device}")

        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully")

    def summarize(self, text, max_length=None, min_length=None,
                  num_beams=None, length_penalty=None):
        """
        Summarize a document. Automatically handles chunking for long texts.

        Args:
            text: The document text to summarize
            max_length: Maximum tokens in the summary
            min_length: Minimum tokens in the summary
            num_beams: Beam search width (higher = better but slower)
            length_penalty: Controls summary length (higher = longer)

        Returns:
            dict with summary text, stats, and timing info
        """
        start_time = time.time()

        max_length = max_length or CONFIG["max_summary_length"]
        min_length = min_length or CONFIG["min_summary_length"]
        num_beams = num_beams or CONFIG["num_beams"]
        length_penalty = length_penalty or CONFIG.get("length_penalty", 2.0)

        # Preprocess
        cleaned_text = self.preprocessor.clean(text)
        original_word_count = len(cleaned_text.split())

        if not cleaned_text.strip():
            return {
                "summary": "",
                "original_words": 0,
                "summary_words": 0,
                "compression_ratio": 0,
                "processing_time": 0,
                "chunks_used": 0
            }

        # Check if we need to chunk
        token_count = self.preprocessor.count_tokens(cleaned_text)
        max_input = CONFIG["max_input_tokens"]

        if token_count <= max_input:
            summary = self._summarize_single(
                cleaned_text, max_length, min_length, num_beams, length_penalty
            )
            chunks_used = 1
        else:
            summary = self._summarize_long(
                cleaned_text, max_length, min_length, num_beams, length_penalty
            )
            chunks_used = len(self.chunk_text(cleaned_text))

        summary_word_count = len(summary.split())
        processing_time = time.time() - start_time

        compression_ratio = 1 - (summary_word_count / original_word_count) if original_word_count > 0 else 0

        return {
            "summary": summary,
            "original_words": original_word_count,
            "summary_words": summary_word_count,
            "compression_ratio": round(compression_ratio, 4),
            "processing_time": round(processing_time, 2),
            "chunks_used": chunks_used
        }

    def _summarize_single(self, text, max_length, min_length,
                          num_beams, length_penalty):
        """Summarize a single chunk of text that fits in the model's context."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=CONFIG["max_input_tokens"],
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=True,
                no_repeat_ngram_size=3
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()

    def _summarize_long(self, text, max_length, min_length,
                        num_beams, length_penalty, depth=0):
        """
        Summarize a long document using recursive chunk summarization.
        Splits text into overlapping chunks, summarizes each, then
        combines and re-summarizes if needed.
        """
        max_depth = CONFIG.get("max_recursion_depth", 3)

        if depth > max_depth:
            # Safety limit - just truncate and summarize
            return self._summarize_single(
                text, max_length, min_length, num_beams, length_penalty
            )

        chunks = self.chunk_text(text)
        print(f"  Depth {depth}: Split into {len(chunks)} chunks")

        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            chunk_max = max(max_length // 2, 60)
            chunk_min = min(min_length, chunk_max - 10)
            summary = self._summarize_single(
                chunk, chunk_max, chunk_min, num_beams, length_penalty
            )
            chunk_summaries.append(summary)
            print(f"    Chunk {i + 1}/{len(chunks)} summarized ({len(summary.split())} words)")

        combined = " ".join(chunk_summaries)
        combined_tokens = self.preprocessor.count_tokens(combined)

        if combined_tokens <= CONFIG["max_input_tokens"]:
            return self._summarize_single(
                combined, max_length, min_length, num_beams, length_penalty
            )
        else:
            return self._summarize_long(
                combined, max_length, min_length,
                num_beams, length_penalty, depth + 1
            )

    def chunk_text(self, text, max_tokens=None, overlap_tokens=None):
        """
        Split text into overlapping chunks based on token count.
        Splits on sentence boundaries to avoid cutting mid-sentence.
        """
        max_tokens = max_tokens or CONFIG.get("chunk_max_tokens", 900)
        overlap_tokens = overlap_tokens or CONFIG["chunk_overlap"]

        sentences = self.preprocessor.split_sentences(text)

        if not sentences:
            return [text] if text.strip() else []

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.preprocessor.count_tokens(sentence)

            # If a single sentence exceeds max_tokens, add it as its own chunk
            if sentence_tokens > max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                chunks.append(sentence)
                current_chunk = []
                current_tokens = 0
                continue

            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))

                # Keep last few sentences for overlap
                overlap_sents = []
                overlap_count = 0
                for s in reversed(current_chunk):
                    s_tokens = self.preprocessor.count_tokens(s)
                    if overlap_count + s_tokens > overlap_tokens:
                        break
                    overlap_sents.insert(0, s)
                    overlap_count += s_tokens

                current_chunk = overlap_sents
                current_tokens = overlap_count

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def summarize_file(self, file_path, **kwargs):
        """Read and summarize a text file."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        result = self.summarize(text, **kwargs)
        result["file"] = file_path
        return result

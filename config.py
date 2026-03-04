"""
Configuration settings for the NLP Document Summarizer.
"""

CONFIG = {
    # Model
    "model_name": "facebook/bart-large-cnn",

    # Tokenization
    "max_input_tokens": 1024,

    # Chunking
    "chunk_max_tokens": 900,
    "chunk_overlap": 100,
    "max_recursion_depth": 3,

    # Summary defaults
    "max_summary_length": 150,
    "min_summary_length": 40,
    "num_beams": 4,
    "length_penalty": 2.0,

    # Batch processing
    "batch_size": 8,

    # API
    "api_host": "0.0.0.0",
    "api_port": 5000,
}

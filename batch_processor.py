"""
NLP Document Summarizer - Batch Processing
Summarize multiple documents from a directory.
"""

import os
import csv
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from summarizer import DocumentSummarizer
from config import CONFIG


SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".text"}


def find_documents(input_dir):
    """Find all supported text files in a directory."""
    documents = []
    for root, dirs, files in os.walk(input_dir):
        for filename in sorted(files):
            ext = os.path.splitext(filename)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                documents.append(os.path.join(root, filename))
    return documents


def process_document(summarizer, file_path, output_dir, **kwargs):
    """Summarize a single document and save the result."""
    try:
        result = summarizer.summarize_file(file_path, **kwargs)

        # Create output filename
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_summary.txt")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Source: {file_path}\n")
            f.write(f"Original words: {result['original_words']}\n")
            f.write(f"Summary words: {result['summary_words']}\n")
            f.write(f"Compression: {result['compression_ratio']:.1%}\n")
            f.write(f"Processing time: {result['processing_time']}s\n")
            f.write(f"\n{'=' * 60}\n\n")
            f.write(result["summary"])

        result["output_path"] = output_path
        result["status"] = "success"
        return result

    except Exception as e:
        return {
            "file": file_path,
            "status": "error",
            "error": str(e)
        }


def save_report(results, output_dir):
    """Save a CSV report of all batch processing results."""
    report_path = os.path.join(output_dir, "summary_report.csv")

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "File", "Status", "Original Words", "Summary Words",
            "Compression Ratio", "Processing Time (s)", "Chunks Used"
        ])

        for result in results:
            if result["status"] == "success":
                writer.writerow([
                    os.path.basename(result.get("file", "")),
                    result["status"],
                    result.get("original_words", ""),
                    result.get("summary_words", ""),
                    f"{result.get('compression_ratio', 0):.1%}",
                    result.get("processing_time", ""),
                    result.get("chunks_used", "")
                ])
            else:
                writer.writerow([
                    os.path.basename(result.get("file", "")),
                    result["status"],
                    "", "", "", "",
                    result.get("error", "")
                ])

    return report_path

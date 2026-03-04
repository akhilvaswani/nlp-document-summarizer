# NLP Document Summarizer

This project is a document summarization tool I built using Hugging Face's transformers library. It takes long text documents and generates concise summaries using a pretrained BART model (facebook/bart-large-cnn). The tricky part with summarization is handling documents that are longer than the model's token limit, so I built a chunking system that splits long documents into overlapping segments, summarizes each one, and then combines them into a final coherent summary.

I also wrapped the whole thing in a Flask API so it can be used as a microservice, and added a batch processing mode for summarizing multiple documents at once.

## What's in This Repo

- `summarizer.py` - Core summarization engine with chunking and multi-pass summarization
- `api.py` - Flask REST API for the summarizer
- `batch_processor.py` - Process multiple documents from a directory
- `preprocessor.py` - Text cleaning and preprocessing utilities
- `config.py` - Configuration settings for model parameters and chunking
- `test_summarizer.py` - Unit tests for the summarization pipeline
- `sample_documents/` - Example documents for testing
  - `article_tech.txt` - Sample tech article
  - `article_science.txt` - Sample science article
  - `article_business.txt` - Sample business article
- `requirements.txt` - Python dependencies
- `.gitignore` - Standard Python ignores

## How I Built It

### Step 1 - Setting Up and Choosing a Model

I started by researching different summarization approaches. There are two main types - extractive (pulling key sentences directly from the text) and abstractive (generating new sentences that capture the meaning). I went with abstractive because it produces more natural-sounding summaries.

For the model, I chose BART (Bidirectional and Auto-Regressive Transformers) because it was specifically fine-tuned on the CNN/DailyMail dataset for summarization tasks. It consistently produces good results out of the box.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The first time you run the summarizer, it'll download the BART model (about 1.6GB). After that it gets cached locally.

### Step 2 - Building the Text Preprocessor

Before feeding text into the model, I built a preprocessing pipeline in `preprocessor.py` that handles:

- Removing extra whitespace and normalizing line breaks
- Stripping HTML tags if the input came from a web page
- Handling special characters and encoding issues
- Splitting text into sentences using NLTK's sentence tokenizer
- Counting tokens using the BART tokenizer to determine if chunking is needed

```python
from preprocessor import TextPreprocessor

preprocessor = TextPreprocessor()
cleaned_text = preprocessor.clean(raw_text)
sentences = preprocessor.split_sentences(cleaned_text)
token_count = preprocessor.count_tokens(cleaned_text)
```

This step matters more than you'd think. I ran into cases where weird formatting or hidden characters were messing up the summaries, and the preprocessor fixed all of that.

### Step 3 - Handling Long Documents with Chunking

BART has a maximum input length of 1024 tokens, which is roughly 700-800 words. Most real documents are way longer than that, so I had to figure out how to handle them.

My approach in `summarizer.py`:

1. Count the tokens in the cleaned document
2. If it fits within the model's limit, summarize directly
3. If it's too long, split it into chunks with overlap
4. Summarize each chunk individually
5. If the combined chunk summaries are still too long, do another pass (recursive summarization)

The overlap between chunks is important because it helps maintain context across boundaries. Without it, you can lose information that falls right at the split point.

```python
def chunk_text(self, text, max_tokens=900, overlap_tokens=100):
    """Split text into overlapping chunks that fit the model's context window."""
    sentences = self.preprocessor.split_sentences(text)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = self.preprocessor.count_tokens(sentence)

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
```

### Step 4 - The Summarization Pipeline

The main `DocumentSummarizer` class ties everything together. It handles the full pipeline from raw text to final summary:

```python
from summarizer import DocumentSummarizer

summarizer = DocumentSummarizer()

# Quick summary
summary = summarizer.summarize("Your long document text here...")

# With custom parameters
summary = summarizer.summarize(
    text,
    max_length=200,      # max tokens in summary
    min_length=50,       # min tokens in summary
    num_beams=4,         # beam search width
    length_penalty=2.0   # higher = longer summaries
)
```

Under the hood, the summarizer:
1. Cleans and preprocesses the text
2. Checks if chunking is needed
3. Runs the BART model on each chunk (or the whole text if it fits)
4. Combines chunk summaries
5. Optionally runs a second summarization pass if the combined output is still long

I added beam search with 4 beams because it produces noticeably better results than greedy decoding. The length penalty parameter lets you control how long the summary should be relative to the input.

### Step 5 - Building the Flask API

I wanted this to be usable as a service, so I built a REST API with Flask in `api.py`:

```bash
python api.py
```

This starts the server on port 5000 with three endpoints:

**POST /summarize** - Summarize a single document
```bash
curl -X POST http://localhost:5000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document text...", "max_length": 150}'
```

**POST /summarize/file** - Upload a text file for summarization
```bash
curl -X POST http://localhost:5000/summarize/file \
  -F "file=@document.txt" \
  -F "max_length=200"
```

**GET /health** - Health check endpoint
```bash
curl http://localhost:5000/health
```

The API returns JSON responses with the summary, word count, compression ratio (how much shorter the summary is compared to the original), and processing time.

### Step 6 - Batch Processing

For processing multiple documents at once, I built `batch_processor.py`:

```bash
python batch_processor.py --input-dir ./documents --output-dir ./summaries
```

This scans the input directory for .txt, .md, and .html files, summarizes each one, and saves the results to the output directory. It also generates a `summary_report.csv` with stats like the original word count, summary word count, compression ratio, and processing time for each document.

I added a `--workers` flag for parallel processing, which helps a lot if you have a GPU and are processing many documents.

### Step 7 - Testing

I wrote unit tests in `test_summarizer.py` that cover:

- Basic summarization of short text
- Chunking behavior for long documents
- Preprocessing edge cases (HTML, special characters, empty input)
- API endpoint responses
- Batch processing with mixed file types

```bash
python -m pytest test_summarizer.py -v
```

## Example Output

Here's what the summarizer produces on a ~2000 word tech article about cloud computing:

**Input:** 2,147 words (article about cloud migration strategies)
**Output:** 89 words
**Compression ratio:** 96%
**Processing time:** 3.2 seconds

The summary accurately captured the three main migration strategies discussed in the article (rehosting, replatforming, refactoring), the key cost considerations, and the recommended approach for enterprises.

## Configuration

You can adjust the model and summarization parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | facebook/bart-large-cnn | Hugging Face model to use |
| `max_input_tokens` | 1024 | Maximum tokens per model input |
| `chunk_overlap` | 100 | Token overlap between chunks |
| `max_summary_length` | 150 | Default max summary tokens |
| `min_summary_length` | 40 | Default min summary tokens |
| `num_beams` | 4 | Beam search width |
| `batch_size` | 8 | Batch size for batch processing |

## What I Learned

The biggest challenge was getting the chunking right. My first attempt just split on sentence count, which sometimes produced chunks of wildly different sizes. Switching to token-based chunking with overlap made a huge difference in summary quality. I also learned that recursive summarization (summarizing the summaries when they're still too long) works surprisingly well and produces more coherent results than just concatenating chunk summaries. The Flask API was pretty straightforward, but it taught me the importance of proper error handling and input validation when building ML-powered services.

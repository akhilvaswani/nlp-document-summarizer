# NLP Document Summarizer

Takes long documents and spits out summaries using Hugging Face's BART model (facebook/bart-large-cnn). I wrapped it in a Flask API so you can hit it as a service.

The main challenge was handling documents longer than BART's 1024 token limit. My solution was to chunk the text into overlapping segments, summarize each chunk, then combine them. If the combined summaries are still too long, it does another pass -- kind of like summarizing the summary. Took some trial and error to get the chunking right but it works well now.

## Usage

```bash
pip install -r requirements.txt

# summarize from the command line
python summarizer.py --input document.txt

# start the API server
python api.py

# batch process a folder of documents
python batch_processor.py --input-dir ./documents --output-dir ./summaries
```

First run will download the BART model (~1.6GB), gets cached after that.

## API endpoints

- `POST /summarize` -- send JSON with a `text` field, get back a summary
- `POST /summarize/file` -- upload a text file directly
- `GET /health` -- health check

Returns the summary, word count, compression ratio, and processing time.

## How it handles long docs

Documents get split into chunks based on token count (not just sentence count -- I tried that first and the chunks came out wildly uneven). Each chunk overlaps with the next by about 100 tokens so you don't lose context at the boundaries. Each chunk gets summarized individually, then the summaries get combined. If the result is still over the token limit, it runs another summarization pass.

## Files

- `summarizer.py` -- core summarization engine with chunking logic
- `api.py` -- Flask REST API
- `batch_processor.py` -- processes a folder of docs at once
- `preprocessor.py` -- text cleaning (strips HTML, fixes encoding, etc.)
- `config.py` -- model parameters and defaults
- `sample_documents/` -- some example docs to test with

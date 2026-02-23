# mini-rag

This is a minimal implementation of the RAG model for question answering.

## Requirements

- Python 3.8 or later

#### Install Python using MiniConda

1) Download and install MiniConda from [here](https://docs.anaconda.com/free/miniconda/#quick-command-line-install)
2) Create a new environment using the following command:
```bash
$ conda create -n mini-rag-app python=3.8
```
3) Activate the environment:
```bash
$ conda activate mini-rag-app
```

### (Optional) Setup you command line interface for better readability

```bash
export PS1="\[\033[01;32m\]\u@\h:\w\n\[\033[00m\]\$ "
```

## Installation

### Install the required packages

```bash
$ pip install -r requirements.txt
```

### Setup the environment variables

```bash
$ cp .env.example .env
```

Set your environment variables in the `.env` file. For example, copy the `.env.example` and adjust values:

```
cp src/.env.example src/.env
```

The important variables are:

* `OPENAI_API_KEY` – your OpenAI credential (required for processing).
* `FILE_ALLOWED_TYPES` – a comma-separated list of **file extensions** (e.g. `txt,pdf,docx`),
  **MIME types** (`text/plain,application/pdf`), or a JSON array such as
  `["txt","pdf"]`.  The application will match both the uploaded file’s
  `content_type` and its extension.
* `FILE_MAX_SIZE` – maximum size in bytes (10 MB by default).
* `FILE_DEFAULT_CHUNK_SIZE` – chunk size used when streaming uploads.

**Error signals**

* `file_not_found` – returned when the specified project or file ID does not
  exist (avoids a 500 error).
* `processing_failed` – generic failure when something goes wrong while
  splitting or loading the document.

You can add or remove allowed formats simply by editing `FILE_ALLOWED_TYPES`.


## Run the FastAPI server

```bash
$ uvicorn main:app --reload --host 0.0.0.0 --port 5000
```

## POSTMAN Collection

Download the POSTMAN collection from [/assets/mini-rag-app.postman_collection.json](/assets/mini-rag-app.postman_collection.json)
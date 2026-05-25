# RAG Chatbot with Document Upload

A simple Python-based Retrieval-Augmented Generation (RAG) chatbot with a web UI for uploading a document and asking questions.

## Features

- Upload a PDF, TXT, or MD file
- Create embeddings from the uploaded document
- Chat with the document using retrieval-based QA
- Simple Flask UI with upload and chat

## Setup

1. Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

4. Run the app:

```bash
python app.py
```

5. Open `http://127.0.0.1:5000` in your browser.

## Notes

- Upload a document before asking questions.
- The app uses in-memory vector storage; restart to reset uploaded content.

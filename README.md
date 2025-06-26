# Server Documentation

## 1. Overview

This server handles regulatory document processing with the following core components:

- PDF document ingestion
- Q&A pair generation
- JSON data storage in `data_json/`

## 2. Key Files

| File Name               | Purpose                                  |
| ----------------------- | ---------------------------------------- |
| regulation_extractor.py | PDF to text conversion and preprocessing |
| qa_generator.py         | Q&A pair generation using NLP models     |
| rag_server_reranker.py  | Main API server for document retrieval   |
| data_json/sample.json   | Example Q&A structure format             |

## 3. Directory Structure

```
server/
├── regulation_extractor.py     # Document parsing logic
├── qa_generator.py             # Q&A generation algorithm
├── rag_server_reranker.py      # Main API server implementation
├── data_json/                  # Generated Q&A JSON files
│   └── sample.json             # Example output format
├── .gitignore                  # Ignored files (logs, env files)
└── README.md                   # This file
```

## 4. Usage Guide

1. **Setup**

```bash
# Install dependencies
pip install -r requirements.txt

# Start Main server
python3 rag_server_reranker.py

# Start Document server
python3 regulation_extractor.py
```

2. **API Endpoints**

- `POST /upload-pdf` - Upload regulatory documents
- `GET /qa/:document_id` - Retrieve Q&A pairs

## 5. Data Format

Sample JSON structure (auto-generated in `data_json/`):

```json
{
  "document_id": "reg_2025_01",
  "content": [
    {
      "question": "What is the purpose of this regulation?",
      "answer": "To establish employee management guidelines..."
    }
  ]
}
```

## 6. Configuration

- PDF files should be placed in `server/data/`
- Q&A results are saved to `server/data_json/`

## 7. Dependencies

Required packages (install via `pip`):

- PyPDF2
- langchain
- fastapi
- uvicorn (for development server)

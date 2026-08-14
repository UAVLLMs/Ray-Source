# Raysource Backend: Independent Deployment

This package contains the FastAPI service, retrieval and section data, BM25 and
FAISS index, image captions, local manual images, chunk pipeline, and tests. It
does not import or read files from the web client.

## Start

1. Create a Python 3.11 or 3.12 virtual environment.
2. Install `requirements.txt`.
3. Copy `.env.example` to `.env` and configure provider keys and
   `KAFU_API_TOKEN`.
4. Run `./start.ps1` from this directory, or start Uvicorn directly:

   `python -m uvicorn api_server:app --host 127.0.0.1 --port 8011 --workers 1`

The backend exposes `/health`, `/retrieve`, `/chat`, `/translate`, and the
authenticated chunk-management API. It can be called by the packaged web
gateway, another application, or an evaluation runner.

Manual image grounding defaults to `data/manual-images`. Override it only when
needed with `RAYSOURCE_MANUAL_IMAGE_DIR`.

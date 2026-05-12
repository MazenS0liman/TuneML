# <img src="static/images/music.png" alt="Tune ML" width="100" align="center" style="margin-right: 10px;"/> Tune ML

Tune ML is a Python project for audio processing, classification, and generation with deep learning. The repository combines a FastAPI application shell, model implementations, training utilities and notebooks.

## Requirements

- Python 3.10 or newer
- PyTorch and the project dependencies listed in `pyproject.toml`

## Installation

Using Poetry:

```bash
poetry install
```

Using pip:

```bash
pip install -e .
```

## Running the App

Start the API server directly:

```bash
python main.py
```

Or run it with Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

When the server is running, open:

- `http://localhost:8001/` for the root status response
- `http://localhost:8001/docs` for the Swagger UI in non-production environments

## :warning: Note
- This project is still under construction.


# Who's My Good Boy - AI Image Classification Service

A FastAPI-based backend service for classifying images using various AI models.

## Features

- General image classification endpoint
- Dog breed detection endpoint
- "Apolo" (specific dog) detection endpoint
- Clean API with proper error handling
- Comprehensive test suite

## Project Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── health.py      # Health check endpoint
│   │   │   └── predictions.py # Classification endpoints
│   │   └── dependencies.py    # Dependency injection
│   ├── core/
│   │   ├── config.py          # App configuration
│   │   └── errors.py          # Error handling
│   ├── models/
│   │   └── schemas.py         # Pydantic models
│   ├── services/
│   │   ├── base.py            # Base classifier service
│   │   ├── general_classifier.py # General image classifier
│   │   ├── dog_classifier.py     # Dog detection
│   │   └── apolo_classifier.py   # Apolo detection
│   └── main.py                # FastAPI app
├── tests/
│   ├── api/
│   │   ├── test_health.py
│   │   └── test_predictions.py
│   └── test_data/             # Test images
└── requirements.txt           # Dependencies
```

## Setup

### Prerequisites

- Python 3.9+
- pip

### Installation

1. Clone the repository:

   ```
   git clone <repository-url>
   cd whos-my-good-boy/backend
   ```

2. Create a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the application

```
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000

### Running tests

```
pytest
```

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /api/v1/classify` - General image classification
- `POST /api/v1/is-dog` - Dog detection
- `POST /api/v1/is-apolo` - Apolo detection

## Docker

You can build and run the application using Docker:

```
docker build -t whos-my-good-boy-api .
docker run -p 8000:8000 whos-my-good-boy-api
```

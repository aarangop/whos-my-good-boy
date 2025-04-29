import time
import uuid
from loguru import logger
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, predictions
from app.core.config import settings
from app.core.logging import setup_logging

app = FastAPI(
    title="Who's My Good Boy API",
    description="AI service for image classification",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging with settings
setup_logging(log_level=settings.LOG_LEVEL, json_logs=settings.JSON_LOGS)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Middleware to log all requests and responses"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Log request details
    logger.info(f"Request {request_id} - {request.method} {request.url.path}")

    # Measure time
    start_time = time.time()

    # Process request
    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        # Log successful response
        logger.info(
            f"Response {request_id} - Status: {response.status_code} - Took: {process_time:.3f}s")

        return response
    except Exception as e:
        # Log exceptions
        process_time = time.time() - start_time
        logger.error(
            f"Response {request_id} - Exception: {str(e)} - Took: {process_time:.3f}s")
        raise


# Include routers
app.include_router(health.router)
app.include_router(predictions.router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

# To run this application from the command line:
# If you are in the backend directory:
# uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
#
# If you are in the project root directory:
# uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

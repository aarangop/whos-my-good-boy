import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, predictions
from app.core.config import settings

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

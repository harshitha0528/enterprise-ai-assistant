from fastapi import FastAPI

from app.config import APP_NAME, APP_ENV
from app.routers import documents, incidents
from contextlib import asynccontextmanager
from app.database import Base, engine
import app.models  # noqa: F401  (importing registers the tables)

app = FastAPI(
    title=APP_NAME,
    description="An AI assistant for document and incident intelligence",
    version="0.1.0",
)


@app.get("/")
def home():
    return {
        "message": f"Welcome to {APP_NAME}!",
        "environment": APP_ENV,
        "status": "Application is running",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


app.include_router(documents.router)
app.include_router(incidents.router)

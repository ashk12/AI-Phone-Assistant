from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes
from app.config import FRONTEND_URL

app = FastAPI(
    title="Phone AI Assistant",
    description="Multi-intent phone assistant API",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(routes.router)

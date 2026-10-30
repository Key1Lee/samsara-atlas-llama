from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.storage.database import init_db
from app.api.routes import router
from app.middleware.rate_limit import employee_daily_limit
from app.middleware.request_id import correlate_request


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    import os
    provider = os.getenv("PROVIDER_TYPE", "stub")
    print(f"[Samsara Atlas] Provider: {provider}")
    yield


app = FastAPI(title="Samsara Atlas", lifespan=lifespan)
app.middleware("http")(employee_daily_limit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(correlate_request)
app.include_router(router, prefix="", tags=["cases"])

# Disabled by default; dedicated developer credential plus employee policy scope.
from app.rag.diagnostics import router as rag_diagnostic_router
app.include_router(rag_diagnostic_router, tags=["developer diagnostics"])

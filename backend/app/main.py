import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import cvs, query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# chromadb's bundled posthog telemetry is incompatible with the installed posthog
# version (capture() signature mismatch), so it logs a spurious error on every call.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

app = FastAPI(title="CV RAG Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cvs.router)
app.include_router(query.router)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}

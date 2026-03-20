#!/usr/bin/env python3
"""
Enterprise-Grade FastAPI server for the Bilingual NLP Toolkit.

Implemented Features:
1. P0: Singleton Model Management with Warmup
2. P0: Structured Exception Handling (No silent failures)
3. P1.a: Security Layer (API Key, Rate Limiting, Payload Size Limit)
4. P1.b: Observability Core (Prometheus Metrics, JSON Structured Logging, Request Tracking)
"""

import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Internal Modules
try:
    import bilingual as bb
    from bilingual.models.manager import model_manager
    from bilingual.exceptions import BilingualError, ModelLoadError, InferenceError
    from bilingual.api.security import (
        validate_api_key, 
        limit_payload_size, 
        global_rate_limiter,
        API_KEY_NAME
    )
    BILINGUAL_AVAILABLE = True
except ImportError as e:
    print(f"Critical Import Error: {e}")
    BILINGUAL_AVAILABLE = False

# --- OBSERBABILITY CONFIGURATION (P1.b) ---

class JsonFormatter(logging.Formatter):
    """Formats logs as JSON for production observability."""
    def format(self, record):
        log_records = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "request_id": getattr(record, "request_id", "GLOBAL"),
        }
        if record.exc_info:
            log_records["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_records)

# Setup Structured Logging
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logging.getLogger().handlers = [handler]
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("bilingual.api")

# Prometheus Metrics
REQUEST_COUNT = Counter("bilingual_requests_total", "Total API Requests", ["endpoint", "status"])
LATENCY_HISTOGRAM = Histogram(
    "bilingual_request_latency_seconds", 
    "Request Latency in Seconds", 
    ["endpoint", "model"]
)
INFERENCE_LATENCY = Histogram(
    "bilingual_inference_latency_seconds",
    "Model Inference Latency",
    ["model_name"]
)

# --- REQUEST/RESPONSE SCHEMAS ---

class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    source_lang: str = Field("auto")
    target_lang: str = Field("bn")
    model: str = Field("t5-small")

class TranslationResponse(BaseModel):
    translated_text: str
    processing_time_ms: float
    model_used: str
    request_id: str

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    uptime_seconds: float

# --- LIFECYCLE MANAGEMENT (P0) ---

# --- SCHEMAS ---
class RAGRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    version: Optional[str] = None

class RAGResponse(BaseModel):
    answer: str
    metrics: Dict[str, Any]
    request_id: str
    processing_time_ms: float

# --- LIFECYCLE (P3.2) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Initializing Distributed Gateway (Ray Mode)...")
    # Initialize Ray Serve Handle
    try:
        # Connect to existing deployment (Expects 'bilingual_rag' to be live)
        app.state.ray_handle = serve.get_app_handle("bilingual_rag")
        logger.info("✅ Connected to Ray Serve cluster.")
    except Exception as e:
        logger.error(f"⚠️ Ray Serve not found: {e}. Falling back to local mode.")
        app.state.ray_handle = None
        
    if BILINGUAL_AVAILABLE and not app.state.ray_handle:
        model_manager.warmup(["t5-small"])
    
    yield
    model_manager.clear_cache()

# --- APP CONFIGURATION ---

app = FastAPI(
    title="Bilingual NLP Gateway",
    version="1.1.0",
    lifespan=lifespan,
    dependencies=[Depends(limit_payload_size())] # Global Payload Size Safeguard
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ERROR HANDLING (P0) ---

@app.exception_handler(BilingualError)
async def structured_error_handler(request: Request, exc: BilingualError):
    logger.error(f"Structured Error: {type(exc).__name__}", extra={"request_id": getattr(request.state, "request_id", None)})
    return JSONResponse(
        status_code=500,
        content={
            "error_type": type(exc).__name__,
            "message": str(exc),
            "details": exc.details,
            "timestamp": datetime.now().isoformat()
        }
    )

# --- SECURITY & TRACING MIDDLEWARE (P1.a + P1.b) ---

@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    client_ip = request.client.host
    
    # 1. Rate Limiting Check
    if not global_rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for {client_ip}")
        return JSONResponse(status_code=429, content={"error": "RateLimitExceeded"})

    # 2. Latency Tracking
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    # 3. Metrics Recording
    REQUEST_COUNT.labels(endpoint=request.url.path, status=response.status_code).inc()
    LATENCY_HISTOGRAM.labels(endpoint=request.url.path, model="system").observe(duration)
    
    response.headers["X-Request-ID"] = request_id
    return response


# Error handling middleware
@app.middleware("http")
async def error_handling_middleware(request, call_next):
    global ERROR_COUNT

    try:
        response = await call_next(request)

        # Count 4xx and 5xx errors
        if response.status_code >= 400:
            ERROR_COUNT += 1
            if PROMETHEUS_AVAILABLE:
                ERROR_COUNTER.labels(
                    endpoint=str(request.url.path), error_type=str(response.status_code)
                ).inc()

        return response

    except Exception as e:
        ERROR_COUNT += 1
        if PROMETHEUS_AVAILABLE:
            ERROR_COUNTER.labels(endpoint=str(request.url.path), error_type=type(e).__name__).inc()

        # Return error response
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            },
        )


# Routes


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bilingual NLP API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .hero { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; border-radius: 10px; text-align: center; margin-bottom: 30px; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }
            .method { font-weight: bold; color: #007bff; }
            code { background: #e9ecef; padding: 2px 6px; border-radius: 3px; font-family: 'Courier New', monospace; }
        </style>
    </head>
    <body>
        <div class="hero">
            <h1>🌏 Bilingual NLP API</h1>
            <p>Production-ready API for advanced Bangla-English text processing</p>
        </div>

        <h2>🚀 Quick Start</h2>

        <div class="endpoint">
            <div class="method">GET /health</div>
            <p>Check server health and status</p>
            <code>curl http://localhost:8000/health</code>
        </div>

        <div class="endpoint">
            <div class="method">POST /detect-language</div>
            <p>Detect language of text</p>
            <code>curl -X POST "http://localhost:8000/detect-language" -H "Content-Type: application/json" -d '{"text": "Hello world"}'</code>
        </div>

        <div class="endpoint">
            <div class="method">POST /translate</div>
            <p>Translate text between languages</p>
            <code>curl -X POST "http://localhost:8000/translate" -H "Content-Type: application/json" -d '{"text": "Hello world", "target_lang": "bn"}'</code>
        </div>

        <div class="endpoint">
            <div class="method">POST /generate</div>
            <p>Generate text using AI models</p>
            <code>curl -X POST "http://localhost:8000/generate" -H "Content-Type: application/json" -d '{"prompt": "Write a story about friendship"}'</code>
        </div>

        <div class="endpoint">
            <div class="method">POST /evaluate</div>
            <p>Evaluate model outputs</p>
            <code>curl -X POST "http://localhost:8000/evaluate" -H "Content-Type: application/json" -d '{"task": "translation", "references": ["Hello"], "candidates": ["Hello"]}'</code>
        </div>

        <h2>📚 Documentation</h2>
        <p><a href="/docs">Interactive API Documentation (Swagger/ReDoc)</a></p>
        <p><a href="/redoc">Alternative API Documentation</a></p>

        <h2>🔧 Features</h2>
        <ul>
            <li>🚀 High-performance async API</li>
            <li>🌍 Bilingual Bangla-English support</li>
            <li>🤖 State-of-the-art transformer models</li>
            <li>📊 Comprehensive evaluation metrics</li>
            <li>🛡️ Production-ready with monitoring</li>
            <li>🔒 Type-safe with Pydantic validation</li>
        </ul>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        uptime_seconds=time.time() - getattr(app.state, "start_time", time.time())
    )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics scraper endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# --- REGISTRY ENDPOINTS ---

@app.get("/registry/models", response_model=ModelListResponse)
async def list_registry_models():
    """List all available models and versions in the registry."""
    entries = model_registry.list_models()
    return {"models": [e.to_dict() for e in entries]}

# --- DISTRIBUTED ENDPOINTS ---

@app.post("/rag/query", response_model=RAGResponse)
async def distributed_rag_query(
    request: RAGRequest,
    api_key: str = Security(validate_api_key)
):
    """
    Proxies RAG requests with versioning support.
    Ensures backpressure-aware async distribution.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Pass version to Ray or local executor
        payload = {
            "query": request.query, 
            "version": request.version, 
            "top_k": request.top_k
        }

        if app.state.ray_handle:
            # Async call to Ray RAGService (P3.2 Requirement)
            # handle.remote() returns a Ray ObjectRef (wrapped in DeploymentHandle)
            result_ref = await app.state.ray_handle.remote(payload)
            result = result_ref
        else:
            # Fallback to local orchestrator if Ray is down
            from bilingual.rag.orchestrator import RAGOrchestrator
            from bilingual.rag.vector_store.faiss_index import BilingualVectorStore
            # Note: In production, VectorStore should be pre-loaded
            vs = BilingualVectorStore(dimension=384)
            # Orchestrator should ideally be version-aware now
            orch = RAGOrchestrator(vector_store=vs, generation_model_name="bilingual-small")
            # Update: RAGOrchestrator.generate_with_context now accepts version via manager
            result = orch.generate_with_context(request.query)

        duration_ms = (time.time() - start_time) * 1000
        
        return RAGResponse(
            answer=result["answer"],
            metrics=result.get("metrics", {}),
            request_id=request_id,
            processing_time_ms=duration_ms
        )

    except Exception as e:
        logger.exception(f"Distributed Proxy Error [ID: {request_id}]")
        raise InferenceError(f"Gateway failed to relay to Ray: {str(e)}")

@app.post("/translate", response_model=TranslationResponse)
async def translate(
    request: TranslationRequest, 
    api_key: str = Security(validate_api_key) # API Key Protection
):
    """
    Protected Translation Endpoint.
    Uses Singleton ModelManager for efficient inference.
    """
    start_time = time.time()
    try:
        # Load model through manager (quantized by default)
        model = bb.load_model(request.model)
        
        # Inference
        with INFERENCE_LATENCY.labels(model_name=request.model).time():
            result = bb.translate_text(
                model, 
                request.text, 
                src_lang=request.source_lang, 
                tgt_lang=request.target_lang
            )
        
        return TranslationResponse(
            translated_text=result,
            processing_time_ms=(time.time() - start_time) * 1000,
            model_used=request.model,
            request_id=str(uuid.uuid4()) # Added for extra traceability
        )
    except Exception as e:
        logger.exception("Inference Failure")
        raise InferenceError(f"Model failed to process translation: {str(e)}")

@app.get("/")
async def root():
    return {
        "app": "Bilingual NLP Toolkit",
        "docs": "/docs",
        "metrics": "/metrics",
        "status": "Enterprise Ready"
    }


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app


def run_server(host: str = "localhost", port: int = 8000, workers: int = 1, reload: bool = False):
    """Run the FastAPI server."""
    # Use configuration defaults when available and arguments are not explicitly overridden
    if BILINGUAL_AVAILABLE:
        try:
            settings = get_settings()
            if host == "localhost":
                host = settings.api.host
            if port == 8000:
                port = settings.api.port
            if workers == 1:
                workers = settings.api.workers
        except Exception:
            pass

    print(f"🚀 Starting server on {host}:{port}")
    print(f"📚 API docs available at: http://{host}:{port}/docs")
    print(f"🔍 Health check: http://{host}:{port}/health")

    uvicorn.run(
        "bilingual.server:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    # Record start time for uptime tracking
    app.state.start_time = time.time()
    uvicorn.run(app, host="0.0.0.0", port=8000)

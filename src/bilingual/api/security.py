"""
Security and Access Control for the Bilingual NLP API.

Implements:
1. API Key Validation
2. Thread-safe Rate Limiting
3. Payload Size Constraints
"""

import time
import threading
import logging
from typing import Dict, Optional
from fastapi import Security, HTTPException, Request
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN, HTTP_429_TOO_MANY_REQUESTS

logger = logging.getLogger("bilingual.security")

# Constants
API_KEY_NAME = "X-Bilingual-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Simple In-memory store for Rate Limiting (P1.a Requirement)
# In production, this should be moved to Redis
class RateLimiter:
    def __init__(self, requests_limit: int, window_seconds: int):
        self.limit = requests_limit
        self.window = window_seconds
        self.requests: Dict[str, list] = {}
        self._lock = threading.Lock()

    def is_allowed(self, identifier: str) -> bool:
        now = time.time()
        with self._lock:
            if identifier not in self.requests:
                self.requests[identifier] = [now]
                return True
            
            # Remove expired timestamps
            self.requests[identifier] = [t for t in self.requests[identifier] if now - t < self.window]
            
            if len(self.requests[identifier]) < self.limit:
                self.requests[identifier].append(now)
                return True
            return False

# Global instances (Configurable via ENV normally)
# Default: 60 requests per minute per IP
global_rate_limiter = RateLimiter(requests_limit=60, window_seconds=60)
VALID_API_KEYS = {"kothagpt-dev-2026", "internal-test-key"} # Placeholder storage

async def validate_api_key(api_key: str = Security(api_key_header)):
    """Validates the API Key from headers."""
    if api_key in VALID_API_KEYS:
        return api_key
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, detail="Invalid or missing API Key"
    )

async def rate_limit_middleware(request: Request, call_next):
    """Global rate limiting middleware based on Client IP."""
    client_ip = request.client.host
    if not global_rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return HTTPException(
            status_code=HTTP_429_TOO_MANY_REQUESTS, detail="Too many requests. Please slow down."
        )
    return await call_next(request)

def limit_payload_size(max_size: int = 5 * 1024 * 1024): # 5MB Default
    """Dependency to limit request body size."""
    async def size_limiter(request: Request):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > max_size:
            raise HTTPException(status_code=413, detail="Payload too large")
    return size_limiter

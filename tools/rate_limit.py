"""Shared rate limits for deployments using the Upstash Redis REST API."""

import hashlib
import os

import httpx
from fastapi import HTTPException, Request


_LIMIT = 5
_WINDOW_SECONDS = 60
_INCREMENT = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end
return {count, redis.call('TTL', KEYS[1])}
"""


def enforce_research_limit(request: Request) -> None:
    url = os.getenv("UPSTASH_REDIS_REST_URL")
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
    if not url and not token:
        return
    if not url or not token:
        raise HTTPException(503, "The research rate limiter is unavailable.")

    address = request.client.host if request.client else "unknown"
    if os.getenv("VERCEL") == "1":
        address = request.headers.get("x-vercel-forwarded-for", address).split(",", 1)[0].strip()
    fingerprint = hashlib.sha256(address.encode()).hexdigest()
    environment = os.getenv("VERCEL_ENV", "local")
    key = f"research:rate:{environment}:{fingerprint}"
    try:
        response = httpx.post(
            url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {token}"},
            json=["EVAL", _INCREMENT, "1", key, str(_WINDOW_SECONDS)],
            timeout=2,
        )
        response.raise_for_status()
        count, ttl = response.json()["result"]
        if type(count) is not int or type(ttl) is not int or count < 1 or ttl < 0:
            raise ValueError("Unexpected rate limit counter")
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(503, "The research rate limiter is unavailable.") from exc

    if count > _LIMIT:
        raise HTTPException(429, "Too many research requests. Please wait and retry.",
                            headers={"Retry-After": str(max(1, ttl))})

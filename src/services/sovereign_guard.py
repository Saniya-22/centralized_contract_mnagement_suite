"""Optional Sovereign-AI guardrail integration for generated responses."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging
import time

import httpx

from src.config import settings

logger = logging.getLogger(__name__)


class SovereignGuard:
    """Calls a Sovereign-AI detection API and normalizes the verdict."""

    def __init__(self) -> None:
        self.enabled = bool(settings.SOVEREIGN_GUARD_ENABLED)
        self.base_url = settings.SOVEREIGN_GUARD_BASE_URL.rstrip("/")
        self.detect_path = settings.SOVEREIGN_GUARD_DETECT_PATH
        self.timeout_seconds = float(settings.SOVEREIGN_GUARD_TIMEOUT_SECONDS)
        self.fail_open = bool(settings.SOVEREIGN_GUARD_FAIL_OPEN)
        self.auth_token = settings.SOVEREIGN_GUARD_AUTH_TOKEN

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _build_detect_urls(self) -> List[str]:
        detect_path = (self.detect_path or "/detect").strip()
        if not detect_path.startswith("/"):
            detect_path = f"/{detect_path}"

        urls = [f"{self.base_url}{detect_path}"]

        # Sovereign-AI examples vary between /detect and /api/detect; try both.
        if detect_path == "/detect":
            urls.append(f"{self.base_url}/api/detect")
        elif detect_path == "/api/detect":
            urls.append(f"{self.base_url}/detect")

        # Preserve order while de-duplicating
        seen: set[str] = set()
        deduped: List[str] = []
        for url in urls:
            if url not in seen:
                deduped.append(url)
                seen.add(url)
        return deduped

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def _parse_verdict(
        self,
        payload: Dict[str, Any],
        processing_time_ms: Optional[float] = None,
        endpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        action = str(payload.get("action") or "allow").lower()
        blocked = bool(
            payload.get("should_block") or payload.get("blocked") or action == "block"
        )
        reason = payload.get("reason") or payload.get("explanation")

        return {
            "provider": "sovereign_ai",
            "action": action,
            "should_block": blocked,
            "confidence": self._safe_float(payload.get("confidence")),
            "tier_used": self._safe_int(payload.get("tier_used")),
            "method": payload.get("method"),
            "reason": reason,
            "processing_time_ms": self._safe_float(payload.get("processing_time_ms"))
            or processing_time_ms,
            "endpoint": endpoint,
        }

    def _error_verdict(self, reason: str) -> Dict[str, Any]:
        should_block = not self.fail_open
        return {
            "provider": "sovereign_ai",
            "action": "block" if should_block else "allow",
            "should_block": should_block,
            "confidence": None,
            "tier_used": None,
            "method": "integration_error",
            "reason": reason,
            "processing_time_ms": None,
            "endpoint": None,
        }

    def evaluate_response(
        self,
        response_text: str,
        query: str,
        documents: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate generated response against external Sovereign-AI guardrails."""
        if not self.enabled:
            return None
        if not response_text or not response_text.strip():
            return None

        payload = {
            # Keep both keys for compatibility across Sovereign-AI API variants.
            "text": response_text,
            "llm_response": response_text,
            "context": {
                "query": query,
                "document_count": len(documents or []),
            },
        }

        urls = self._build_detect_urls()
        headers = self._headers()

        with httpx.Client(timeout=self.timeout_seconds) as client:
            for url in urls:
                start = time.perf_counter()
                try:
                    response = client.post(url, json=payload, headers=headers)
                    elapsed_ms = (time.perf_counter() - start) * 1000.0

                    if response.status_code == 404:
                        logger.info(
                            "Sovereign guard endpoint not found at %s, trying fallback",
                            url,
                        )
                        continue
                    if response.status_code >= 400:
                        reason = (
                            f"Sovereign guard HTTP {response.status_code} at {url}: "
                            f"{response.text[:200]}"
                        )
                        logger.warning(reason)
                        return self._error_verdict(reason)

                    data = response.json()
                    verdict = self._parse_verdict(
                        payload=data,
                        processing_time_ms=elapsed_ms,
                        endpoint=url,
                    )
                    logger.info(
                        "Sovereign guard verdict action=%s block=%s tier=%s conf=%s",
                        verdict.get("action"),
                        verdict.get("should_block"),
                        verdict.get("tier_used"),
                        verdict.get("confidence"),
                    )
                    return verdict
                except Exception as exc:
                    reason = f"Sovereign guard request failed at {url}: {exc}"
                    logger.warning(reason)
                    return self._error_verdict(reason)

        return self._error_verdict(
            f"Sovereign guard endpoint not found on configured paths: {', '.join(urls)}"
        )

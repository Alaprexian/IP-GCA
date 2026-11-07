"""
Latency profiler for IP-GCA retrieval pipeline.
Measures TTFT, per-stage latency and P95/P99 distributions.
"""

import time
import statistics
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
from pydantic import BaseModel


class StageTimings(BaseModel):
    stage: str
    duration_ms: float


class RequestProfile(BaseModel):
    query: str
    total_latency_ms: float
    stage_timings: List[StageTimings]
    ttft_ms: Optional[float] = None
    result_count: int = 0


class LatencyTracker:
    def __init__(self):
        self._profiles: List[RequestProfile] = []
        self._current_stages: List[StageTimings] = []
        self._request_start: Optional[float] = None

    def start_request(self, query: str) -> None:
        self._current_query = query
        self._current_stages = []
        self._request_start = time.perf_counter()

    @contextmanager
    def measure(self, stage_name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._current_stages.append(StageTimings(stage=stage_name, duration_ms=elapsed_ms))

    def finish_request(self, result_count: int = 0, ttft_ms: Optional[float] = None) -> RequestProfile:
        if self._request_start is None:
            raise RuntimeError("Call start_request() first.")
        total_ms = (time.perf_counter() - self._request_start) * 1000.0
        profile = RequestProfile(
            query=self._current_query,
            total_latency_ms=total_ms,
            stage_timings=list(self._current_stages),
            ttft_ms=ttft_ms,
            result_count=result_count
        )
        self._profiles.append(profile)
        self._request_start = None
        return profile

    def get_percentile(self, p: float = 0.95) -> float:
        """Returns the Pth percentile of total latency across all profiled requests."""
        if not self._profiles:
            return 0.0
        latencies = sorted(r.total_latency_ms for r in self._profiles)
        idx = max(0, int(len(latencies) * p) - 1)
        return latencies[idx]

    def summary(self) -> Dict[str, Any]:
        if not self._profiles:
            return {}
        latencies = [r.total_latency_ms for r in self._profiles]
        return {
            "n_requests": len(self._profiles),
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "p95_ms": self.get_percentile(0.95),
            "p99_ms": self.get_percentile(0.99),
            "min_ms": min(latencies),
            "max_ms": max(latencies)
        }

"""
Pipeline tracing — records per-agent timing, token usage, and flags.
The tracer is created per job in main.py and stored in the job store.

Logging emits structured JSON so lines are parseable by Datadog / CloudWatch:
    {"timestamp":"…","level":"INFO","logger":"genesight","message":"…","job_id":"…","agent":"…"}

The job_id and agent name are injected automatically via contextvars when
PipelineTracer.start() is active, so every log line in that async context
carries them — no need to pass them manually to every logger.info() call.
"""
from __future__ import annotations
import json
import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

# ── Context vars — set per-job so all log lines carry job_id/agent ────────────
_ctx_job_id: ContextVar[str] = ContextVar("job_id", default="")
_ctx_agent:  ContextVar[str] = ContextVar("agent",  default="")


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        doc: dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
        }
        # Inject pipeline context when available
        if job_id := _ctx_job_id.get():
            doc["job_id"] = job_id
        if agent := _ctx_agent.get():
            doc["agent"] = agent
        # Pass-through any extra fields the caller attached
        for key in ("exc_info", "exc_text", "stack_info"):
            pass
        if record.exc_info:
            doc["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(doc, ensure_ascii=False)


def _configure_logging() -> None:
    root = logging.getLogger()
    if any(isinstance(h, logging.StreamHandler) and
           isinstance(h.formatter, _JsonFormatter)
           for h in root.handlers):
        return  # already configured

    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    root.setLevel(logging.INFO)
    # Replace any existing handlers so we don't get duplicate lines
    root.handlers = [handler]


_configure_logging()
logger = logging.getLogger("genesight")


# OpenAI pricing (USD per 1k tokens, as of mid-2025 — update if pricing changes)
_COST_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o":            (0.005,  0.015),   # input, output
    "gpt-4o-mini":       (0.00015, 0.0006),
    "gpt-4-turbo":       (0.01,   0.03),
    "gpt-3.5-turbo":     (0.001,  0.002),
}
_DEFAULT_COST = (0.005, 0.015)  # fall back to gpt-4o rates if model unknown


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return estimated USD cost for a single LLM call."""
    in_rate, out_rate = _COST_PER_1K.get(model, _DEFAULT_COST)
    return round(
        (prompt_tokens / 1000) * in_rate + (completion_tokens / 1000) * out_rate,
        6,
    )


@dataclass
class AgentTrace:
    agent: str
    started_at: float = field(default_factory=time.monotonic)
    ended_at: float | None = None
    duration_s: float | None = None
    input_summary: str = ""
    output_summary: str = ""
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    estimated_cost_usd: float = 0.0
    flags: list[str] = field(default_factory=list)
    error: str | None = None

    def finish(
        self,
        output_summary: str = "",
        tokens: int = 0,
        flags: list[str] | None = None,
        model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        self.ended_at = time.monotonic()
        self.duration_s = round(self.ended_at - self.started_at, 2)
        self.output_summary = output_summary
        self.tokens_used = tokens or (prompt_tokens + completion_tokens)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.model = model
        if model and (prompt_tokens or completion_tokens):
            self.estimated_cost_usd = estimate_cost_usd(model, prompt_tokens, completion_tokens)
        if flags:
            self.flags = flags
        logger.info(
            "✓ [%s] %.2fs | %s%s%s",
            self.agent,
            self.duration_s,
            self.output_summary,
            f" | flags={self.flags}" if self.flags else "",
            f" | cost=${self.estimated_cost_usd:.5f}" if self.estimated_cost_usd else "",
        )

    def fail(self, error: str) -> None:
        self.ended_at = time.monotonic()
        self.duration_s = round(self.ended_at - self.started_at, 2)
        self.error = error
        logger.error("✗ [%s] %.2fs | %s", self.agent, self.duration_s, error[:300])


class PipelineTracer:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self.traces: list[AgentTrace] = []
        _ctx_job_id.set(job_id)
        _ctx_agent.set("")
        logger.info("Pipeline starting")

    def start(self, agent_name: str, input_summary: str = "") -> AgentTrace:
        _ctx_agent.set(agent_name)
        trace = AgentTrace(agent=agent_name, input_summary=input_summary)
        self.traces.append(trace)
        logger.info("Agent starting | %s", input_summary)
        return trace

    def to_dict(self) -> list[dict]:
        return [asdict(t) for t in self.traces]

    @property
    def total_tokens(self) -> int:
        return sum(t.tokens_used for t in self.traces)

    @property
    def total_duration_s(self) -> float:
        times = [t.ended_at for t in self.traces if t.ended_at]
        starts = [t.started_at for t in self.traces if t.started_at]
        if not times or not starts:
            return 0.0
        return round(max(times) - min(starts), 2)

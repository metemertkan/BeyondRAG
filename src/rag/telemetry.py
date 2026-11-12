"""Traces, token counts and one outcome label per request, on OpenTelemetry.

Three things live here.

1. A request record. It collects what happened while one question was answered: the model
   calls with their token counts, the guardrail decisions, the time per stage, and the
   events that deeper layers report with ``note()``. It is returned in the chat response,
   so it works with no backend at all.
2. Spans and metrics, named after the OpenTelemetry GenAI conventions. They leave the
   process only when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set.
3. A LangChain callback handler. ``get_llm`` attaches it to every model it builds, so a
   model call is recorded without the calling code knowing. A call that does not go through
   LangChain has to use ``llm_call()`` itself, or it is invisible.

Stored text is always masked for personal data. Prompts and answers are stored only when
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Final, Iterator, List, Optional, Sequence
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from opentelemetry import metrics, trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from src.rag.pii import redact_pii

logger = logging.getLogger(__name__)
request_logger = logging.getLogger("beyondrag.request")

# One label per request. The group is what a dashboard or an alert counts.
OUTCOME_GROUPS: Final[Dict[str, str]] = {
    "answered": "ok",
    "answered_extractive": "ok",
    "degraded_llm_fallback": "degraded",
    "degraded_model_substituted": "degraded",
    "budget_exceeded": "degraded",
    "answered_with_unsupported_claims": "degraded",
    "abstained_model": "refused",
    "abstained_low_relevance": "refused",
    "abstained_unverified": "refused",
    "no_documents": "refused",
    "access_blocked": "refused",
    "blocked_input": "refused",
    "blocked_output": "refused",
    "error": "error",
}

_DURATION_BUCKETS: Final[List[float]] = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120, 300]
_TOKEN_BUCKETS: Final[List[float]] = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144]
_PROVIDER_NAMES: Final[Dict[str, str]] = {"google_genai": "gcp.gemini", "google": "gcp.gemini"}
_MAX_TEXT: Final[int] = 2000
_MAX_DETAIL: Final[int] = 300


def safe_text(text: Any, limit: int = _MAX_TEXT) -> str:
    """Mask personal data and cut to ``limit``. Everything telemetry stores goes through this."""
    masked, _ = redact_pii(str(text or ""))
    return masked if len(masked) <= limit else masked[:limit] + "…"


def capture_content() -> bool:
    return os.getenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false").strip().lower() == "true"


@dataclass
class LlmCallRecord:
    stage: str
    model: str
    provider: str
    operation: str = "chat"
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    duration_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class RequestRecord:
    tenant: str
    synthesis_mode: str
    trace_id: str = ""
    outcome: str = ""
    detail: str = ""
    events: List[Dict[str, str]] = field(default_factory=list)
    llm_calls: List[LlmCallRecord] = field(default_factory=list)
    guardrails: List[Dict[str, str]] = field(default_factory=list)
    stages_ms: Dict[str, float] = field(default_factory=dict)
    started: float = field(default_factory=time.perf_counter)
    duration_ms: float = 0.0

    def has(self, event: str) -> bool:
        return any(item["event"] == event for item in self.events)

    @property
    def input_tokens(self) -> int:
        return sum(call.input_tokens or 0 for call in self.llm_calls)

    @property
    def output_tokens(self) -> int:
        return sum(call.output_tokens or 0 for call in self.llm_calls)

    def to_dict(self) -> Dict[str, Any]:
        by_stage: Dict[str, Dict[str, int]] = {}
        for call in self.llm_calls:
            bucket = by_stage.setdefault(call.stage, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
            bucket["calls"] += 1
            bucket["input_tokens"] += call.input_tokens or 0
            bucket["output_tokens"] += call.output_tokens or 0
        return {
            "trace_id": self.trace_id,
            "outcome": self.outcome,
            "outcome_group": OUTCOME_GROUPS.get(self.outcome, "error"),
            "detail": self.detail,
            "duration_ms": round(self.duration_ms, 2),
            "events": list(self.events),
            "guardrails": list(self.guardrails),
            "stages_ms": {name: round(value, 2) for name, value in self.stages_ms.items()},
            "tokens": {"input": self.input_tokens, "output": self.output_tokens, "by_stage": by_stage},
            "llm_calls": [
                {
                    "stage": call.stage,
                    "operation": call.operation,
                    "model": call.model,
                    "provider": call.provider,
                    "input_tokens": call.input_tokens,
                    "output_tokens": call.output_tokens,
                    "duration_ms": round(call.duration_ms, 2),
                    "error": call.error,
                }
                for call in self.llm_calls
            ],
        }


_RECORD: ContextVar[Optional[RequestRecord]] = ContextVar("beyondrag_request_record", default=None)
_STAGE: ContextVar[str] = ContextVar("beyondrag_stage", default="none")
_TENANT: ContextVar[str] = ContextVar("beyondrag_tenant", default="")

_SETUP_LOCK = threading.Lock()
_SETUP_DONE = False
_INSTRUMENTS: Optional["_Instruments"] = None
# Called with (tenant, input_tokens, output_tokens) after every model call that reports usage.
_USAGE_LISTENERS: List[Callable[[str, int, int], None]] = []


class _RedactingFilter(logging.Filter):
    """Mask personal data in every log line. A log line that prints a user's question is easy
    to write and easy to forget, so the mask sits on the handler, not at each call."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        masked, found = redact_pii(message)
        if found:
            record.msg, record.args = masked, None
        return True


class _Instruments:
    def __init__(self) -> None:
        meter = metrics.get_meter("beyondrag.rag")
        self.tokens = meter.create_histogram(
            "gen_ai.client.token.usage", unit="{token}",
            description="Tokens per model call, by direction, stage and tenant.",
            explicit_bucket_boundaries_advisory=_TOKEN_BUCKETS,
        )
        self.llm_duration = meter.create_histogram(
            "gen_ai.client.operation.duration", unit="s", description="Time per model call.",
            explicit_bucket_boundaries_advisory=_DURATION_BUCKETS,
        )
        self.requests = meter.create_counter(
            "rag.requests", unit="{request}", description="Questions, by outcome."
        )
        self.request_duration = meter.create_histogram(
            "rag.request.duration", unit="s", description="Time per question.",
            explicit_bucket_boundaries_advisory=_DURATION_BUCKETS,
        )
        self.stage_duration = meter.create_histogram(
            "rag.stage.duration", unit="s", description="Time per pipeline stage.",
            explicit_bucket_boundaries_advisory=_DURATION_BUCKETS,
        )
        self.events = meter.create_counter(
            "rag.events", unit="{event}", description="Things that went differently than asked, by kind."
        )
        self.guardrails = meter.create_counter(
            "rag.guardrail.decisions", unit="{decision}", description="Guardrail decisions, by guardrail and action."
        )


def setup_telemetry(*, span_processors: Sequence[Any] = (), metric_readers: Sequence[Any] = ()) -> bool:
    """Install the OpenTelemetry SDK once. Returns False when it was already installed.

    The SDK is always installed, so every request has a trace id even with no backend.
    Exporters are added only when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set. The standard
    ``OTEL_SDK_DISABLED=true`` turns spans and metrics into no-ops.
    """
    global _SETUP_DONE
    with _SETUP_LOCK:
        if _SETUP_DONE:
            return False
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "beyondrag")})
        tracer_provider = TracerProvider(resource=resource, sampler=_sampler())
        for processor in span_processors:
            tracer_provider.add_span_processor(processor)
        readers = list(metric_readers)

        root = logging.getLogger()
        if not root.handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
        elif root.level > logging.INFO or root.level == logging.NOTSET:
            root.setLevel(logging.INFO)
        # Each search writes a line at INFO in these two; the trace already has a span for it.
        for noisy in ("httpx", "elastic_transport"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip() and not _sdk_disabled():
            from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
            from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            readers.append(PeriodicExportingMetricReader(OTLPMetricExporter()))
            logger_provider = LoggerProvider(resource=resource)
            logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
            root.addHandler(LoggingHandler(level=logging.INFO, logger_provider=logger_provider))

        for handler in root.handlers:
            handler.addFilter(_RedactingFilter())
        trace.set_tracer_provider(tracer_provider)
        metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=readers))
        _SETUP_DONE = True
        return True


def _sampler() -> Any:
    """A span with no parent is kept only if it is an HTTP request, a stage or a model call.

    The health check pings Elasticsearch every few seconds, and the Elasticsearch client
    traces each ping as a trace of its own (an INTERNAL span named "ping"). Without this,
    those pings were most of what Tempo held. Anything inside a kept trace is kept.
    """
    from opentelemetry.sdk.trace.sampling import ALWAYS_ON, Decision, ParentBased, Sampler, SamplingResult

    class _RootsThatMatter(Sampler):
        def should_sample(self, parent_context, trace_id, name, kind=None, attributes=None, links=None, trace_state=None):
            if kind != SpanKind.SERVER and not name.startswith(("rag.", "chat ", "embeddings ")):
                return SamplingResult(Decision.DROP)
            return ALWAYS_ON.should_sample(parent_context, trace_id, name, kind, attributes, links, trace_state)

        def get_description(self) -> str:
            return "RootsThatMatter"

    return ParentBased(root=_RootsThatMatter())


def _sdk_disabled() -> bool:
    return os.getenv("OTEL_SDK_DISABLED", "false").strip().lower() == "true"


def _instruments() -> _Instruments:
    global _INSTRUMENTS
    if _INSTRUMENTS is None:
        setup_telemetry()
        _INSTRUMENTS = _Instruments()
    return _INSTRUMENTS


def _tracer() -> trace.Tracer:
    if not _SETUP_DONE:
        setup_telemetry()
    return trace.get_tracer("beyondrag.rag")


def add_usage_listener(listener: Callable[[str, int, int], None]) -> None:
    if listener not in _USAGE_LISTENERS:
        _USAGE_LISTENERS.append(listener)


def current_record() -> Optional[RequestRecord]:
    return _RECORD.get()


def current_tenant() -> str:
    return _TENANT.get()


def note(event: str, detail: Any = "") -> None:
    """Report that something went differently than asked: a fallback, a refusal, a skipped step.

    Safe to call anywhere. Outside a request it still counts the event and marks the span.
    """
    text = safe_text(detail, _MAX_DETAIL)
    record = _RECORD.get()
    if record is not None:
        record.events.append({"event": event, "stage": _STAGE.get(), "detail": text})
    trace.get_current_span().add_event(event, {"rag.detail": text} if text else {})
    _instruments().events.add(1, {"rag.event": event, "rag.stage": _STAGE.get(), "rag.tenant": _TENANT.get() or "none"})


def guardrail_decision(guardrail: str, action: str, reason: str = "") -> None:
    text = safe_text(reason, _MAX_DETAIL)
    record = _RECORD.get()
    if record is not None:
        record.guardrails.append({"guardrail": guardrail, "action": action, "reason": text})
    trace.get_current_span().add_event(
        "guardrail", {"rag.guardrail": guardrail, "rag.guardrail.action": action, "rag.detail": text}
    )
    _instruments().guardrails.add(
        1, {"rag.guardrail": guardrail, "rag.guardrail.action": action, "rag.tenant": _TENANT.get() or "none"}
    )


@contextmanager
def stage(name: str, *, tenant: Optional[str] = None, **attributes: Any) -> Iterator[trace.Span]:
    """A span named ``rag.<name>``. Model calls made inside it carry the stage name."""
    stage_token = _STAGE.set(name)
    tenant_token = _TENANT.set(tenant) if tenant else None
    started = time.perf_counter()
    span_attributes = {f"rag.{key}": value for key, value in attributes.items() if value is not None}
    if _TENANT.get():
        span_attributes["rag.tenant"] = _TENANT.get()
    try:
        with _tracer().start_as_current_span(f"rag.{name}", attributes=span_attributes) as span:
            yield span
    finally:
        elapsed = time.perf_counter() - started
        record = _RECORD.get()
        if record is not None:
            record.stages_ms[name] = record.stages_ms.get(name, 0.0) + elapsed * 1000.0
        _instruments().stage_duration.record(elapsed, {"rag.stage": name})
        if tenant_token is not None:
            _TENANT.reset(tenant_token)
        _STAGE.reset(stage_token)


@contextmanager
def request_scope(tenant: str, synthesis_mode: str, question: str = "") -> Iterator[RequestRecord]:
    """The root span of one question. The caller sets ``record.outcome`` before leaving."""
    record = RequestRecord(tenant=tenant, synthesis_mode=synthesis_mode)
    record_token = _RECORD.set(record)
    tenant_token = _TENANT.set(tenant)
    attributes: Dict[str, Any] = {"rag.tenant": tenant, "rag.synthesis_mode": synthesis_mode}
    if question and capture_content():
        attributes["rag.question"] = safe_text(question)
    try:
        with _tracer().start_as_current_span("rag.query", attributes=attributes) as span:
            context = span.get_span_context()
            if context.is_valid:
                record.trace_id = format(context.trace_id, "032x")
            try:
                yield record
            except Exception as exc:
                record.outcome = "error"
                record.detail = safe_text(f"{type(exc).__name__}: {exc}", _MAX_DETAIL)
                raise
            finally:
                _finish_request(record, span)
    finally:
        _TENANT.reset(tenant_token)
        _RECORD.reset(record_token)


def _finish_request(record: RequestRecord, span: trace.Span) -> None:
    record.duration_ms = (time.perf_counter() - record.started) * 1000.0
    if record.outcome not in OUTCOME_GROUPS:
        # Leaving without a label is a bug in the caller. Count it where it will be seen.
        record.detail = record.detail or f"no outcome was set (got {record.outcome!r})"
        record.outcome = "error"
    group = OUTCOME_GROUPS[record.outcome]
    if not record.detail and group != "ok":
        # The reason, in one field, so nobody has to open the trace to learn why.
        record.detail = next((item["detail"] for item in record.events if item["detail"]), "")
    span.set_attribute("rag.outcome", record.outcome)
    span.set_attribute("rag.outcome_group", group)
    span.set_attribute("gen_ai.usage.input_tokens", record.input_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", record.output_tokens)
    span.set_attribute("rag.llm_calls", len(record.llm_calls))
    if record.detail:
        span.set_attribute("rag.detail", record.detail)
    if group == "error":
        span.set_status(Status(StatusCode.ERROR, record.detail))
    labels = {
        "rag.tenant": record.tenant,
        "rag.outcome": record.outcome,
        "rag.outcome_group": group,
        "rag.synthesis_mode": record.synthesis_mode,
    }
    instruments = _instruments()
    instruments.requests.add(1, labels)
    instruments.request_duration.record(record.duration_ms / 1000.0, labels)
    # One line per question with everything needed to find and explain it.
    request_logger.log(
        logging.INFO if group in {"ok", "refused"} else logging.WARNING,
        "%s",
        json.dumps(
            {
                "event": "rag.request",
                "trace_id": record.trace_id,
                "tenant": record.tenant,
                "outcome": record.outcome,
                "outcome_group": group,
                "synthesis_mode": record.synthesis_mode,
                "duration_ms": round(record.duration_ms, 1),
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "llm_calls": len(record.llm_calls),
                "llm_errors": sum(1 for call in record.llm_calls if call.error),
                "events": [item["event"] for item in record.events],
                "guardrails": [f"{item['guardrail']}:{item['action']}" for item in record.guardrails],
                "detail": record.detail,
            },
            ensure_ascii=False,
        ),
    )


class LlmCall:
    """One model call in progress. Made by ``begin_llm_call``, closed by ``finish``."""

    def __init__(self, span: trace.Span, model: str, provider: str, operation: str = "chat") -> None:
        self.span = span
        self.model = model
        self.provider = provider
        self.operation = operation
        self.stage = _STAGE.get()
        self.tenant = _TENANT.get()
        self.record = _RECORD.get()
        self.started = time.perf_counter()
        self.input_tokens: Optional[int] = None
        self.output_tokens: Optional[int] = None
        self.completion: Optional[str] = None
        self._finished = False

    def set_usage(self, input_tokens: Any, output_tokens: Any) -> None:
        self.input_tokens = int(input_tokens) if isinstance(input_tokens, (int, float)) else None
        self.output_tokens = int(output_tokens) if isinstance(output_tokens, (int, float)) else None

    def finish(self, error: Optional[BaseException] = None) -> None:
        if self._finished:
            return
        self._finished = True
        elapsed = time.perf_counter() - self.started
        labels = {
            "gen_ai.operation.name": self.operation,
            "gen_ai.provider.name": self.provider,
            "gen_ai.request.model": self.model,
            "rag.stage": self.stage,
            "rag.tenant": self.tenant or "none",
        }
        instruments = _instruments()
        error_type = type(error).__name__ if error is not None else None
        if error is not None:
            self.span.record_exception(error)
            self.span.set_attribute("error.type", error_type)
            self.span.set_status(Status(StatusCode.ERROR, safe_text(error, _MAX_DETAIL)))
        if self.input_tokens is not None:
            self.span.set_attribute("gen_ai.usage.input_tokens", self.input_tokens)
            instruments.tokens.record(self.input_tokens, {**labels, "gen_ai.token.type": "input"})
        if self.output_tokens is not None:
            self.span.set_attribute("gen_ai.usage.output_tokens", self.output_tokens)
            instruments.tokens.record(self.output_tokens, {**labels, "gen_ai.token.type": "output"})
        if error is None and self.input_tokens is None and self.output_tokens is None:
            # A provider that reports no usage would otherwise look free on the dashboard.
            self.span.add_event("usage_missing")
            instruments.events.add(
                1, {"rag.event": "usage_missing", "rag.stage": self.stage, "rag.tenant": self.tenant or "none"}
            )
        if self.completion is not None and capture_content():
            self.span.set_attribute("gen_ai.output.messages", safe_text(self.completion))
        instruments.llm_duration.record(elapsed, {**labels, "error.type": error_type} if error_type else labels)
        self.span.end()
        if self.record is not None:
            self.record.llm_calls.append(
                LlmCallRecord(
                    stage=self.stage,
                    model=self.model,
                    provider=self.provider,
                    operation=self.operation,
                    input_tokens=self.input_tokens,
                    output_tokens=self.output_tokens,
                    duration_ms=elapsed * 1000.0,
                    error=error_type,
                )
            )
        if self.tenant and (self.input_tokens or self.output_tokens):
            for listener in list(_USAGE_LISTENERS):
                try:
                    listener(self.tenant, self.input_tokens or 0, self.output_tokens or 0)
                except Exception as exc:  # A broken listener must not fail the answer.
                    logger.warning("Token usage listener failed: %s", exc)


def begin_llm_call(
    model: str, provider: str, *, operation: str = "chat", prompt: Optional[str] = None, **request: Any
) -> LlmCall:
    """Open a ``chat <model>`` (or ``embeddings <model>``) client span under the current span."""
    provider = _PROVIDER_NAMES.get(provider, provider)
    attributes: Dict[str, Any] = {
        "gen_ai.operation.name": operation,
        "gen_ai.provider.name": provider,
        # The older name. Some backends still group by it.
        "gen_ai.system": provider,
        "gen_ai.request.model": model,
        "rag.stage": _STAGE.get(),
    }
    if _TENANT.get():
        attributes["rag.tenant"] = _TENANT.get()
    for key, value in request.items():
        if isinstance(value, (int, float, str, bool)):
            attributes[f"gen_ai.request.{key}"] = value
    if prompt is not None and capture_content():
        attributes["gen_ai.input.messages"] = safe_text(prompt)
    span = _tracer().start_span(f"{operation} {model}", kind=SpanKind.CLIENT, attributes=attributes)
    return LlmCall(span, model, provider, operation)


@contextmanager
def llm_call(
    model: str, provider: str, *, operation: str = "chat", prompt: Optional[str] = None, **request: Any
) -> Iterator[LlmCall]:
    """For a model call that does not go through LangChain. Records the error and re-raises."""
    call = begin_llm_call(model, provider, operation=operation, prompt=prompt, **request)
    try:
        yield call
    except BaseException as exc:
        call.finish(exc)
        raise
    else:
        call.finish()


class GenAiTelemetryHandler(BaseCallbackHandler):
    """Turns LangChain model callbacks into GenAI spans, token metrics and request records."""

    raise_error = False
    # Run in the caller's thread, where the stage, tenant and record context variables are set.
    run_inline = True

    def __init__(self) -> None:
        self._calls: Dict[UUID, LlmCall] = {}
        self._lock = threading.Lock()

    def _begin(self, run_id: UUID, prompt: str, metadata: Optional[Dict[str, Any]], kwargs: Dict[str, Any]) -> None:
        params = kwargs.get("invocation_params") or {}
        metadata = metadata or {}
        model = str(metadata.get("ls_model_name") or params.get("model") or params.get("model_name") or "unknown")
        provider = str(metadata.get("ls_provider") or params.get("_type") or "unknown")
        call = begin_llm_call(model, provider, prompt=prompt, temperature=metadata.get("ls_temperature"))
        with self._lock:
            self._calls[run_id] = call

    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[Any]], *, run_id: UUID,
        metadata: Optional[Dict[str, Any]] = None, **kwargs: Any,
    ) -> None:
        prompt = ""
        if capture_content():
            prompt = "\n".join(f"{getattr(m, 'type', 'message')}: {getattr(m, 'content', m)}" for group in messages for m in group)
        self._begin(run_id, prompt, metadata, kwargs)

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID,
        metadata: Optional[Dict[str, Any]] = None, **kwargs: Any,
    ) -> None:
        self._begin(run_id, "\n".join(prompts) if capture_content() else "", metadata, kwargs)

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            call = self._calls.pop(run_id, None)
        if call is None:
            return
        try:
            generation = response.generations[0][0]
            message = getattr(generation, "message", None)
            usage = getattr(message, "usage_metadata", None) or {}
            if not usage:
                usage = ((getattr(response, "llm_output", None) or {}).get("token_usage")) or {}
            call.set_usage(
                usage.get("input_tokens", usage.get("prompt_tokens")),
                usage.get("output_tokens", usage.get("completion_tokens")),
            )
            response_model = (getattr(message, "response_metadata", None) or {}).get("model")
            if response_model:
                call.span.set_attribute("gen_ai.response.model", str(response_model))
            call.completion = str(getattr(message, "content", "") or getattr(generation, "text", ""))
        except Exception as exc:  # Telemetry must never fail a model call that worked.
            logger.warning("Could not read token usage from the model response: %s", exc)
        call.finish()

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            call = self._calls.pop(run_id, None)
        if call is not None:
            call.finish(error)


_HANDLER: Optional[GenAiTelemetryHandler] = None


def genai_handler() -> GenAiTelemetryHandler:
    global _HANDLER
    if _HANDLER is None:
        _HANDLER = GenAiTelemetryHandler()
    return _HANDLER


def classify_outcome(record: RequestRecord, synthesis_mode: str, trust_report: Optional[Dict[str, Any]]) -> str:
    """Pick the one label for a finished request. The first match wins."""
    for event, outcome in (
        ("blocked_input", "blocked_input"),
        ("blocked_output", "blocked_output"),
        ("access_blocked", "access_blocked"),
        ("no_documents", "no_documents"),
        ("low_relevance", "abstained_low_relevance"),
        ("budget_exceeded", "budget_exceeded"),
        ("llm_fallback", "degraded_llm_fallback"),
        ("model_substituted", "degraded_model_substituted"),
        ("model_abstained", "abstained_model"),
    ):
        if record.has(event):
            return outcome
    if trust_report:
        if trust_report.get("abstained"):
            return "abstained_unverified"
        if trust_report.get("verdict") in {"has_unsupported_claims", "citation_problems"}:
            return "answered_with_unsupported_claims"
    return "answered" if synthesis_mode == "llm" else "answered_extractive"

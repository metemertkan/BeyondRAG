"""Keep the test suite away from the running app's index and workspace.

Some tests go through the real service. It writes to the index named by RAG_INDEX_PREFIX
and to the registry, uploads and tables under the workspace folder, and in demo mode it
resets all of them at start. Without this, a test run on a machine that also runs the app
replaces the app's documents. Set before any test module imports src.config.

Tokens in tests are signed with a secret that exists only here.
"""

import atexit
import os
import shutil
import tempfile

os.environ["RAG_INDEX_PREFIX"] = "beyondrag-pytest-kb"
# A fixed signing secret, so the suite never reads or creates workspace/.auth_secret.
os.environ["RAG_AUTH_MODE"] = "demo"
os.environ["RAG_AUTH_SECRET"] = "pytest-only-signing-secret-0123456789abcdef"

_WORKSPACE = tempfile.mkdtemp(prefix="beyondrag-pytest-")
os.environ["RAG_WORKSPACE_DIR"] = _WORKSPACE
atexit.register(shutil.rmtree, _WORKSPACE, ignore_errors=True)

# Telemetry goes to memory in tests, never to a collector, whatever the shell has set.
os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
os.environ["RAG_GUARDRAILS"] = "off"

from opentelemetry.sdk.metrics.export import InMemoryMetricReader  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

from src.rag import telemetry as _telemetry  # noqa: E402

SPAN_EXPORTER = InMemorySpanExporter()
METRIC_READER = InMemoryMetricReader()
# The first call wins, so it happens here, before any test or app import can install another.
_telemetry.setup_telemetry(span_processors=[SimpleSpanProcessor(SPAN_EXPORTER)], metric_readers=[METRIC_READER])

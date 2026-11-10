"""Keep the test suite away from the running app's Elasticsearch index.

Some tests go through the real service, which writes to the index named by
RAG_INDEX_PREFIX. Without this, a test run on a machine that also runs the app replaces
the app's documents. Set before any test module imports src.rag.profiles.

Tokens in tests are signed with a secret that exists only here.
"""

import os

os.environ["RAG_INDEX_PREFIX"] = "beyondrag-pytest-kb"
# A fixed signing secret, so the suite never reads or creates workspace/.auth_secret.
os.environ["RAG_AUTH_MODE"] = "demo"
os.environ["RAG_AUTH_SECRET"] = "pytest-only-signing-secret-0123456789abcdef"

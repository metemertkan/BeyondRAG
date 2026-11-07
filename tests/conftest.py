"""Keep the test suite away from the running app's Elasticsearch index.

Some tests go through the real service, which writes to the index named by
RAG_INDEX_PREFIX. Without this, a test run on a machine that also runs the app replaces
the app's documents. Set before any test module imports src.rag.profiles.
"""

import os

os.environ["RAG_INDEX_PREFIX"] = "beyondrag-pytest-kb"

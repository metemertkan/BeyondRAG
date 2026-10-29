#!/usr/bin/env python3
"""Launch the FastAPI RBAC RAG Knowledge Base Web Portal."""

from __future__ import annotations

import os
import uvicorn


def main() -> None:
    host = os.getenv("WEB_HOST", "0.0.0.0")
    port = int(os.getenv("WEB_PORT", "8090"))
    uvicorn.run("src.web.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()

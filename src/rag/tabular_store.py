"""Relational home for tables too large to embed row by row.

A 50,000-row sheet embedded row by row is expensive and answers "what is the total?"
badly. Instead the raw rows live here and one summary chunk goes into the vector
index. The LLM asks for data through a structured ``TableQuery`` (never raw SQL), which
is compiled to a parameterized statement with identifiers checked against the schema.

Rows are personal data like any chunk, so the store participates in ACL checks and in
GDPR erasure via ``source_key``.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Final, List, Literal, Optional, Sequence

from pydantic import BaseModel, Field

from src.rag.models import ParsedTable

MAX_RESULT_ROWS: Final[int] = 200
_OPERATORS: Final[Dict[str, str]] = {
    "eq": "=",
    "ne": "!=",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
    "contains": "LIKE",
}
_SAFE_TABLE_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9_]+$")


class TableFilter(BaseModel):
    column: str
    op: Literal["eq", "ne", "gt", "gte", "lt", "lte", "contains"] = "eq"
    value: Any


class TableQuery(BaseModel):
    """Structured tool call the LLM fills in to read a stored table."""

    table_id: str
    select: List[str] = Field(default_factory=list, description="Columns to return; empty means all.")
    filters: List[TableFilter] = Field(default_factory=list)
    aggregate: Optional[Literal["sum", "avg", "min", "max", "count"]] = None
    aggregate_column: Optional[str] = None
    group_by: Optional[str] = None
    order_by: Optional[str] = None
    descending: bool = False
    limit: int = Field(default=20, ge=1, le=MAX_RESULT_ROWS)


class TabularStore:
    """SQLite-backed table store. ``:memory:`` by default so unit tests stay isolated."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS table_catalog (
                   table_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, source_key TEXT NOT NULL,
                   name TEXT NOT NULL, columns TEXT NOT NULL, acl_tags TEXT NOT NULL)"""
        )
        self._conn.commit()

    def save_table(self, table: ParsedTable, *, doc_id: str, source_key: str, acl_tags: Sequence[str]) -> None:
        if not _SAFE_TABLE_ID_RE.match(table.table_id):
            raise ValueError(f"Unsafe table id: {table.table_id!r}")
        column_defs = ", ".join(f'"c{i}"' for i in range(len(table.columns)))
        placeholders = ", ".join("?" for _ in table.columns)
        with self._lock, self._conn:
            self._conn.execute(f'DROP TABLE IF EXISTS "{table.table_id}"')
            self._conn.execute(f'CREATE TABLE "{table.table_id}" ({column_defs})')
            self._conn.executemany(
                f'INSERT INTO "{table.table_id}" VALUES ({placeholders})',
                [[_sqlite_value(cell) for cell in row] for row in table.rows],
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO table_catalog VALUES (?, ?, ?, ?, ?, ?)",
                (
                    table.table_id,
                    doc_id,
                    source_key,
                    table.name,
                    json.dumps(table.columns, ensure_ascii=False),
                    json.dumps(list(acl_tags)),
                ),
            )

    def describe(self, table_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT table_id, doc_id, source_key, name, columns, acl_tags FROM table_catalog WHERE table_id = ?",
                (table_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "table_id": row[0],
            "doc_id": row[1],
            "source_key": row[2],
            "name": row[3],
            "columns": json.loads(row[4]),
            "acl_tags": json.loads(row[5]),
        }

    def query(self, query: TableQuery, effective_acl_tags: Sequence[str]) -> Dict[str, Any]:
        """Run a structured query. Raises ``PermissionError`` before touching any row."""
        meta = self.describe(query.table_id)
        if meta is None:
            raise KeyError(f"Unknown table: {query.table_id}")
        if not set(meta["acl_tags"]).intersection(effective_acl_tags):
            raise PermissionError(f"Role cannot read table {query.table_id}")

        columns: List[str] = meta["columns"]

        def ident(name: str) -> str:
            if name not in columns:
                raise ValueError(f"Unknown column {name!r}; table has {columns}")
            return f'"c{columns.index(name)}"'

        params: List[Any] = []
        where: List[str] = []
        for flt in query.filters:
            if flt.op == "contains":
                where.append(f"{ident(flt.column)} LIKE ?")
                params.append(f"%{flt.value}%")
            else:
                where.append(f"{ident(flt.column)} {_OPERATORS[flt.op]} ?")
                params.append(flt.value)

        if query.aggregate:
            target = "*" if query.aggregate == "count" and not query.aggregate_column else None
            if target is None:
                if not query.aggregate_column:
                    raise ValueError(f"{query.aggregate} needs aggregate_column")
                target = ident(query.aggregate_column)
            label = f"{query.aggregate}({query.aggregate_column or '*'})"
            selected = [f"{query.aggregate.upper()}({target})"]
            result_columns = [label]
            if query.group_by:
                selected.insert(0, ident(query.group_by))
                result_columns.insert(0, query.group_by)
        else:
            names = query.select or columns
            selected = [ident(name) for name in names]
            result_columns = list(names)

        sql = f'SELECT {", ".join(selected)} FROM "{query.table_id}"'
        if where:
            sql += " WHERE " + " AND ".join(where)
        if query.aggregate and query.group_by:
            sql += f" GROUP BY {ident(query.group_by)}"
        if query.order_by:
            sql += f" ORDER BY {ident(query.order_by)} {'DESC' if query.descending else 'ASC'}"
        sql += " LIMIT ?"
        params.append(query.limit)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return {
            "table_id": query.table_id,
            "name": meta["name"],
            "columns": result_columns,
            "rows": [list(row) for row in rows],
        }

    def table_ids_for(self, *, doc_id: str | None = None, source_key: str | None = None) -> List[str]:
        column, value = ("doc_id", doc_id) if doc_id is not None else ("source_key", source_key)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT table_id FROM table_catalog WHERE {column} = ?", (value,)
            ).fetchall()
        return [row[0] for row in rows]

    def delete_tables(self, *, doc_id: str | None = None, source_key: str | None = None) -> int:
        """Drop tables and catalog rows, then VACUUM so erased rows leave the file."""
        table_ids = self.table_ids_for(doc_id=doc_id, source_key=source_key)
        if not table_ids:
            return 0
        with self._lock, self._conn:
            for table_id in table_ids:
                self._conn.execute(f'DROP TABLE IF EXISTS "{table_id}"')
                self._conn.execute("DELETE FROM table_catalog WHERE table_id = ?", (table_id,))
        with self._lock:
            self._conn.execute("VACUUM")
        return len(table_ids)

    def update_acl(self, doc_id: str, acl_tags: Sequence[str]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE table_catalog SET acl_tags = ? WHERE doc_id = ?",
                (json.dumps(list(acl_tags)), doc_id),
            )


def _sqlite_value(value: Any) -> Any:
    if value is None or isinstance(value, (int, float, str)):
        return value
    return str(value)

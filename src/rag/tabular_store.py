"""Relational home for tables too large to embed row by row.

A 50,000-row sheet embedded row by row is expensive and answers "what is the total?"
badly. Instead the raw rows live here and one summary chunk goes into the vector
index. The LLM asks for data through a structured ``TableQuery`` (never raw SQL), which
is compiled to a parameterized statement with identifiers checked against the schema.

Rows are personal data like any chunk, so the store participates in ACL checks and in
GDPR erasure via ``source_key``.

Tables belong to a tenant. The table id comes from the file's bytes, so two tenants who
upload the same spreadsheet get the same id. Each tenant's rows live in their own
physical table, and the catalog key is (tenant, table_id). Asking for another tenant's
table gives "unknown table", the same answer as for a table that does not exist.
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
from src.rag.tenancy import DEFAULT_TENANT, normalize_tenant, tenant_digest

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
        with self._lock, self._conn:
            self._ensure_catalog()

    def _ensure_catalog(self) -> None:
        """Create the catalog, or move a pre-tenant catalog into the ``default`` tenant."""
        columns = [row[1] for row in self._conn.execute("PRAGMA table_info(table_catalog)").fetchall()]
        legacy = bool(columns) and "tenant" not in columns
        if legacy:
            self._conn.execute("ALTER TABLE table_catalog RENAME TO table_catalog_pre_tenant")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS table_catalog (
                   tenant TEXT NOT NULL, table_id TEXT NOT NULL, physical_id TEXT NOT NULL UNIQUE,
                   doc_id TEXT NOT NULL, source_key TEXT NOT NULL,
                   name TEXT NOT NULL, columns TEXT NOT NULL, acl_tags TEXT NOT NULL,
                   PRIMARY KEY (tenant, table_id))"""
        )
        if legacy:
            self._conn.execute(
                """INSERT INTO table_catalog (tenant, table_id, physical_id, doc_id, source_key, name, columns, acl_tags)
                   SELECT ?, table_id, table_id, doc_id, source_key, name, columns, acl_tags FROM table_catalog_pre_tenant""",
                (DEFAULT_TENANT,),
            )
            self._conn.execute("DROP TABLE table_catalog_pre_tenant")

    @staticmethod
    def _physical_id(table_id: str, tenant: str) -> str:
        """Name of the SQLite table holding one tenant's rows for a logical table id."""
        return table_id if tenant == DEFAULT_TENANT else f"tn_{tenant_digest(tenant)}_{table_id}"

    def save_table(
        self,
        table: ParsedTable,
        *,
        doc_id: str,
        source_key: str,
        acl_tags: Sequence[str],
        tenant: str | None = None,
    ) -> None:
        if not _SAFE_TABLE_ID_RE.match(table.table_id):
            raise ValueError(f"Unsafe table id: {table.table_id!r}")
        scope = normalize_tenant(tenant)
        physical = self._physical_id(table.table_id, scope)
        column_defs = ", ".join(f'"c{i}"' for i in range(len(table.columns)))
        placeholders = ", ".join("?" for _ in table.columns)
        with self._lock, self._conn:
            holder = self._conn.execute(
                "SELECT tenant, table_id FROM table_catalog WHERE physical_id = ?", (physical,)
            ).fetchone()
            if holder is not None and tuple(holder) != (scope, table.table_id):
                # Never drop a physical table that another catalog entry points at.
                raise ValueError(f"Table id is not available: {table.table_id!r}")
            self._conn.execute(f'DROP TABLE IF EXISTS "{physical}"')
            self._conn.execute(f'CREATE TABLE "{physical}" ({column_defs})')
            self._conn.executemany(
                f'INSERT INTO "{physical}" VALUES ({placeholders})',
                [[_sqlite_value(cell) for cell in row] for row in table.rows],
            )
            self._conn.execute(
                """INSERT OR REPLACE INTO table_catalog
                   (tenant, table_id, physical_id, doc_id, source_key, name, columns, acl_tags)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    scope,
                    table.table_id,
                    physical,
                    doc_id,
                    source_key,
                    table.name,
                    json.dumps(table.columns, ensure_ascii=False),
                    json.dumps(list(acl_tags)),
                ),
            )

    def describe(self, table_id: str, *, tenant: str | None = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                """SELECT table_id, doc_id, source_key, name, columns, acl_tags, tenant, physical_id
                   FROM table_catalog WHERE tenant = ? AND table_id = ?""",
                (normalize_tenant(tenant), table_id),
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
            "tenant": row[6],
            "physical_id": row[7],
        }

    def query(self, query: TableQuery, effective_acl_tags: Sequence[str], *, tenant: str | None = None) -> Dict[str, Any]:
        """Run a structured query. Raises ``PermissionError`` before touching any row.

        The table is looked up inside the caller's tenant. Another tenant's table raises
        ``KeyError`` like a missing one, so the error does not confirm that it exists.
        """
        meta = self.describe(query.table_id, tenant=tenant)
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

        sql = f'SELECT {", ".join(selected)} FROM "{meta["physical_id"]}"'
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

    def _entries_for(self, *, doc_id: str | None, source_key: str | None, tenant: str) -> List[tuple[str, str]]:
        column, value = ("doc_id", doc_id) if doc_id is not None else ("source_key", source_key)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT table_id, physical_id FROM table_catalog WHERE tenant = ? AND {column} = ?", (tenant, value)
            ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def table_ids_for(
        self, *, doc_id: str | None = None, source_key: str | None = None, tenant: str | None = None
    ) -> List[str]:
        return [
            table_id
            for table_id, _ in self._entries_for(doc_id=doc_id, source_key=source_key, tenant=normalize_tenant(tenant))
        ]

    def _drop(self, tenant: str, entries: Sequence[tuple[str, str]]) -> int:
        if not entries:
            return 0
        with self._lock, self._conn:
            for table_id, physical in entries:
                self._conn.execute(f'DROP TABLE IF EXISTS "{physical}"')
                self._conn.execute("DELETE FROM table_catalog WHERE tenant = ? AND table_id = ?", (tenant, table_id))
        with self._lock:
            self._conn.execute("VACUUM")
        return len(entries)

    def delete_tables(
        self, *, doc_id: str | None = None, source_key: str | None = None, tenant: str | None = None
    ) -> int:
        """Drop one tenant's tables and catalog rows, then VACUUM so erased rows leave the file."""
        scope = normalize_tenant(tenant)
        return self._drop(scope, self._entries_for(doc_id=doc_id, source_key=source_key, tenant=scope))

    def delete_tenant(self, tenant: str) -> int:
        """Offboarding: drop every table a tenant stored."""
        scope = normalize_tenant(tenant)
        with self._lock:
            rows = self._conn.execute(
                "SELECT table_id, physical_id FROM table_catalog WHERE tenant = ?", (scope,)
            ).fetchall()
        return self._drop(scope, [(row[0], row[1]) for row in rows])

    def update_acl(self, doc_id: str, acl_tags: Sequence[str], *, tenant: str | None = None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE table_catalog SET acl_tags = ? WHERE tenant = ? AND doc_id = ?",
                (json.dumps(list(acl_tags)), normalize_tenant(tenant), doc_id),
            )


def _sqlite_value(value: Any) -> Any:
    if value is None or isinstance(value, (int, float, str)):
        return value
    return str(value)

"""AST-backed, allowlisted SQL validation for the enterprise operations schema."""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

BUSINESS_TABLE_COLUMNS: dict[str, frozenset[str]] = {
    "products": frozenset(
        {"product_id", "sku", "product_name", "category", "status", "safety_stock"}
    ),
    "suppliers": frozenset({"supplier_id", "supplier_name", "status"}),
    "sales_orders": frozenset({"sales_order_id", "order_number", "order_date", "status"}),
    "sales_order_items": frozenset(
        {"sales_order_item_id", "sales_order_id", "product_id", "quantity", "unit_price"}
    ),
    "inventory_snapshots": frozenset(
        {"inventory_snapshot_id", "product_id", "snapshot_date", "on_hand_quantity"}
    ),
    "purchase_orders": frozenset(
        {
            "purchase_order_id",
            "purchase_order_number",
            "supplier_id",
            "product_id",
            "order_date",
            "status",
            "quantity",
            "unit_cost",
            "promised_delivery_date",
            "actual_delivery_date",
        }
    ),
}

_WRITE_ROOTS = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Alter,
    exp.Drop,
    exp.TruncateTable,
    exp.Grant,
    exp.Revoke,
    exp.Command,
)
_SYSTEM_SCHEMAS = frozenset({"information_schema", "mysql", "performance_schema", "sys"})
_DANGEROUS_FUNCTIONS = frozenset({"LOAD_FILE", "SLEEP", "BENCHMARK"})


@dataclass(frozen=True)
class SQLGuardDecision:
    """Outcome of parsing, authorization, and resource-bound SQL validation."""

    allowed: bool
    normalized_sql: str | None
    rejection_code: str | None
    accessed_tables: list[str]


class SQLGuard:
    """Permit exactly one read-only, allowlisted MySQL query with a bounded LIMIT."""

    def __init__(
        self, *, max_rows: int, table_columns: dict[str, frozenset[str]] | None = None
    ) -> None:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        self._max_rows = max_rows
        self._table_columns = table_columns or BUSINESS_TABLE_COLUMNS

    def validate(self, sql: str) -> SQLGuardDecision:
        """Parse and authorize SQL without executing or semantically rewriting it."""
        normalized_source = sql.lower()
        if "into outfile" in normalized_source or "into dumpfile" in normalized_source:
            return self._reject("dangerous_function_not_allowed")
        try:
            statements = [statement for statement in sqlglot.parse(sql, read="mysql") if statement]
        except sqlglot.errors.ParseError:
            return self._reject("sql_parse_failed")
        if len(statements) != 1:
            return self._reject("multiple_statements_not_allowed")

        expression = statements[0]
        if isinstance(expression, _WRITE_ROOTS):
            return self._reject("write_statement_not_allowed")
        if not isinstance(expression, exp.Query):
            return self._reject("write_statement_not_allowed")
        if any(not select.expressions for select in expression.find_all(exp.Select)):
            return self._reject("sql_parse_failed")
        if expression.find(exp.Lock):
            return self._reject("locking_read_not_allowed")
        if expression.find(exp.Into):
            return self._reject("dangerous_function_not_allowed")
        if self._contains_dangerous_function(expression):
            return self._reject("dangerous_function_not_allowed")

        cte_names = {cte.alias_or_name.lower() for cte in expression.find_all(exp.CTE)}
        accessed_tables: set[str] = set()
        for table in expression.find_all(exp.Table):
            schema = (table.db or table.catalog or "").lower()
            if schema in _SYSTEM_SCHEMAS:
                return self._reject("system_schema_not_allowed")
            table_name = table.name.lower()
            if table_name in cte_names:
                continue
            if table_name not in self._table_columns:
                return self._reject("unauthorized_table")
            accessed_tables.add(table_name)

        if self._has_disallowed_wildcard(expression):
            return self._reject("wildcard_not_allowed")
        column_error = self._validate_columns(expression, cte_names)
        if column_error is not None:
            return self._reject(column_error)
        limit_error = self._apply_or_validate_limit(expression)
        if limit_error is not None:
            return self._reject(limit_error)
        return SQLGuardDecision(
            allowed=True,
            normalized_sql=expression.sql(dialect="mysql"),
            rejection_code=None,
            accessed_tables=sorted(accessed_tables),
        )

    def _validate_columns(self, expression: exp.Expression, cte_names: set[str]) -> str | None:
        for scope in traverse_scope(expression):
            aliases: dict[str, frozenset[str]] = {}
            for table in scope.tables:
                name = table.name.lower()
                if name in cte_names:
                    continue
                allowed_columns = self._table_columns.get(name)
                if allowed_columns is not None:
                    aliases[table.alias_or_name.lower()] = allowed_columns
            columns = list(scope.columns)
            having = scope.expression.args.get("having")
            if having is not None:
                columns.extend(having.find_all(exp.Column))
            for column in columns:
                if isinstance(column.this, exp.Star):
                    continue
                column_name = column.name.lower()
                qualifier = column.table.lower()
                if qualifier:
                    allowed = aliases.get(qualifier)
                    if allowed is not None and column_name not in allowed:
                        return "unauthorized_column"
                    continue
                if not aliases:
                    continue
                if not any(column_name in allowed for allowed in aliases.values()):
                    return "unauthorized_column"
        return None

    def _has_disallowed_wildcard(self, expression: exp.Expression) -> bool:
        return any(star.find_ancestor(exp.Count) is None for star in expression.find_all(exp.Star))

    def _contains_dangerous_function(self, expression: exp.Expression) -> bool:
        for function in expression.find_all(exp.Func):
            function_name = (
                function.name.upper()
                if isinstance(function, exp.Anonymous)
                else function.sql_name().upper()
            )
            if function_name in _DANGEROUS_FUNCTIONS:
                return True
        return False

    def _apply_or_validate_limit(self, expression: exp.Query) -> str | None:
        limit = expression.args.get("limit")
        if limit is None:
            expression.set("limit", exp.Limit(expression=exp.Literal.number(self._max_rows)))
            return None
        value = limit.expression
        if not isinstance(value, exp.Literal) or not value.is_int:
            return "limit_exceeded"
        if int(value.this) > self._max_rows:
            return "limit_exceeded"
        return None

    @staticmethod
    def _reject(code: str) -> SQLGuardDecision:
        return SQLGuardDecision(
            allowed=False,
            normalized_sql=None,
            rejection_code=code,
            accessed_tables=[],
        )

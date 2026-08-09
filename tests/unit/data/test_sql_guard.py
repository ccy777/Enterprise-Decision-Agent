from __future__ import annotations

import pytest

from decision_agent.data.sql_guard import SQLGuard

pytestmark = pytest.mark.offline_integration


@pytest.fixture
def guard() -> SQLGuard:
    return SQLGuard(max_rows=25)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT product_id, product_name FROM products",
        "SELECT p.product_name, SUM(i.quantity) AS quantity FROM products AS p "
        "JOIN sales_order_items AS i ON p.product_id = i.product_id GROUP BY p.product_name",
        "WITH monthly AS (SELECT product_id, SUM(quantity) AS quantity FROM sales_order_items "
        "GROUP BY product_id) SELECT p.product_name, monthly.quantity FROM products AS p "
        "JOIN monthly ON p.product_id = monthly.product_id",
        "SELECT product_id FROM (SELECT product_id FROM products) AS selected_products",
        "SELECT p.product_name FROM products AS p WHERE p.product_id IN "
        "(SELECT i.product_id FROM sales_order_items AS i)",
        "SELECT product_name FROM products UNION ALL SELECT product_name FROM products",
        "SELECT COUNT(*) AS product_count FROM products",
        "SELECT product_name FROM products LIMIT 10",
    ],
)
def test_guard_allows_read_only_query_forms(guard: SQLGuard, sql: str) -> None:
    decision = guard.validate(sql)
    assert decision.allowed is True
    assert decision.rejection_code is None


def test_guard_adds_default_limit_without_changing_the_query_shape(guard: SQLGuard) -> None:
    decision = guard.validate("SELECT product_name FROM products")
    assert decision.normalized_sql == "SELECT product_name FROM products LIMIT 25"


def test_guard_adds_limit_only_to_the_top_level_cte_query(guard: SQLGuard) -> None:
    decision = guard.validate(
        "WITH selected_products AS (SELECT product_id FROM products) "
        "SELECT product_id FROM selected_products"
    )
    assert decision.normalized_sql is not None
    assert decision.normalized_sql.endswith("SELECT product_id FROM selected_products LIMIT 25")
    assert "FROM products LIMIT" not in decision.normalized_sql


def test_guard_preserves_a_bounded_offset_limit(guard: SQLGuard) -> None:
    decision = guard.validate("SELECT product_name FROM products LIMIT 5 OFFSET 1000")
    assert decision.allowed is True
    assert decision.normalized_sql == "SELECT product_name FROM products LIMIT 5 OFFSET 1000"


@pytest.mark.parametrize(
    ("sql", "code"),
    [
        ("SELECT * FROM products", "wildcard_not_allowed"),
        ("SELECT p.* FROM products AS p", "wildcard_not_allowed"),
        ("SELECT product_name FROM products LIMIT 26", "limit_exceeded"),
        ("SELECT product_name FROM products LIMIT ?", "limit_exceeded"),
        ("INSERT INTO products(product_id) VALUES ('P900')", "write_statement_not_allowed"),
        ("UPDATE products SET product_name = 'x'", "write_statement_not_allowed"),
        ("DELETE FROM products", "write_statement_not_allowed"),
        ("DROP TABLE products", "write_statement_not_allowed"),
        ("SET SESSION MAX_EXECUTION_TIME = 999999", "write_statement_not_allowed"),
        (
            "SELECT product_name FROM products; DELETE FROM products",
            "multiple_statements_not_allowed",
        ),
        (
            "SELECT product_id INTO OUTFILE '/tmp/data' FROM products",
            "dangerous_function_not_allowed",
        ),
        (
            "SELECT product_id INTO DUMPFILE '/tmp/data' FROM products",
            "dangerous_function_not_allowed",
        ),
        ("SELECT LOAD_FILE('/etc/passwd') FROM products", "dangerous_function_not_allowed"),
        ("SELECT SLEEP(1) FROM products", "dangerous_function_not_allowed"),
        (
            "/* comment */ SeLeCt SLeEp(1) FrOm products",
            "dangerous_function_not_allowed",
        ),
        ("SELECT BENCHMARK(1, 1) FROM products", "dangerous_function_not_allowed"),
        ("SELECT user FROM mysql.user", "system_schema_not_allowed"),
        ("SELECT product_name FROM unknown_table", "unauthorized_table"),
        ("SELECT secret_cost FROM products", "unauthorized_column"),
        ("SELECT product_name FROM products FOR UPDATE", "locking_read_not_allowed"),
        ("SELECT product_name FROM products FOR SHARE", "locking_read_not_allowed"),
        ("SELECT FROM products", "sql_parse_failed"),
    ],
)
def test_guard_rejects_unsafe_or_unauthorized_sql(guard: SQLGuard, sql: str, code: str) -> None:
    decision = guard.validate(sql)
    assert decision.allowed is False
    assert decision.rejection_code == code


def test_guard_validates_columns_below_ctes_and_subqueries(guard: SQLGuard) -> None:
    decision = guard.validate(
        "WITH selected_products AS (SELECT secret_cost FROM products) "
        "SELECT secret_cost FROM selected_products"
    )
    assert decision.rejection_code == "unauthorized_column"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT product_name FROM products WHERE secret_cost > 0",
        "SELECT p.product_name FROM products AS p JOIN sales_order_items AS i "
        "ON p.secret_cost = i.quantity",
        "SELECT product_name FROM products GROUP BY secret_cost",
        "SELECT product_name FROM products GROUP BY product_name HAVING SUM(secret_cost) > 0",
        "SELECT product_name FROM products ORDER BY secret_cost",
        "SELECT ABS(secret_cost) FROM products",
    ],
)
def test_guard_rejects_unauthorized_columns_in_every_expression_context(
    guard: SQLGuard, sql: str
) -> None:
    assert guard.validate(sql).rejection_code == "unauthorized_column"


def test_guard_reports_only_physical_business_tables(guard: SQLGuard) -> None:
    decision = guard.validate(
        "WITH item_totals AS (SELECT product_id, SUM(quantity) AS quantity "
        "FROM sales_order_items GROUP BY product_id) "
        "SELECT p.product_name, item_totals.quantity FROM products AS p "
        "JOIN item_totals ON p.product_id = item_totals.product_id"
    )
    assert decision.accessed_tables == ["products", "sales_order_items"]

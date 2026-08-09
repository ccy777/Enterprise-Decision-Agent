from __future__ import annotations

from pathlib import Path


def test_schema_and_seed_cover_the_fixed_business_model() -> None:
    schema = Path("docker/mysql/init/01-schema.sql").read_text(encoding="utf-8")
    seed = Path("docker/mysql/init/02-seed.sql").read_text(encoding="utf-8")
    for table in (
        "products",
        "suppliers",
        "sales_orders",
        "sales_order_items",
        "inventory_snapshots",
        "purchase_orders",
    ):
        assert f"CREATE TABLE {table}" in schema
        assert f"INSERT INTO {table}" in seed
    assert "DECIMAL(12, 2)" in schema
    assert "FOREIGN KEY" in schema
    assert schema.startswith("SET NAMES utf8mb4;")
    assert seed.startswith("SET NAMES utf8mb4;")
    assert "'cancelled'" in seed
    assert "'partially_delivered'" in seed


def test_compose_pins_mysql_and_creates_a_select_only_application_account() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    readonly_init = Path("docker/mysql/init/03-create-readonly-user.sh").read_text(encoding="utf-8")
    assert "mysql:8.4.4" in compose
    assert "healthcheck:" in compose
    assert "mysql_data" in compose
    assert "GRANT SELECT ON enterprise_operations.*" in readonly_init
    for privilege in ("INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "GRANT"):
        assert f"GRANT {privilege}" not in readonly_init

SET NAMES utf8mb4;
USE enterprise_operations;

CREATE TABLE products (
    product_id VARCHAR(20) PRIMARY KEY,
    sku VARCHAR(40) NOT NULL UNIQUE,
    product_name VARCHAR(120) NOT NULL,
    category VARCHAR(60) NOT NULL,
    status VARCHAR(20) NOT NULL,
    safety_stock INT NOT NULL,
    CONSTRAINT chk_products_status CHECK (status IN ('active', 'discontinued')),
    CONSTRAINT chk_products_safety_stock CHECK (safety_stock >= 0)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE suppliers (
    supplier_id VARCHAR(20) PRIMARY KEY,
    supplier_name VARCHAR(120) NOT NULL,
    status VARCHAR(20) NOT NULL,
    CONSTRAINT chk_suppliers_status CHECK (status IN ('active', 'inactive'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE sales_orders (
    sales_order_id VARCHAR(20) PRIMARY KEY,
    order_number VARCHAR(40) NOT NULL UNIQUE,
    order_date DATE NOT NULL,
    status VARCHAR(20) NOT NULL,
    CONSTRAINT chk_sales_orders_status CHECK (status IN ('confirmed', 'shipped', 'completed', 'cancelled'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE sales_order_items (
    sales_order_item_id VARCHAR(20) PRIMARY KEY,
    sales_order_id VARCHAR(20) NOT NULL,
    product_id VARCHAR(20) NOT NULL,
    quantity INT NOT NULL,
    unit_price DECIMAL(12, 2) NOT NULL,
    CONSTRAINT fk_sales_order_items_order FOREIGN KEY (sales_order_id) REFERENCES sales_orders(sales_order_id),
    CONSTRAINT fk_sales_order_items_product FOREIGN KEY (product_id) REFERENCES products(product_id),
    CONSTRAINT chk_sales_order_items_quantity CHECK (quantity > 0),
    CONSTRAINT chk_sales_order_items_price CHECK (unit_price >= 0)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE inventory_snapshots (
    inventory_snapshot_id VARCHAR(20) PRIMARY KEY,
    product_id VARCHAR(20) NOT NULL,
    snapshot_date DATE NOT NULL,
    on_hand_quantity INT NOT NULL,
    CONSTRAINT fk_inventory_snapshots_product FOREIGN KEY (product_id) REFERENCES products(product_id),
    CONSTRAINT uq_inventory_snapshot UNIQUE (product_id, snapshot_date),
    CONSTRAINT chk_inventory_snapshot_quantity CHECK (on_hand_quantity >= 0)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE purchase_orders (
    purchase_order_id VARCHAR(20) PRIMARY KEY,
    purchase_order_number VARCHAR(40) NOT NULL UNIQUE,
    supplier_id VARCHAR(20) NOT NULL,
    product_id VARCHAR(20) NOT NULL,
    order_date DATE NOT NULL,
    status VARCHAR(30) NOT NULL,
    quantity INT NOT NULL,
    unit_cost DECIMAL(12, 2) NOT NULL,
    promised_delivery_date DATE NOT NULL,
    actual_delivery_date DATE NULL,
    CONSTRAINT fk_purchase_orders_supplier FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id),
    CONSTRAINT fk_purchase_orders_product FOREIGN KEY (product_id) REFERENCES products(product_id),
    CONSTRAINT chk_purchase_orders_status CHECK (status IN ('ordered', 'partially_delivered', 'delivered', 'cancelled')),
    CONSTRAINT chk_purchase_orders_quantity CHECK (quantity > 0),
    CONSTRAINT chk_purchase_orders_cost CHECK (unit_cost >= 0)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE INDEX idx_sales_orders_date_status ON sales_orders(order_date, status);
CREATE INDEX idx_sales_order_items_product ON sales_order_items(product_id);
CREATE INDEX idx_inventory_snapshots_product_date ON inventory_snapshots(product_id, snapshot_date);
CREATE INDEX idx_purchase_orders_status_date ON purchase_orders(status, order_date);
CREATE INDEX idx_purchase_orders_supplier ON purchase_orders(supplier_id);

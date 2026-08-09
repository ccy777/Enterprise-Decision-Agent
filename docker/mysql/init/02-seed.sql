SET NAMES utf8mb4;
USE enterprise_operations;

INSERT INTO products (product_id, sku, product_name, category, status, safety_stock) VALUES
    ('P100', 'PUMP-A1', 'Aster 工业泵', '泵', 'active', 50),
    ('P200', 'VALVE-B1', 'Boreal 控制阀', '阀门', 'active', 80),
    ('P300', 'SENSOR-C1', 'Cirrus 传感器', '传感器', 'active', 100),
    ('P400', 'DRIVE-D1', 'Delta 驱动器', '驱动器', 'active', 40),
    ('P500', 'FILTER-E1', 'Echo 过滤器', '过滤器', 'discontinued', 30),
    ('P600', 'CABLE-F1', 'Flux 工业线缆', '线缆', 'active', 120),
    ('P700', 'METER-G1', 'Grove 流量计', '仪表', 'active', 60),
    ('P800', 'SEAL-H1', 'Harbor 密封件', '密封件', 'active', 200);

INSERT INTO suppliers (supplier_id, supplier_name, status) VALUES
    ('S100', '安海制造', 'active'),
    ('S200', '北辰供应', 'active'),
    ('S300', '长江工业', 'active'),
    ('S400', '东方零部件', 'active');

INSERT INTO sales_orders (sales_order_id, order_number, order_date, status) VALUES
    ('SO1001', 'SO-2026-0401', '2026-04-05', 'confirmed'),
    ('SO1002', 'SO-2026-0402', '2026-04-22', 'shipped'),
    ('SO1003', 'SO-2026-0501', '2026-05-03', 'completed'),
    ('SO1004', 'SO-2026-0502', '2026-05-10', 'confirmed'),
    ('SO1005', 'SO-2026-0503', '2026-05-20', 'cancelled'),
    ('SO1006', 'SO-2026-0504', '2026-05-25', 'shipped'),
    ('SO1007', 'SO-2026-0601', '2026-06-02', 'completed'),
    ('SO1008', 'SO-2026-0602', '2026-06-10', 'confirmed'),
    ('SO1009', 'SO-2026-0603', '2026-06-18', 'shipped'),
    ('SO1010', 'SO-2026-0604', '2026-06-24', 'cancelled');

INSERT INTO sales_order_items (sales_order_item_id, sales_order_id, product_id, quantity, unit_price) VALUES
    ('SOI1001', 'SO1001', 'P100', 80, 100.00),
    ('SOI1002', 'SO1001', 'P200', 40, 80.00),
    ('SOI1003', 'SO1002', 'P300', 60, 70.00),
    ('SOI1004', 'SO1002', 'P400', 20, 150.00),
    ('SOI1005', 'SO1003', 'P100', 120, 100.00),
    ('SOI1006', 'SO1004', 'P200', 60, 80.00),
    ('SOI1007', 'SO1004', 'P600', 150, 25.00),
    ('SOI1008', 'SO1005', 'P300', 90, 70.00),
    ('SOI1009', 'SO1006', 'P400', 30, 150.00),
    ('SOI1010', 'SO1007', 'P100', 40, 100.00),
    ('SOI1011', 'SO1008', 'P200', 40, 80.00),
    ('SOI1012', 'SO1008', 'P300', 100, 70.00),
    ('SOI1013', 'SO1009', 'P600', 100, 25.00),
    ('SOI1014', 'SO1010', 'P100', 50, 100.00);

INSERT INTO inventory_snapshots (inventory_snapshot_id, product_id, snapshot_date, on_hand_quantity) VALUES
    ('INV1001', 'P100', '2026-05-31', 65), ('INV1002', 'P100', '2026-06-30', 40),
    ('INV1003', 'P200', '2026-05-31', 95), ('INV1004', 'P200', '2026-06-30', 85),
    ('INV1005', 'P300', '2026-05-31', 130), ('INV1006', 'P300', '2026-06-30', 90),
    ('INV1007', 'P400', '2026-05-31', 55), ('INV1008', 'P400', '2026-06-30', 55),
    ('INV1009', 'P500', '2026-05-31', 35),
    ('INV1011', 'P600', '2026-05-31', 150), ('INV1012', 'P600', '2026-06-30', 110),
    ('INV1013', 'P700', '2026-05-31', 75), ('INV1014', 'P700', '2026-06-30', 75),
    ('INV1015', 'P800', '2026-05-31', 250), ('INV1016', 'P800', '2026-06-30', 250);

INSERT INTO purchase_orders (
    purchase_order_id, purchase_order_number, supplier_id, product_id, order_date, status,
    quantity, unit_cost, promised_delivery_date, actual_delivery_date
) VALUES
    ('PO1001', 'PO-2026-0401', 'S100', 'P100', '2026-04-01', 'delivered', 100, 65.00, '2026-04-20', '2026-04-19'),
    ('PO1002', 'PO-2026-0501', 'S100', 'P200', '2026-05-01', 'delivered', 200, 50.00, '2026-05-20', '2026-05-20'),
    ('PO1003', 'PO-2026-0502', 'S200', 'P300', '2026-05-03', 'delivered', 150, 40.00, '2026-05-18', '2026-05-22'),
    ('PO1004', 'PO-2026-0601', 'S200', 'P600', '2026-06-01', 'delivered', 300, 15.00, '2026-06-20', '2026-06-18'),
    ('PO1005', 'PO-2026-0602', 'S300', 'P400', '2026-06-05', 'ordered', 80, 90.00, '2026-06-25', NULL),
    ('PO1006', 'PO-2026-0503', 'S300', 'P100', '2026-05-06', 'delivered', 60, 64.00, '2026-05-25', '2026-05-28'),
    ('PO1007', 'PO-2026-0603', 'S400', 'P800', '2026-06-08', 'partially_delivered', 500, 5.00, '2026-06-28', NULL),
    ('PO1008', 'PO-2026-0402', 'S400', 'P700', '2026-04-11', 'cancelled', 100, 30.00, '2026-04-30', NULL);

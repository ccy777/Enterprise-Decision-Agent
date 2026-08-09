"""Canonical enterprise-operations definitions shared by data-facing components."""

from __future__ import annotations

BUSINESS_DEFINITIONS: dict[str, str] = {
    "effective_sales": (
        "有效销售订单状态为 confirmed、shipped、completed; cancelled 不计入。"
        "销售额为 sales_order_items.quantity * unit_price。"
    ),
    "effective_purchase_amount": (
        "有效采购订单状态为 ordered、partially_delivered、delivered; cancelled 不计入。"
        "采购金额为 quantity * unit_cost。"
    ),
    "current_inventory": (
        "当前库存按每个产品各自 inventory_snapshots.snapshot_date 的最大值选择; "
        "库存风险仅在 on_hand_quantity < safety_stock 时成立。"
    ),
    "on_time_delivery_rate": (
        "准时交付率只统计 status 为 delivered 且 actual_delivery_date 非空的采购订单; "
        "actual_delivery_date <= promised_delivery_date 为准时; 零交付样本为 NULL, 不是 0%。"
    ),
    "not_delivered": (
        "未交付为 status 属于 ordered 或 partially_delivered 且 actual_delivery_date 为 NULL。"
    ),
    "natural_month": "自然月区间为含起始日、不含下一月起始日, 业务日期使用 Asia/Shanghai 日历日。",
}

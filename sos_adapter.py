from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from config import settings


# Plug your real SOS API logic here.
# The rest of the app expects a list of orders with this shape:
# {
#   "so_number": "2026-1045",
#   "customer_name": "Nabtesco",
#   "order_date": date,
#   "status": "Open" | "Partial" | "Shipped" | "Closed",
#   "shipped_flag": bool,
#   "active": bool,
#   "source_last_modified": datetime,
#   "source_hash": str,
#   "lines": [
#       {
#           "line_no": 1,
#           "item_number": "130-000101",
#           "description": "TR-302 Edge",
#           "qty_ordered": 12,
#           "qty_shipped": 0,
#           "qty_remaining": 12,
#           "line_status": "Open",
#           "line_hash": "...",
#       }
#   ],
#   "analysis": {
#       "buildable_qty": 8,
#       "shortage_count": 2,
#       "main_blocker": "720-000141",
#       "missing_parts": [
#           {"part_number": "720-000141", "short": 4},
#           {"part_number": "345-000031", "short": 2}
#       ],
#       "summary": {"ready": False},
#       "analysis_hash": "..."
#   },
#   "raw_json": {...}
# }


def _hash_dict(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def fetch_orders_modified_since(last_sync_at: datetime | None) -> list[dict[str, Any]]:
    if settings.sos_use_mock:
        return _mock_orders(last_sync_at)

    # TODO: Replace this block with your existing sos_requests integration.
    # Suggested idea:
    # 1. refresh token if needed
    # 2. fetch open / partial / recently modified SOs
    # 3. normalize into the structure documented above
    # 4. return the list
    raise NotImplementedError(
        "SOS live integration is not wired yet. Open sos_adapter.py and connect your existing SOS code."
    )


def _mock_orders(last_sync_at: datetime | None) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    if last_sync_at and now - last_sync_at < timedelta(minutes=1):
        return []

    raw_orders: list[dict[str, Any]] = [
        {
            "so_number": "2026-1045",
            "customer_name": "Nabtesco",
            "order_date": now.date(),
            "status": "Open",
            "shipped_flag": False,
            "active": True,
            "priority_rank": 1,
            "priority_label": "Rush",
            "assigned_to": "Thiago",
            "internal_note": "Need before Friday",
            "source_last_modified": now,
            "shipped_line_color": "red",
            "lines": [
                {
                    "line_no": 1,
                    "item_number": "130-000101",
                    "description": "TR-302 Edge",
                    "qty_ordered": 12,
                    "qty_shipped": 0,
                    "qty_remaining": 12,
                    "line_status": "Open",
                }
            ],
            "analysis": {
                "buildable_qty": 8,
                "shortage_count": 2,
                "main_blocker": "720-000141",
                "missing_parts": [
                    {"part_number": "720-000141", "short": 4},
                    {"part_number": "345-000031", "short": 2},
                ],
                "summary": {"ready": False, "status": "Shortage"},
            },
        },
        {
            "so_number": "2026-1052",
            "customer_name": "Acme Robotics",
            "order_date": now.date(),
            "status": "Shipped",
            "shipped_flag": True,
            "active": False,
            "priority_rank": 2,
            "priority_label": "Normal",
            "assigned_to": "Maria",
            "internal_note": "Completed and shipped",
            "source_last_modified": now,
            "shipped_line_color": "green",
            "lines": [
                {
                    "line_no": 1,
                    "item_number": "140-000027",
                    "description": "OC-301-30-ST",
                    "qty_ordered": 6,
                    "qty_shipped": 6,
                    "qty_remaining": 0,
                    "line_status": "Shipped",
                }
            ],
            "analysis": {
                "buildable_qty": 6,
                "shortage_count": 0,
                "main_blocker": None,
                "missing_parts": [],
                "summary": {"ready": True, "status": "Shipped"},
            },
        },
    ]

    normalized: list[dict[str, Any]] = []
    for order in raw_orders:
        lines = []
        for line in order["lines"]:
            line = dict(line)
            line["line_hash"] = _hash_dict(line)
            lines.append(line)
        analysis = dict(order["analysis"])
        analysis["analysis_hash"] = _hash_dict(analysis)

        normalized_order = {
            **order,
            "lines": lines,
            "analysis": analysis,
        }
        normalized_order["source_hash"] = _hash_dict(
            {
                "header": {k: v for k, v in order.items() if k not in {"lines", "analysis"}},
                "lines": lines,
            }
        )
        normalized_order["raw_json"] = normalized_order
        normalized.append(normalized_order)

    return normalized

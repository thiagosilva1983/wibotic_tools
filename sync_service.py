from __future__ import annotations

import time
from datetime import datetime

from config import settings
from db import get_session, get_sync_state, set_sync_state, upsert_analysis, upsert_sales_order, replace_sales_order_lines
from sos_adapter import fetch_orders_modified_since


LAST_SYNC_KEY = "last_sync_at"
LAST_SUCCESS_KEY = "last_success_at"
LAST_ERROR_KEY = "last_error"


def parse_sync_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def run_once() -> int:
    with get_session() as session:
        last_sync_raw = get_sync_state(session, LAST_SYNC_KEY, "")
        last_sync_at = parse_sync_timestamp(last_sync_raw)

    changed_orders = fetch_orders_modified_since(last_sync_at)

    if not changed_orders:
        with get_session() as session:
            set_sync_state(session, LAST_SUCCESS_KEY, datetime.now().isoformat())
        return 0

    with get_session() as session:
        for order in changed_orders:
            header = {k: v for k, v in order.items() if k not in {"lines", "analysis"}}
            upsert_sales_order(session, header)
            replace_sales_order_lines(session, order["so_number"], order["lines"])
            upsert_analysis(session, order["so_number"], order["analysis"])

        set_sync_state(session, LAST_SYNC_KEY, datetime.now().isoformat())
        set_sync_state(session, LAST_SUCCESS_KEY, datetime.now().isoformat())
        set_sync_state(session, LAST_ERROR_KEY, "")

    return len(changed_orders)


def main() -> None:
    print(f"Starting sync loop. Interval: {settings.sync_interval_seconds}s")
    while True:
        try:
            changed = run_once()
            print(f"Sync complete. Changed orders: {changed}")
        except Exception as exc:
            print(f"Sync error: {exc}")
            with get_session() as session:
                set_sync_state(session, LAST_ERROR_KEY, str(exc))
        time.sleep(settings.sync_interval_seconds)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from config import settings


if not settings.database_url:
    raise RuntimeError("DATABASE_URL is missing. Fill .env before running the app.")

engine: Engine = create_engine(settings.database_url, future=True, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


@contextmanager
def get_session() -> Iterable[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def get_sync_state(session: Session, key: str, default: str = "") -> str:
    row = session.execute(
        text("SELECT state_value FROM sync_state WHERE state_key = :key"),
        {"key": key},
    ).fetchone()
    return row[0] if row else default


def set_sync_state(session: Session, key: str, value: str) -> None:
    session.execute(
        text(
            """
            INSERT INTO sync_state (state_key, state_value, updated_at)
            VALUES (:key, :value, NOW())
            ON CONFLICT (state_key)
            DO UPDATE SET state_value = EXCLUDED.state_value, updated_at = NOW()
            """
        ),
        {"key": key, "value": value},
    )


def upsert_sales_order(session: Session, so: dict[str, Any]) -> None:
    session.execute(
        text(
            """
            INSERT INTO sales_orders (
                so_number, customer_name, order_date, status, shipped_flag, active,
                priority_rank, priority_label, assigned_to, internal_note,
                source_last_modified, source_hash, buildable_qty, shortage_count,
                main_blocker, shipped_line_color, raw_json, updated_at
            ) VALUES (
                :so_number, :customer_name, :order_date, :status, :shipped_flag, :active,
                COALESCE(:priority_rank, 99), :priority_label, :assigned_to, :internal_note,
                :source_last_modified, :source_hash, :buildable_qty, :shortage_count,
                :main_blocker, :shipped_line_color, CAST(:raw_json AS JSONB), NOW()
            )
            ON CONFLICT (so_number)
            DO UPDATE SET
                customer_name = EXCLUDED.customer_name,
                order_date = EXCLUDED.order_date,
                status = EXCLUDED.status,
                shipped_flag = EXCLUDED.shipped_flag,
                active = EXCLUDED.active,
                source_last_modified = EXCLUDED.source_last_modified,
                source_hash = EXCLUDED.source_hash,
                buildable_qty = EXCLUDED.buildable_qty,
                shortage_count = EXCLUDED.shortage_count,
                main_blocker = EXCLUDED.main_blocker,
                shipped_line_color = EXCLUDED.shipped_line_color,
                raw_json = EXCLUDED.raw_json,
                updated_at = NOW()
            """
        ),
        {
            **so,
            "raw_json": json.dumps(so.get("raw_json", so)),
        },
    )


def replace_sales_order_lines(session: Session, so_number: str, lines: list[dict[str, Any]]) -> None:
    session.execute(text("DELETE FROM sales_order_lines WHERE so_number = :so_number"), {"so_number": so_number})
    for line in lines:
        session.execute(
            text(
                """
                INSERT INTO sales_order_lines (
                    so_number, line_no, item_number, description, qty_ordered,
                    qty_shipped, qty_remaining, line_status, line_hash, raw_json, updated_at
                ) VALUES (
                    :so_number, :line_no, :item_number, :description, :qty_ordered,
                    :qty_shipped, :qty_remaining, :line_status, :line_hash,
                    CAST(:raw_json AS JSONB), NOW()
                )
                """
            ),
            {
                **line,
                "so_number": so_number,
                "raw_json": json.dumps(line.get("raw_json", line)),
            },
        )


def upsert_analysis(session: Session, so_number: str, analysis: dict[str, Any]) -> None:
    session.execute(
        text(
            """
            INSERT INTO so_analysis (
                so_number, buildable_qty, shortage_count, main_blocker,
                missing_parts_json, summary_json, analysis_hash, analyzed_at
            ) VALUES (
                :so_number, :buildable_qty, :shortage_count, :main_blocker,
                CAST(:missing_parts_json AS JSONB), CAST(:summary_json AS JSONB),
                :analysis_hash, NOW()
            )
            ON CONFLICT (so_number)
            DO UPDATE SET
                buildable_qty = EXCLUDED.buildable_qty,
                shortage_count = EXCLUDED.shortage_count,
                main_blocker = EXCLUDED.main_blocker,
                missing_parts_json = EXCLUDED.missing_parts_json,
                summary_json = EXCLUDED.summary_json,
                analysis_hash = EXCLUDED.analysis_hash,
                analyzed_at = NOW()
            """
        ),
        {
            "so_number": so_number,
            "buildable_qty": analysis.get("buildable_qty"),
            "shortage_count": analysis.get("shortage_count"),
            "main_blocker": analysis.get("main_blocker"),
            "missing_parts_json": json.dumps(analysis.get("missing_parts", [])),
            "summary_json": json.dumps(analysis.get("summary", {})),
            "analysis_hash": analysis.get("analysis_hash"),
        },
    )

    session.execute(
        text(
            """
            UPDATE sales_orders
            SET buildable_qty = :buildable_qty,
                shortage_count = :shortage_count,
                main_blocker = :main_blocker,
                last_analyzed_at = NOW(),
                updated_at = NOW()
            WHERE so_number = :so_number
            """
        ),
        {
            "so_number": so_number,
            "buildable_qty": analysis.get("buildable_qty"),
            "shortage_count": analysis.get("shortage_count"),
            "main_blocker": analysis.get("main_blocker"),
        },
    )


def update_shared_fields(
    session: Session,
    so_number: str,
    priority_rank: int | None,
    priority_label: str | None,
    assigned_to: str | None,
    internal_note: str | None,
    changed_by: str = "streamlit_user",
) -> None:
    existing = session.execute(
        text(
            "SELECT priority_rank, priority_label, assigned_to, internal_note FROM sales_orders WHERE so_number = :so_number"
        ),
        {"so_number": so_number},
    ).mappings().first()

    if not existing:
        return

    session.execute(
        text(
            """
            UPDATE sales_orders
            SET priority_rank = :priority_rank,
                priority_label = :priority_label,
                assigned_to = :assigned_to,
                internal_note = :internal_note,
                updated_at = NOW()
            WHERE so_number = :so_number
            """
        ),
        {
            "so_number": so_number,
            "priority_rank": priority_rank,
            "priority_label": priority_label,
            "assigned_to": assigned_to,
            "internal_note": internal_note,
        },
    )

    changes = {
        "priority_rank": (existing["priority_rank"], priority_rank),
        "priority_label": (existing["priority_label"], priority_label),
        "assigned_to": (existing["assigned_to"], assigned_to),
        "internal_note": (existing["internal_note"], internal_note),
    }
    for field_name, (old_value, new_value) in changes.items():
        if str(old_value) != str(new_value):
            session.execute(
                text(
                    """
                    INSERT INTO audit_log (so_number, field_name, old_value, new_value, changed_by)
                    VALUES (:so_number, :field_name, :old_value, :new_value, :changed_by)
                    """
                ),
                {
                    "so_number": so_number,
                    "field_name": field_name,
                    "old_value": None if old_value is None else str(old_value),
                    "new_value": None if new_value is None else str(new_value),
                    "changed_by": changed_by,
                },
            )

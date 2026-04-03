from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from sqlalchemy import text

from config import settings
from db import get_session, update_shared_fields

st.set_page_config(page_title=settings.app_title, layout="wide")

st_autorefresh(interval=settings.ui_refresh_seconds * 1000, key="shared_app_refresh")

st.title(settings.app_title)
st.caption(f"Shared database mode • auto refresh every {settings.ui_refresh_seconds} sec")


@st.cache_data(ttl=30)
def load_orders() -> pd.DataFrame:
    query = text(
        """
        SELECT
            so_number,
            customer_name,
            order_date,
            status,
            shipped_flag,
            active,
            priority_rank,
            priority_label,
            assigned_to,
            internal_note,
            buildable_qty,
            shortage_count,
            main_blocker,
            source_last_modified,
            last_analyzed_at,
            updated_at,
            CASE
                WHEN shipped_flag THEN '🟢 Shipped'
                WHEN COALESCE(shortage_count, 0) = 0 THEN '🟢 Ready'
                WHEN COALESCE(buildable_qty, 0) > 0 THEN '🟡 Partial'
                ELSE '🔴 Shortage'
            END AS visual_status
        FROM sales_orders
        ORDER BY priority_rank ASC, order_date DESC, so_number DESC
        """
    )
    with get_session() as session:
        df = pd.read_sql(query, session.bind)
    return df


@st.cache_data(ttl=15)
def load_sync_state() -> dict[str, str]:
    with get_session() as session:
        rows = session.execute(text("SELECT state_key, state_value FROM sync_state")).fetchall()
    return {row[0]: row[1] for row in rows}


@st.cache_data(ttl=30)
def load_so_lines(so_number: str) -> pd.DataFrame:
    query = text(
        """
        SELECT line_no, item_number, description, qty_ordered, qty_shipped, qty_remaining, line_status
        FROM sales_order_lines
        WHERE so_number = :so_number
        ORDER BY line_no
        """
    )
    with get_session() as session:
        return pd.read_sql(query, session.bind, params={"so_number": so_number})


@st.cache_data(ttl=30)
def load_analysis(so_number: str) -> pd.DataFrame:
    query = text(
        """
        SELECT missing_parts_json
        FROM so_analysis
        WHERE so_number = :so_number
        """
    )
    with get_session() as session:
        row = session.execute(query, {"so_number": so_number}).mappings().first()
    if not row or not row["missing_parts_json"]:
        return pd.DataFrame(columns=["part_number", "short"])
    return pd.DataFrame(row["missing_parts_json"])


def clear_caches() -> None:
    load_orders.clear()
    load_sync_state.clear()
    load_so_lines.clear()
    load_analysis.clear()


sync_state = load_sync_state()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Last sync", sync_state.get("last_sync_at", "Never"))
col2.metric("Last success", sync_state.get("last_success_at", "Never"))
col3.metric("Last error", sync_state.get("last_error", ""))
if col4.button("Refresh now"):
    clear_caches()
    st.rerun()

orders_df = load_orders()

if orders_df.empty:
    st.info("No orders found yet. Start sync_service.py and confirm DATABASE_URL is correct.")
    st.stop()

left, right = st.columns([3, 1])
with left:
    show_inactive = st.checkbox("Show shipped / inactive", value=True)
with right:
    search_text = st.text_input("Search SO / customer / item")

filtered_df = orders_df.copy()
if not show_inactive:
    filtered_df = filtered_df[filtered_df["active"] == True]  # noqa: E712
if search_text:
    mask = (
        filtered_df["so_number"].astype(str).str.contains(search_text, case=False, na=False)
        | filtered_df["customer_name"].astype(str).str.contains(search_text, case=False, na=False)
        | filtered_df["main_blocker"].astype(str).str.contains(search_text, case=False, na=False)
    )
    filtered_df = filtered_df[mask]

st.subheader("Weekly Production")
st.dataframe(
    filtered_df[
        [
            "priority_rank",
            "priority_label",
            "visual_status",
            "so_number",
            "customer_name",
            "status",
            "buildable_qty",
            "shortage_count",
            "main_blocker",
            "assigned_to",
            "internal_note",
            "updated_at",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

so_numbers = filtered_df["so_number"].tolist()
selected_so = st.selectbox("Select a Sales Order", so_numbers)
selected_row = filtered_df.loc[filtered_df["so_number"] == selected_so].iloc[0]

st.markdown("---")
col_a, col_b = st.columns([2, 1])
with col_a:
    st.subheader(f"Sales Order {selected_so}")
    st.write(f"**Customer:** {selected_row['customer_name']}")
    st.write(f"**Status:** {selected_row['visual_status']}")
    st.write(f"**Main blocker:** {selected_row['main_blocker']}")
with col_b:
    st.subheader("Shared fields")
    with st.form("edit_shared_fields"):
        priority_rank = st.number_input("Priority rank", min_value=1, max_value=999, value=int(selected_row["priority_rank"] or 99))
        priority_label = st.text_input("Priority label", value=selected_row["priority_label"] or "")
        assigned_to = st.text_input("Assigned to", value=selected_row["assigned_to"] or "")
        internal_note = st.text_area("Internal note", value=selected_row["internal_note"] or "")
        changed_by = st.text_input("Changed by", value="streamlit_user")
        submitted = st.form_submit_button("Save shared fields")
        if submitted:
            with get_session() as session:
                update_shared_fields(
                    session=session,
                    so_number=selected_so,
                    priority_rank=int(priority_rank),
                    priority_label=priority_label or None,
                    assigned_to=assigned_to or None,
                    internal_note=internal_note or None,
                    changed_by=changed_by or "streamlit_user",
                )
            clear_caches()
            st.success("Saved. All users will see this after refresh.")
            st.rerun()

st.markdown("### Order lines")
st.dataframe(load_so_lines(selected_so), use_container_width=True, hide_index=True)

st.markdown("### Missing parts")
missing_df = load_analysis(selected_so)
if missing_df.empty:
    st.success("No missing parts in saved analysis.")
else:
    st.dataframe(missing_df, use_container_width=True, hide_index=True)

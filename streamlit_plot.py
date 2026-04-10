import streamlit as st
import pandas as pd

# ===============================
# Wibotic ERP - MRP Module REV A
# Standalone demo module
# ===============================

st.set_page_config(page_title="Wibotic ERP", layout="wide")

# ===============================
# HEADER
# ===============================
st.title("Wibotic ERP")
st.subheader("MRP - Materials Planning")

st.info(
    "This is a standalone REV A demo module. It uses mock data now, "
    "and you can later replace the mock functions with your real SOS Inventory fetch logic."
)

# ===============================
# MOCK DATA (REPLACE LATER)
# ===============================

def get_open_sales_orders() -> list[dict]:
    return [
        {"so": "SO-2026-101", "product": "PROD-A", "qty": 2},
        {"so": "SO-2026-102", "product": "PROD-B", "qty": 1},
        {"so": "SO-2026-103", "product": "PROD-C", "qty": 3},
    ]


def get_bom(product: str) -> list[dict]:
    bom_data = {
        "PROD-A": [
            {"part": "355-000001", "qty": 10},
            {"part": "140-000010", "qty": 2},
        ],
        "PROD-B": [
            {"part": "355-000001", "qty": 20},
            {"part": "190-000020", "qty": 1},
        ],
        "PROD-C": [
            {"part": "355-000001", "qty": 5},
            {"part": "130-000030", "qty": 3},
        ],
    }
    return bom_data.get(product, [])


def get_inventory() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"part": "355-000001", "on_hand": 120},
            {"part": "140-000010", "on_hand": 5},
            {"part": "190-000020", "on_hand": 0},
            {"part": "130-000030", "on_hand": 4},
        ]
    )


def get_purchase_orders() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"part": "355-000001", "incoming": 50, "eta": "2026-04-15"},
            {"part": "190-000020", "incoming": 2, "eta": "2026-04-12"},
        ]
    )


# ===============================
# MRP LOGIC
# ===============================

def build_mrp() -> tuple[pd.DataFrame, pd.DataFrame]:
    sales_orders = get_open_sales_orders()
    demand_rows: list[dict] = []

    for so in sales_orders:
        bom = get_bom(so["product"])
        for item in bom:
            demand_rows.append(
                {
                    "so": so["so"],
                    "product": so["product"],
                    "part": item["part"],
                    "qty_per_product": item["qty"],
                    "so_qty": so["qty"],
                    "qty_required": item["qty"] * so["qty"],
                }
            )

    df_demand = pd.DataFrame(demand_rows)

    if df_demand.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Aggregate demand by part number
    df_summary = (
        df_demand.groupby("part", as_index=False)
        .agg(
            total_demand=("qty_required", "sum"),
            impacted_sos_count=("so", "nunique"),
        )
    )

    # Merge inventory
    df_inventory = get_inventory()
    df_summary = df_summary.merge(df_inventory, on="part", how="left")

    # Merge PO data
    df_po = get_purchase_orders()
    df_summary = df_summary.merge(df_po, on="part", how="left")

    # Fill blanks
    df_summary["on_hand"] = df_summary["on_hand"].fillna(0)
    df_summary["incoming"] = df_summary["incoming"].fillna(0)
    df_summary["eta"] = df_summary["eta"].fillna("")

    # Calculations
    df_summary["available_now"] = df_summary["on_hand"]
    df_summary["available_total"] = df_summary["on_hand"] + df_summary["incoming"]
    df_summary["shortage_now"] = (df_summary["total_demand"] - df_summary["on_hand"]).clip(lower=0)
    df_summary["shortage_after_po"] = (
        df_summary["total_demand"] - df_summary["available_total"]
    ).clip(lower=0)
    df_summary["covered_by_po"] = (df_summary["shortage_now"] - df_summary["shortage_after_po"]).clip(lower=0)

    # Add SO list per part
    impacted_sos = (
        df_demand.groupby("part")["so"]
        .apply(lambda x: ", ".join(sorted(set(x))))
        .reset_index(name="impacted_sos")
    )
    df_summary = df_summary.merge(impacted_sos, on="part", how="left")

    # Order columns
    df_summary = df_summary[
        [
            "part",
            "total_demand",
            "on_hand",
            "incoming",
            "eta",
            "available_now",
            "available_total",
            "shortage_now",
            "covered_by_po",
            "shortage_after_po",
            "impacted_sos_count",
            "impacted_sos",
        ]
    ]

    df_summary = df_summary.sort_values(
        by=["shortage_after_po", "shortage_now", "total_demand"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return df_summary, df_demand


# ===============================
# UI HELPERS
# ===============================

def highlight_shortage(row: pd.Series) -> list[str]:
    if row.get("shortage_after_po", 0) > 0:
        return ["background-color: #5c1f1f"] * len(row)
    if row.get("shortage_now", 0) > 0:
        return ["background-color: #4a3b1f"] * len(row)
    return [""] * len(row)


# ===============================
# MAIN UI
# ===============================

col1, col2 = st.columns([1, 1])
with col1:
    run_clicked = st.button("Run MRP", type="primary", use_container_width=True)
with col2:
    st.button("Refresh Mock Data", use_container_width=True)

if run_clicked:
    df_summary, df_demand = build_mrp()

    if df_summary.empty:
        st.warning("No demand data found.")
    else:
        st.markdown("## Part Summary")
        st.dataframe(
            df_summary.style.apply(highlight_shortage, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        critical = df_summary[df_summary["shortage_after_po"] > 0].copy()
        covered_by_po = df_summary[
            (df_summary["shortage_now"] > 0) & (df_summary["shortage_after_po"] == 0)
        ].copy()
        good_now = df_summary[df_summary["shortage_now"] == 0].copy()

        m1, m2, m3 = st.columns(3)
        m1.metric("Critical Parts", len(critical))
        m2.metric("Covered by Open PO", len(covered_by_po))
        m3.metric("Fully Covered Now", len(good_now))

        st.markdown("## Critical Shortages")
        if critical.empty:
            st.success("No critical shortages after incoming PO coverage.")
        else:
            st.dataframe(critical, use_container_width=True, hide_index=True)

        st.markdown("## Demand Details by Sales Order")
        st.dataframe(df_demand, use_container_width=True, hide_index=True)

        st.markdown("## Sales Order Summary")
        so_summary = (
            df_demand.groupby(["so", "product"], as_index=False)
            .agg(
                total_parts=("part", "count"),
                total_qty_required=("qty_required", "sum"),
            )
            .sort_values(by="so")
        )
        st.dataframe(so_summary, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "Next step: replace get_open_sales_orders(), get_bom(), get_inventory(), and "
    "get_purchase_orders() with your real SOS Inventory functions from your current app."
)

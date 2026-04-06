import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple, Any

st.set_page_config(page_title="Priority Fulfillment Dashboard", layout="wide")

def load_sales_orders() -> pd.DataFrame:
    return pd.DataFrame([
        {"Priority": 1, "SO Number": "SO-2026-101", "Customer": "Nabtesco", "Product": "TR-302", "Qty Remaining": 5},
        {"Priority": 2, "SO Number": "SO-2026-102", "Customer": "Jabil", "Product": "TR-302", "Qty Remaining": 3},
        {"Priority": 3, "SO Number": "SO-2026-103", "Customer": "GM", "Product": "TR-1000", "Qty Remaining": 2},
        {"Priority": 4, "SO Number": "SO-2026-104", "Customer": "Earthsense", "Product": "TR-302", "Qty Remaining": 4},
        {"Priority": 5, "SO Number": "SO-2026-105", "Customer": "Wibotic Sales", "Product": "TR-1000", "Qty Remaining": 3},
    ])

def load_inventory() -> pd.DataFrame:
    return pd.DataFrame([
        {"Part": "TR-302", "On Hand": 2},
        {"Part": "TR-1000", "On Hand": 0},
        {"Part": "PCB-302", "On Hand": 4},
        {"Part": "PCB-1000", "On Hand": 1},
        {"Part": "Ferrite", "On Hand": 20},
        {"Part": "Cable", "On Hand": 12},
        {"Part": "Screw", "On Hand": 100},
    ])

def load_bom() -> Dict[str, List[Tuple[str, int]]]:
    return {
        "TR-302": [("PCB-302", 1), ("Ferrite", 2), ("Cable", 1), ("Screw", 4)],
        "TR-1000": [("PCB-1000", 1), ("Ferrite", 4), ("Cable", 2), ("Screw", 6)],
    }

def inventory_to_pool(inventory_df: pd.DataFrame) -> Dict[str, int]:
    pool: Dict[str, int] = {}
    for _, row in inventory_df.iterrows():
        pool[str(row["Part"])] = int(pd.to_numeric(row["On Hand"], errors="coerce") or 0)
    return pool

def explode_bom_one_unit(part: str, bom: Dict[str, List[Tuple[str, int]]], out: Dict[str, int]) -> None:
    if part not in bom:
        out[part] = out.get(part, 0) + 1
        return
    for comp, qty in bom[part]:
        if comp in bom:
            nested: Dict[str, int] = {}
            explode_bom_one_unit(comp, bom, nested)
            for nested_part, nested_qty in nested.items():
                out[nested_part] = out.get(nested_part, 0) + nested_qty * qty
        else:
            out[comp] = out.get(comp, 0) + qty

def one_unit_requirements(product: str, bom: Dict[str, List[Tuple[str, int]]]) -> Dict[str, int]:
    req: Dict[str, int] = {}
    explode_bom_one_unit(product, bom, req)
    return req

def compute_buildable_qty(product: str, qty_needed: int, available_pool: Dict[str, int], bom: Dict[str, List[Tuple[str, int]]]):
    fg_on_hand = available_pool.get(product, 0)
    use_fg = min(fg_on_hand, qty_needed)
    remaining_units = qty_needed - use_fg
    req_per_unit = one_unit_requirements(product, bom)

    if remaining_units <= 0:
        return qty_needed, {product: use_fg}, {}

    if not req_per_unit:
        return use_fg, {product: use_fg}, {product: qty_needed - use_fg}

    component_limits = []
    for comp, per_unit_qty in req_per_unit.items():
        on_hand = available_pool.get(comp, 0)
        component_limits.append(on_hand // per_unit_qty if per_unit_qty > 0 else 0)

    build_from_components = min(component_limits) if component_limits else 0
    total_buildable = use_fg + min(remaining_units, build_from_components)

    consumption: Dict[str, int] = {}
    if use_fg > 0:
        consumption[product] = use_fg

    units_from_components_used = max(0, total_buildable - use_fg)
    if units_from_components_used > 0:
        for comp, per_unit_qty in req_per_unit.items():
            consumption[comp] = consumption.get(comp, 0) + per_unit_qty * units_from_components_used

    shortage: Dict[str, int] = {}
    if total_buildable < qty_needed:
        fg_missing = max(0, qty_needed - fg_on_hand)
        if fg_missing > 0:
            for comp, per_unit_qty in req_per_unit.items():
                req_qty = per_unit_qty * fg_missing
                available = available_pool.get(comp, 0)
                if available < req_qty:
                    shortage[comp] = req_qty - available

    return total_buildable, consumption, shortage

def apply_consumption(pool: Dict[str, int], consumption: Dict[str, int]) -> Dict[str, int]:
    new_pool = dict(pool)
    for part, qty in consumption.items():
        new_pool[part] = max(0, new_pool.get(part, 0) - qty)
    return new_pool

def fulfillment_status(buildable_qty: int, qty_needed: int) -> str:
    if buildable_qty >= qty_needed:
        return "Full"
    if buildable_qty > 0:
        return "Partial"
    return "Blocked"

def compute_priority_fulfillment(sales_orders_df: pd.DataFrame, inventory_df: pd.DataFrame, bom: Dict[str, List[Tuple[str, int]]]):
    orders = sales_orders_df.copy()
    orders["Qty Remaining"] = pd.to_numeric(orders["Qty Remaining"], errors="coerce").fillna(0).astype(int)
    orders["Priority"] = pd.to_numeric(orders["Priority"], errors="coerce").fillna(999).astype(int)
    orders = orders.sort_values(["Priority", "SO Number"]).reset_index(drop=True)

    pool = inventory_to_pool(inventory_df)
    order_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []

    for _, row in orders.iterrows():
        priority = int(row["Priority"])
        so = str(row["SO Number"])
        customer = str(row.get("Customer", ""))
        product = str(row["Product"])
        qty_needed = int(row["Qty Remaining"])

        buildable_qty, consumption, shortage = compute_buildable_qty(product, qty_needed, pool, bom)
        status = fulfillment_status(buildable_qty, qty_needed)
        main_blocker = next(iter(sorted(shortage, key=shortage.get, reverse=True)), "-")
        allocated_flag = "Yes" if buildable_qty > 0 else "No"

        order_rows.append({
            "Priority": priority,
            "SO Number": so,
            "Customer": customer,
            "Product": product,
            "Qty Remaining": qty_needed,
            "Fulfillable": "Yes" if status == "Full" else "No",
            "Buildable Qty": buildable_qty,
            "Allocated": allocated_flag,
            "Status": status,
            "Main Blocker": main_blocker,
        })

        req_per_unit = one_unit_requirements(product, bom)
        fg_used = consumption.get(product, 0)
        if fg_used:
            detail_rows.append({
                "SO Number": so,
                "Product": product,
                "Component": product,
                "Needed Full SO": qty_needed,
                "Allocated for SO": fg_used,
                "Short": 0,
            })
        units_built_from_components = max(0, buildable_qty - fg_used)
        for comp, per_unit_qty in req_per_unit.items():
            needed_full = per_unit_qty * max(0, qty_needed - min(qty_needed, pool.get(product, 0)))
            allocated = per_unit_qty * units_built_from_components
            short = shortage.get(comp, 0)
            detail_rows.append({
                "SO Number": so,
                "Product": product,
                "Component": comp,
                "Needed Full SO": needed_full,
                "Allocated for SO": allocated,
                "Short": short,
            })

        pool = apply_consumption(pool, consumption)

    order_df = pd.DataFrame(order_rows)
    detail_df = pd.DataFrame(detail_rows)
    remaining_df = pd.DataFrame([{"Part": part, "Remaining After Allocation": qty} for part, qty in sorted(pool.items())])
    return order_df, detail_df, remaining_df

def compute_blocker_summary(detail_df: pd.DataFrame, inventory_df: pd.DataFrame):
    if detail_df.empty:
        return pd.DataFrame(columns=["Component", "Short Qty", "Orders Impacted", "On Hand"])

    short_df = detail_df[detail_df["Short"] > 0].copy()
    if short_df.empty:
        return pd.DataFrame(columns=["Component", "Short Qty", "Orders Impacted", "On Hand"])

    summary = short_df.groupby("Component").agg(**{
        "Short Qty": ("Short", "sum"),
        "Orders Impacted": ("SO Number", "nunique"),
    }).reset_index()

    on_hand_map = dict(zip(inventory_df["Part"], inventory_df["On Hand"]))
    summary["On Hand"] = summary["Component"].map(on_hand_map).fillna(0).astype(int)
    return summary.sort_values(["Orders Impacted", "Short Qty"], ascending=[False, False]).reset_index(drop=True)

st.title("Wibotic Weekly Production Dashboard")
st.caption("Priority-based fulfillment, buildable qty, allocated status, and blockers.")

sales_orders_df = load_sales_orders()
inventory_df = load_inventory()
bom = load_bom()

order_df, detail_df, remaining_df = compute_priority_fulfillment(sales_orders_df, inventory_df, bom)
blocker_df = compute_blocker_summary(detail_df, inventory_df)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Open SOs", len(order_df))
m2.metric("Full", int((order_df["Status"] == "Full").sum()))
m3.metric("Partial", int((order_df["Status"] == "Partial").sum()))
m4.metric("Blocked", int((order_df["Status"] == "Blocked").sum()))

st.markdown("### Priority Fulfillment Board")
st.dataframe(order_df, use_container_width=True)

left, right = st.columns([1.2, 1.0])

with left:
    st.markdown("### Blockers")
    if blocker_df.empty:
        st.success("No blockers. All visible orders are fully fulfillable.")
    else:
        st.dataframe(blocker_df, use_container_width=True)

with right:
    st.markdown("### Remaining Inventory After Allocation")
    st.dataframe(remaining_df, use_container_width=True)

st.markdown("### SO Drill-down")
selected_so = st.selectbox("Select Sales Order", order_df["SO Number"].tolist())
selected_order = order_df[order_df["SO Number"] == selected_so].iloc[0]
selected_detail = detail_df[detail_df["SO Number"] == selected_so].copy()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Priority", int(selected_order["Priority"]))
c2.metric("Fulfillable", selected_order["Fulfillable"])
c3.metric("Buildable Qty", int(selected_order["Buildable Qty"]))
c4.metric("Allocated", selected_order["Allocated"])
c5.metric("Main Blocker", selected_order["Main Blocker"])

st.dataframe(selected_detail, use_container_width=True)

st.download_button(
    "Download Priority Fulfillment CSV",
    data=order_df.to_csv(index=False).encode("utf-8"),
    file_name="priority_fulfillment_board.csv",
    mime="text/csv",
)

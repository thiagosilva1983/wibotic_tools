
from __future__ import annotations

import csv
import io
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

APP_TITLE = "WiBotic Tool C"
BASE_URL = "https://api.sosinventory.com/api/v2/"
TOKEN_URL = "https://api.sosinventory.com/oauth2/token"

DEFAULT_LOCATIONS = [
    "Production Inventory",
    "Storage Unit",
    "PCA Components",
    "MRB Awaiting Evaluation",
    "B Stock",
    "Tate Technology",
    "Consigned Qualtech",
]

SAFE_FILENAME = re.compile(r"[^a-zA-Z0-9_-]+")
SO_RE = re.compile(r"SO-[A-Z0-9-]+", re.IGNORECASE)


@dataclass
class LineItem:
    item_id: int
    on_hand: float
    type: str
    quantity: float
    fullname: str
    description: str
    has_serial: bool
    notes: str = ""


# -----------------------------
# Generic helpers
# -----------------------------
def safe_str(value: Any) -> str:
    return "" if value is None else str(value).strip()


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def safe_filename(title: str) -> str:
    clean = SAFE_FILENAME.sub("_", safe_str(title))
    return (clean or "report")[:60] + ".csv"


def normalize_text(text: str) -> str:
    text = safe_str(text).lower()
    return re.sub(r"\s+", " ", text)


def extract_base_part(text: str) -> str:
    text = normalize_text(text)
    m = re.match(r"([a-z0-9]+-\d+)", text)
    return m.group(1) if m else text


def type_rank(item_type: str) -> int:
    t = normalize_text(item_type)
    if t in ["inventory item", "assembly", "item group"]:
        return 0
    if t in ["non-inventory item"]:
        return 1
    if t in ["service"]:
        return 2
    if t in ["expense"]:
        return 3
    return 4


def first_present(mapping: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping.get(key) is not None:
            return mapping.get(key)
    return default


def extract_location(notes: str) -> str:
    if not notes:
        return ""
    m = re.search(r"(Aisle\s*[^,|]+?(?:,\s*Shelf\s*[^,|]+)?)", notes, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def trim_text(text: str, n: int = 60) -> str:
    text = safe_str(text)
    return text if len(text) <= n else text[: n - 3] + "..."


def load_requests_from_csv(uploaded_file) -> List[Tuple[str, int]]:
    uploaded_file.seek(0)
    wrapper = io.TextIOWrapper(uploaded_file, encoding="utf-8", newline="")
    reader = csv.DictReader(wrapper)
    fieldnames = reader.fieldnames or []
    header_map = {h.lower().strip(): h for h in fieldnames}

    def find_header(*candidates: str) -> Optional[str]:
        for c in candidates:
            key = c.lower().strip()
            if key in header_map:
                return header_map[key]
        return None

    name_col = find_header("Item Name", "ItemName", "Part Name", "Name", "Item Number", "Part Number", "PN")
    qty_col = find_header("quantity", "qty", "QTY", "Quantity")
    if not name_col or not qty_col:
        raise ValueError(f"CSV headers missing. Found: {fieldnames}")

    rows: List[Tuple[str, int]] = []
    for row in reader:
        name = safe_str(row.get(name_col))
        if not name:
            continue
        try:
            qty = int(row.get(qty_col) or 1)
        except Exception:
            qty = 1
        rows.append((name, qty))
    uploaded_file.seek(0)
    return rows


# -----------------------------
# Auth / API
# -----------------------------
def get_secret_or_input(label: str, key: str, password: bool = True) -> str:
    default = ""
    try:
        default = str(st.secrets.get(key, ""))
    except Exception:
        default = os.environ.get(key, "")
    return st.sidebar.text_input(label, value=default, type="password" if password else "default")


def refresh_token(refresh_token_value: str) -> Dict[str, Any]:
    resp = requests.post(
        TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Host": "api.sosinventory.com"},
        data={"grant_type": "refresh_token", "refresh_token": refresh_token_value},
        timeout=30,
    )
    payload = resp.json()
    if not resp.ok or "access_token" not in payload:
        raise RuntimeError(f"Refresh failed: {payload}")
    return payload


class SOSClient:
    def __init__(self, access_token: str):
        self.headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Host": "api.sosinventory.com",
            "Authorization": f"Bearer {access_token}",
        }

    def _get(self, endpoint: str, params: Optional[dict] = None, allowable_attempts: int = 4) -> dict:
        attempts = 0
        last_json: dict = {}
        while attempts < allowable_attempts:
            attempts += 1
            resp = requests.get(BASE_URL + endpoint, headers=self.headers, params=params, timeout=50)
            try:
                last_json = resp.json()
            except Exception:
                last_json = {"status": "error", "message": resp.text or "", "code": resp.status_code}
            if "throttle" in str(last_json.get("message", "")).lower():
                continue
            if resp.ok:
                return last_json
            if resp.status_code == 404:
                raise ValueError(404)
            raise RuntimeError(f"SOS GET failed ({resp.status_code}): {last_json.get('message', resp.text)}")
        return last_json

    def get_items_by_name(self, name: str, location: Optional[str] = None) -> List[LineItem]:
        params: Dict[str, Any] = {"query": name}
        if location and location != "All":
            params["location"] = location
        items = self._get("item", params=params).get("data", [])
        return [
            LineItem(
                item_id=item["id"],
                on_hand=to_float(first_present(item, ["onhand", "onHand"], 0), 0.0),
                type=safe_str(item.get("type")),
                quantity=1.0,
                fullname=safe_str(item.get("fullname") or item.get("name")),
                description=safe_str(item.get("description")),
                has_serial=bool(item.get("serialTracking", False)),
                notes=safe_str(item.get("notes")),
            )
            for item in items
        ]

    def get_items_by_id(self, ids: List[int], location: Optional[str] = None) -> List[LineItem]:
        if not ids:
            return []
        params: Dict[str, Any] = {"ids": ",".join(str(i) for i in ids)}
        if location and location != "All":
            params["location"] = location
        items = self._get("item", params=params).get("data", [])
        return [
            LineItem(
                item_id=item["id"],
                on_hand=to_float(first_present(item, ["onhand", "onHand"], 0), 0.0),
                type=safe_str(item.get("type")),
                quantity=1.0,
                fullname=safe_str(item.get("fullname") or item.get("name")),
                description=safe_str(item.get("description")),
                has_serial=bool(item.get("serialTracking", False)),
                notes=safe_str(item.get("notes")),
            )
            for item in items
        ]

    def get_single_level_bom(self, item_id: int, assembly_quantity: int = 1) -> List[LineItem]:
        try:
            resp = self._get(f"item/{int(item_id)}/bom")
        except ValueError as e:
            if str(e) == "404":
                return []
            raise
        lines = (resp.get("data") or {}).get("lines", []) or []
        if not lines:
            return []
        bom_ids = [e["componentItem"]["id"] for e in lines if e.get("componentItem")]
        bom_data = self.get_items_by_id(bom_ids)
        for bom_item in bom_data:
            item_build_data = next(e for e in lines if e["componentItem"]["id"] == bom_item.item_id)
            bom_item.quantity = to_float(item_build_data.get("quantity"), 0.0) * assembly_quantity
        return bom_data

    def bom_lookup(self, item: LineItem, quantity: int = 1) -> List[LineItem]:
        line_items: List[LineItem] = []
        assemblies = [{"item_id": item.item_id, "quantity": quantity}]

        def has_subassembly(e: LineItem) -> bool:
            return normalize_text(e.type) in ["assembly", "item group"]

        while assemblies:
            assembly = assemblies.pop()
            bom_data = self.get_single_level_bom(int(assembly["item_id"]), int(assembly["quantity"]))
            if not bom_data:
                item_self = self.get_items_by_id([int(assembly["item_id"])])[0]
                item_self.quantity = float(assembly["quantity"])
                existing = next((x for x in line_items if x.item_id == item_self.item_id), None)
                if existing:
                    existing.quantity += item_self.quantity
                else:
                    line_items.append(item_self)
                continue

            for bom_item in bom_data:
                if has_subassembly(bom_item):
                    assemblies.append({"item_id": bom_item.item_id, "quantity": bom_item.quantity})
                    continue
                existing = next((x for x in line_items if x.item_id == bom_item.item_id), None)
                if existing:
                    existing.quantity += bom_item.quantity
                else:
                    line_items.append(bom_item)
        return line_items

    def get_sales_order_by_number(self, so_number: str) -> Optional[Dict[str, Any]]:
        matches = self._get("salesorder", params={"query": so_number, "maxresults": 50}).get("data", []) or []
        so_l = normalize_text(so_number)
        for so in matches:
            if normalize_text(safe_str(so.get("number"))) == so_l:
                return so
        return matches[0] if matches else None

    def get_sales_order_detail(self, so_id: int) -> Dict[str, Any]:
        return self._get(f"salesorder/{int(so_id)}").get("data", {}) or {}

    @staticmethod
    def extract_sales_order_lines(so_detail: Dict[str, Any]) -> List[Dict[str, Any]]:
        for key in ("lines", "lineItems", "items", "salesOrderLines"):
            if isinstance(so_detail.get(key), list):
                return so_detail[key]
        return []

    def sales_order_to_requests(self, so_detail: Dict[str, Any]) -> List[Tuple[str, int]]:
        lines = self.extract_sales_order_lines(so_detail)
        reqs: List[Tuple[str, int]] = []
        for ln in lines:
            qty = ln.get("quantity") or ln.get("qty") or 1
            try:
                qty_i = int(qty)
            except Exception:
                qty_i = 1
            item_obj = ln.get("item") or {}
            name = safe_str(
                item_obj.get("name")
                or item_obj.get("fullname")
                or ln.get("itemName")
                or ln.get("name")
                or ln.get("description")
            )
            if name:
                reqs.append((name, qty_i))
        return reqs


# -----------------------------
# Matching / location matrix
# -----------------------------
def score_item(item: LineItem, query_name: str) -> int:
    q = normalize_text(query_name)
    q_base = extract_base_part(query_name)
    fullname = normalize_text(item.fullname)
    base = extract_base_part(item.fullname)
    score = 999

    if fullname == q:
        score = 0
    elif base == q_base:
        score = 10 + type_rank(item.type)
        if "labor" in fullname:
            score += 20
    elif fullname.startswith(q):
        score = 40 + type_rank(item.type)
        if "labor" in fullname:
            score += 20
    elif q in fullname or q_base in fullname:
        score = 80 + type_rank(item.type)
        if "labor" in fullname:
            score += 20
    return score


def choose_item_interactive(items: List[LineItem], query_name: str, key_prefix: str) -> Optional[LineItem]:
    if not items:
        return None
    scored = sorted(((score_item(it, query_name), idx, it) for idx, it in enumerate(items)), key=lambda x: (x[0], x[1]))
    if len(scored) == 1 or scored[0][0] < scored[1][0]:
        return scored[0][2]

    options = []
    option_map: Dict[str, LineItem] = {}
    for _, _, it in scored:
        label = f"{it.fullname} | {it.type} | On hand: {it.on_hand:.0f} | {it.description}"
        options.append(label)
        option_map[label] = it
    selected = st.selectbox(
        f"Multiple SOS items found for '{query_name}'. Choose one:",
        options,
        key=f"{key_prefix}_{query_name}",
    )
    return option_map[selected]


def choose_item_batch(items: List[LineItem], query_name: str) -> Tuple[Optional[LineItem], str]:
    if not items:
        return None, "No match"
    scored = sorted(((score_item(it, query_name), idx, it) for idx, it in enumerate(items)), key=lambda x: (x[0], x[1]))
    chosen = scored[0][2]
    ambiguity = ""
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        ambiguity = "Multiple similar matches"
    return chosen, ambiguity


def build_location_matrix(client: SOSClient, item_ids: List[int], locations: List[str]) -> Dict[int, Dict[str, float]]:
    matrix: Dict[int, Dict[str, float]] = {item_id: {} for item_id in item_ids}
    for loc in locations:
        rows = client.get_items_by_id(item_ids, location=loc)
        by_id = {r.item_id: r for r in rows}
        for item_id in item_ids:
            matrix[item_id][loc] = by_id.get(item_id, LineItem(item_id, 0, "", 0, "", "", False)).on_hand
    return matrix


def report_rows_from_components(components: List[LineItem], location_matrix: Dict[int, Dict[str, float]], locations: List[str], source_label: str = "") -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    primary_location = locations[0] if locations else "Production Inventory"

    for comp in components:
        loc_map = location_matrix.get(comp.item_id, {})
        primary_qty = to_float(loc_map.get(primary_location, 0), 0.0)
        total_qty = sum(to_float(v, 0.0) for v in loc_map.values())
        needed = to_float(comp.quantity, 0.0)
        short = max(needed - primary_qty, 0.0)
        enough_primary = primary_qty >= needed
        enough_total = total_qty >= needed
        transfer_needed = (not enough_primary) and enough_total
        other_nonzero = [f"{loc}: {to_float(qty, 0.0):.0f}" for loc, qty in loc_map.items() if loc != primary_location and to_float(qty, 0.0) > 0]

        rec: Dict[str, Any] = {
            "Source": source_label,
            "Part Number": comp.fullname,
            "Needed": needed,
            f"At {primary_location}": primary_qty,
            "Total Available": total_qty,
            "Short at Primary": short,
            "Enough at Primary": "✅" if enough_primary else "❌",
            "Transfer Needed": "🟡" if transfer_needed else "",
            "Type": comp.type,
            "Description": comp.description,
            "Primary Location": primary_location,
            "Other Locations": " | ".join(other_nonzero),
            "Notes": trim_text(comp.notes, 50),
        }
        for loc in locations:
            rec[f"Loc: {loc}"] = to_float(loc_map.get(loc, 0), 0.0)
        records.append(rec)
    return pd.DataFrame(records)


# -----------------------------
# Rendering
# -----------------------------
def metric_row(df: pd.DataFrame, title: str, source: str):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Source", source)
    c2.metric("Rows returned", len(df))
    c3.metric("Shortage rows", int((df["Short at Primary"] > 0).sum()) if not df.empty else 0)
    c4.metric("Transfers possible", int((df["Transfer Needed"] == "🟡").sum()) if not df.empty else 0)
    st.caption(title)


def render_dashboard(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("Run a check first to populate the dashboard.")
        return
    st.subheader("Dashboard")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        show_short = st.checkbox("Show shortages only", value=False, key="dash_short")
    with col2:
        transfer_only = st.checkbox("Show transfer-needed only", value=False, key="dash_transfer")
    with col3:
        search = st.text_input("Search part / description / source", key="dash_search")

    filtered = df.copy()
    if show_short:
        filtered = filtered[filtered["Short at Primary"] > 0]
    if transfer_only:
        filtered = filtered[filtered["Transfer Needed"] == "🟡"]
    if search.strip():
        s = search.strip().lower()
        filtered = filtered[
            filtered.astype(str).apply(lambda col: col.str.lower().str.contains(s, na=False)).any(axis=1)
        ]

    st.dataframe(filtered, use_container_width=True, hide_index=True)

    if not filtered.empty:
        with st.expander("Summary by part"):
            by_part = (
                filtered.groupby("Part Number", as_index=False)
                .agg({"Needed": "sum", "Total Available": "sum", "Short at Primary": "sum"})
                .sort_values(["Short at Primary", "Part Number"], ascending=[False, True])
            )
            st.dataframe(by_part, use_container_width=True, hide_index=True)
        with st.expander("Summary by source"):
            by_source = (
                filtered.groupby("Source", as_index=False)
                .agg({"Needed": "sum", "Short at Primary": "sum"})
                .sort_values(["Short at Primary", "Source"], ascending=[False, True])
            )
            st.dataframe(by_source, use_container_width=True, hide_index=True)

        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered CSV",
            data=csv_bytes,
            file_name=safe_filename("wibotic_tool_c_dashboard"),
            mime="text/csv",
            use_container_width=True,
        )


def render_help():
    st.subheader("Help")
    st.markdown(
        """
**Single**
- Enter one item name or part number.
- If SOS returns multiple matches, choose the correct one.
- The report shows where the parts are across the selected SOS locations.

**Batch CSV**
- Upload a CSV with columns like `Part Number,Quantity`.
- The app picks the best SOS match for each row.
- If a row is ambiguous, it notes that in the report.

**Sales Order**
- Enter an SOS sales order number.
- The app loads order lines from SOS and checks component availability.

**How location logic works**
- The first selected location is treated as the **primary** one.
- `At <Primary>` shows availability there.
- `Total Available` shows inventory across all selected locations.
- `Transfer Needed` shows when the primary location is short, but enough stock exists in other locations.
"""
    )
    sample = "Part Number,Quantity\n140-000060,1\n130-000101,2\n"
    st.download_button(
        "Download CSV template",
        data=sample.encode("utf-8"),
        file_name="wibotic_tool_c_template.csv",
        mime="text/csv",
        use_container_width=True,
    )


# -----------------------------
# Main app
# -----------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("SOS inventory checker with item selection, multi-location availability, and transfer insight.")

    with st.sidebar:
        st.subheader("SOS connection")
        refresh_value = get_secret_or_input("SOS refresh token", "SOS_REFRESH_TOKEN")
        st.subheader("Locations")
        locations = st.multiselect(
            "Locations to include in report",
            options=DEFAULT_LOCATIONS,
            default=["Production Inventory", "Storage Unit", "PCA Components"],
        )
        st.caption("The first selected location is treated as the primary one.")

    if "report_df" not in st.session_state:
        st.session_state["report_df"] = pd.DataFrame()

    if not refresh_value.strip():
        st.warning("Enter the SOS refresh token in the sidebar to use the app.")
        render_help()
        return

    try:
        payload = refresh_token(refresh_value)
        client = SOSClient(payload["access_token"])
    except Exception as e:
        st.error(f"Could not connect to SOS: {e}")
        return

    if payload.get("refresh_token"):
        st.info("Connected to SOS. A refreshed token was returned during this session.")

    tab_single, tab_batch, tab_so, tab_dash, tab_help = st.tabs(
        ["Single", "Batch CSV", "Sales Order", "Dashboard", "Help"]
    )

    with tab_single:
        st.subheader("Single item check")
        item_name = st.text_input("Item name / part number", key="single_item_name")
        quantity = st.number_input("Quantity", min_value=1, value=1, step=1, key="single_qty")

        col_a, col_b = st.columns([1, 1])
        find = col_a.button("Find items", use_container_width=True)
        run = col_b.button("Run single check", use_container_width=True)

        if find and item_name.strip():
            st.session_state["single_matches"] = client.get_items_by_name(item_name)
            st.session_state["single_query"] = item_name

        matches: List[LineItem] = st.session_state.get("single_matches", [])
        selected_item: Optional[LineItem] = None
        if matches:
            selected_item = choose_item_interactive(matches, st.session_state.get("single_query", item_name), "single_choice")
            if selected_item:
                st.success(f"Selected: {selected_item.fullname} | {selected_item.type} | On hand: {selected_item.on_hand:.0f}")

        if run:
            if not matches and item_name.strip():
                matches = client.get_items_by_name(item_name)
                st.session_state["single_matches"] = matches
                st.session_state["single_query"] = item_name
            if not matches:
                st.warning("No SOS items found for that query.")
            else:
                selected_item = choose_item_interactive(matches, st.session_state.get("single_query", item_name), "single_choice_run")
                if selected_item is not None:
                    with st.spinner("Fetching BOM and availability across locations..."):
                        components = client.bom_lookup(selected_item, quantity=int(quantity))
                        matrix = build_location_matrix(client, [c.item_id for c in components], locations)
                        df = report_rows_from_components(components, matrix, locations, source_label=selected_item.fullname)
                        st.session_state["report_df"] = df
                        metric_row(df, "Live SOS fetch complete.", "Single item")
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        st.download_button(
                            "Download single-item CSV",
                            data=df.to_csv(index=False).encode("utf-8"),
                            file_name=safe_filename(f"{selected_item.fullname}_availability"),
                            mime="text/csv",
                            use_container_width=True,
                        )

    with tab_batch:
        st.subheader("Batch CSV")
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")
        if uploaded:
            st.download_button(
                "Download template",
                data=b"Part Number,Quantity\n140-000060,1\n",
                file_name="wibotic_tool_c_template.csv",
                mime="text/csv",
            )
            if st.button("Run batch check", use_container_width=True):
                try:
                    reqs = load_requests_from_csv(uploaded)
                except Exception as e:
                    st.error(str(e))
                    reqs = []

                rows = []
                with st.spinner("Checking batch items against SOS..."):
                    for name, qty in reqs:
                        items = client.get_items_by_name(name)
                        chosen, ambiguity = choose_item_batch(items, name)
                        if chosen is None:
                            rows.append({
                                "Source": name,
                                "Part Number": name,
                                "Needed": qty,
                                "At Production Inventory": 0,
                                "Total Available": 0,
                                "Short at Primary": qty,
                                "Enough at Primary": "❌",
                                "Transfer Needed": "",
                                "Type": "",
                                "Description": "No SOS match",
                                "Primary Location": locations[0] if locations else "",
                                "Other Locations": "",
                                "Notes": ambiguity or "No match",
                            })
                            continue

                        components = client.bom_lookup(chosen, quantity=qty)
                        matrix = build_location_matrix(client, [c.item_id for c in components], locations)
                        one_df = report_rows_from_components(components, matrix, locations, source_label=name)
                        if ambiguity:
                            one_df["Notes"] = one_df["Notes"].astype(str) + (" | " if one_df["Notes"].astype(str).ne("").any() else "") + ambiguity
                        rows.append(one_df)

                if rows:
                    df = pd.concat([r if isinstance(r, pd.DataFrame) else pd.DataFrame([r]) for r in rows], ignore_index=True)
                    st.session_state["report_df"] = df
                    metric_row(df, "Batch SOS fetch complete.", "Batch CSV")
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download batch CSV",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name=safe_filename("wibotic_tool_c_batch"),
                        mime="text/csv",
                        use_container_width=True,
                    )

    with tab_so:
        st.subheader("Sales Order")
        so_number = st.text_input("Sales order number", key="so_number")
        if st.button("Run sales-order check", use_container_width=True):
            if not so_number.strip():
                st.warning("Enter a sales order number.")
            else:
                so = client.get_sales_order_by_number(so_number.strip())
                if not so:
                    st.error("Sales order not found in SOS.")
                else:
                    detail = client.get_sales_order_detail(int(so["id"]))
                    reqs = client.sales_order_to_requests(detail)
                    rows = []
                    with st.spinner("Checking sales-order items against SOS..."):
                        for name, qty in reqs:
                            items = client.get_items_by_name(name)
                            chosen, ambiguity = choose_item_batch(items, name)
                            if chosen is None:
                                rows.append(pd.DataFrame([{
                                    "Source": name,
                                    "Part Number": name,
                                    "Needed": qty,
                                    "At Production Inventory": 0,
                                    "Total Available": 0,
                                    "Short at Primary": qty,
                                    "Enough at Primary": "❌",
                                    "Transfer Needed": "",
                                    "Type": "",
                                    "Description": "No SOS match",
                                    "Primary Location": locations[0] if locations else "",
                                    "Other Locations": "",
                                    "Notes": ambiguity or "No match",
                                }]))
                                continue
                            components = client.bom_lookup(chosen, quantity=qty)
                            matrix = build_location_matrix(client, [c.item_id for c in components], locations)
                            one_df = report_rows_from_components(components, matrix, locations, source_label=f"{safe_str(detail.get('number'))}: {name}")
                            if ambiguity:
                                one_df["Notes"] = one_df["Notes"].astype(str) + (" | " if one_df["Notes"].astype(str).ne("").any() else "") + ambiguity
                            rows.append(one_df)

                    if rows:
                        df = pd.concat(rows, ignore_index=True)
                        st.session_state["report_df"] = df
                        metric_row(df, "Sales-order SOS fetch complete.", safe_str(detail.get("number")) or "Sales order")
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        st.download_button(
                            "Download sales-order CSV",
                            data=df.to_csv(index=False).encode("utf-8"),
                            file_name=safe_filename(f"{safe_str(detail.get('number'))}_availability"),
                            mime="text/csv",
                            use_container_width=True,
                        )

    with tab_dash:
        render_dashboard(st.session_state.get("report_df", pd.DataFrame()))

    with tab_help:
        render_help()


if __name__ == "__main__":
    main()

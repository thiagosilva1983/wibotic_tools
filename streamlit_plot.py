import io
import json
import os
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt


APP_NAME = "CSV / TAR Data Plotter"
PAIR_RE = re.compile(r"^(RX|TX|INF)_(\d{4})\.(CSV|TML)$", re.IGNORECASE)


# =========================
# Session defaults
# =========================
def init_state():
    defaults = {
        "data": None,
        "loaded_name": "",
        "loaded_source_type": "",
        "loaded_tar_suffix": "",
        "plot_title": "",
        "signal_filter": "",
        "smoothing_mode": "None",
        "smoothing_window": 5,
        "x_axis_mode": "Seconds",
        "ignore_first_seconds": 60.0,
        "selected_columns": [],
        "scale_factors": {},
        "show_average_lines": True,
        "uploaded_file_bytes": None,
        "uploaded_file_name": "",
        "pair_mapping": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# =========================
# TAR helpers
# =========================
def scan_tar_bytes(file_bytes):
    mapping = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(file_bytes), mode="r:*") as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            for member in members:
                name = Path(member.name).name
                match = PAIR_RE.match(name)
                if not match:
                    continue
                kind, suffix, _ext = match.groups()
                entry = mapping.setdefault(suffix, {})
                entry[kind.upper()] = member.name
    except tarfile.ReadError as e:
        raise ValueError("Invalid or corrupted TAR file.") from e
    return mapping


def read_csv_from_tar(tf, member_name):
    extracted = tf.extractfile(member_name)
    if extracted is None:
        raise ValueError(f"Could not read TAR member: {member_name}")
    return pd.read_csv(extracted, low_memory=False)


def read_tml_lines(tf, member_name):
    extracted = tf.extractfile(member_name)
    if extracted is None:
        raise ValueError(f"Could not read TAR member: {member_name}")
    raw = extracted.read()
    return raw.decode("utf-8", errors="replace").splitlines()


def prefix_columns(df, prefix):
    df = df.copy()
    df.columns = [f"{prefix}{c}" for c in df.columns]
    return df


def build_unified_dataframe(rx_df, tx_df, tml_lines=None):
    max_len = max(len(rx_df), len(tx_df), len(tml_lines or []))
    rx_df = rx_df.reindex(range(max_len))
    tx_df = tx_df.reindex(range(max_len))

    rx_df = prefix_columns(rx_df, "Rx")
    tx_df = prefix_columns(tx_df, "Tx")

    tml_series = pd.Series(tml_lines or [], name="tml_info").reindex(range(max_len))
    return pd.concat([tml_series, tx_df, rx_df], axis=1)


def read_source_from_uploaded_file(file_name, file_bytes, selected_suffix=None):
    ext = Path(file_name).suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
        return df, "csv", ""

    if ext == ".tar":
        mapping = scan_tar_bytes(file_bytes)
        complete = sorted([s for s, entry in mapping.items() if "RX" in entry and "TX" in entry])

        if not complete:
            raise ValueError("No complete RX/TX pairs found in this TAR file.")

        suffix = selected_suffix or complete[0]
        if suffix not in mapping:
            raise ValueError(f"Suffix {suffix} was not found in the TAR file.")

        entry = mapping[suffix]
        with tarfile.open(fileobj=io.BytesIO(file_bytes), mode="r:*") as tf:
            rx_df = read_csv_from_tar(tf, entry["RX"])
            tx_df = read_csv_from_tar(tf, entry["TX"])
            tml_lines = read_tml_lines(tf, entry["INF"]) if "INF" in entry else None

        return build_unified_dataframe(rx_df, tx_df, tml_lines), "tar", suffix

    raise ValueError("Please upload a CSV or TAR file.")


# =========================
# Data preparation
# =========================
def safe_divide(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    result = pd.Series(np.nan, index=numerator.index, dtype="float64")
    valid = denominator.notna() & (denominator != 0) & numerator.notna()
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def add_calculated_columns(data):
    if "RxVBatt" in data.columns and "RxIBatt" in data.columns:
        rx_v = pd.to_numeric(data["RxVBatt"], errors="coerce")
        rx_i = pd.to_numeric(data["RxIBatt"], errors="coerce")
        data["RxPower"] = rx_v * rx_i

    if "TxVMonSys" in data.columns and "TxIMonSys" in data.columns:
        tx_vmon = pd.to_numeric(data["TxVMonSys"], errors="coerce")
        tx_imon = pd.to_numeric(data["TxIMonSys"], errors="coerce")
        data["TxInPower"] = tx_vmon * tx_imon

    if "TxVPA" in data.columns and "TxIPA" in data.columns:
        tx_vpa = pd.to_numeric(data["TxVPA"], errors="coerce")
        tx_ipa = pd.to_numeric(data["TxIPA"], errors="coerce")
        data["TxPaPower"] = tx_vpa * tx_ipa

    if "RxPower" in data.columns and "TxPaPower" in data.columns:
        tx_pa = pd.to_numeric(data["TxPaPower"], errors="coerce")
        rx_p = pd.to_numeric(data["RxPower"], errors="coerce")
        data["WirelessEfficiency"] = safe_divide(rx_p, tx_pa) * 100
        data["PowerLoss"] = tx_pa - rx_p

    if "TxPaPower" in data.columns and "TxInPower" in data.columns:
        tx_pa = pd.to_numeric(data["TxPaPower"], errors="coerce")
        tx_in = pd.to_numeric(data["TxInPower"], errors="coerce")
        data["TxDcEfficiency"] = safe_divide(tx_pa, tx_in) * 100

    if "TxTemp" in data.columns and "RxTemp" in data.columns:
        tx_t = pd.to_numeric(data["TxTemp"], errors="coerce")
        rx_t = pd.to_numeric(data["RxTemp"], errors="coerce")
        data["TempDelta"] = tx_t - rx_t


def prepare_loaded_dataframe(df):
    df = df.copy()

    if "TxRTC" in df.columns:
        df["TxRTC"] = pd.to_numeric(df["TxRTC"], errors="coerce")
        df = df.dropna(subset=["TxRTC"]).reset_index(drop=True)
        if df.empty:
            raise ValueError("'TxRTC' column could not be converted to numeric values.")
        df["Time_sec"] = df["TxRTC"] - df["TxRTC"].iloc[0]

    elif "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"]).reset_index(drop=True)
        if df.empty:
            raise ValueError("'Timestamp' column could not be converted to valid datetimes.")
        df["Time_sec"] = (df["Timestamp"] - df["Timestamp"].iloc[0]).dt.total_seconds()

    else:
        raise ValueError("Neither 'TxRTC' nor 'Timestamp' column is available for time calculation.")

    add_calculated_columns(df)
    return df


# =========================
# Helpers
# =========================
def friendly_label(col):
    if col == "TxTemp":
        return "TxTemp (DC-DC)"
    if col == "TxAmbTemp":
        return "TxAmbTemp (PA)"
    return col


def auto_fill_title_from_file(file_name, tar_suffix=""):
    base_name = Path(file_name).stem.strip()
    return f"{base_name} - {tar_suffix}" if tar_suffix else base_name


def sort_columns(columns):
    def key_fn(c):
        return (
            0 if c.startswith("Tx") else
            1 if c.startswith("Rx") else
            2 if c == "Time_sec" else
            3,
            c.lower()
        )
    return sorted(columns, key=key_fn)


def get_numeric_columns(df):
    numeric_cols = []
    for c in df.columns:
        if c == "tml_info":
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            numeric_cols.append(c)
    return numeric_cols


def smooth_series(series, mode, window):
    if mode.lower() == "none":
        return series

    window = max(1, int(window))

    if mode.lower() == "moving average":
        return series.rolling(window=window, min_periods=1).mean()
    if mode.lower() == "median":
        return series.rolling(window=window, min_periods=1).median()
    if mode.lower() == "ema":
        return series.ewm(span=window, adjust=False).mean()

    return series


def get_time_axis_values(filtered_data, mode):
    time_sec = pd.to_numeric(filtered_data["Time_sec"], errors="coerce")

    if mode == "Seconds":
        return time_sec.values, "Time (seconds)"
    if mode == "Minutes":
        return (time_sec / 60.0).values, "Time (minutes)"
    if mode == "Hours":
        return (time_sec / 3600.0).values, "Time (hours)"
    return np.arange(len(filtered_data)), "Sample index"


def downsample_for_plot(df, max_points=5000):
    if len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].copy()


def compute_stats_df(df, selected_columns, scale_factors):
    rows = []
    for col in selected_columns:
        if col == "Time_sec" or col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue

        scale = float(scale_factors.get(col, 1.0))
        scaled = s * scale

        rows.append({
            "Signal": col,
            "Min": scaled.min(),
            "Max": scaled.max(),
            "Avg": scaled.mean(),
            "Delta": scaled.max() - scaled.min(),
            "Std": scaled.std(),
        })

    return pd.DataFrame(rows)


def apply_preset(name, columns):
    name = name.lower()

    if name == "clear":
        return []

    if name == "temp":
        return [c for c in columns if "temp" in c.lower()]

    if name == "voltage":
        return [c for c in columns if any(k in c.lower() for k in ["vbat", "vpa", "vmon", "vcharge", "vrect", "volt"])]

    if name == "power":
        return [c for c in columns if ("power" in c.lower() or "efficiency" in c.lower() or "loss" in c.lower())]

    if name == "rx":
        return [c for c in columns if c.startswith("Rx")]

    if name == "tx":
        return [c for c in columns if c.startswith("Tx")]

    if name == "recommended":
        preferred_candidates = [
            "RxTemp", "TxTemp", "RxVBatt", "RxPower",
            "TxPaPower", "TxInPower", "WirelessEfficiency",
            "TxDcEfficiency", "PowerLoss", "TempDelta"
        ]
        return [c for c in preferred_candidates if c in columns]

    return []


def build_settings_payload():
    return {
        "plot_title": st.session_state.plot_title,
        "loaded_name": st.session_state.loaded_name,
        "loaded_source_type": st.session_state.loaded_source_type,
        "loaded_tar_suffix": st.session_state.loaded_tar_suffix,
        "signal_filter": st.session_state.signal_filter,
        "smoothing_mode": st.session_state.smoothing_mode,
        "smoothing_window": st.session_state.smoothing_window,
        "x_axis_mode": st.session_state.x_axis_mode,
        "ignore_first_seconds": st.session_state.ignore_first_seconds,
        "selected_columns": st.session_state.selected_columns,
        "scale_factors": st.session_state.scale_factors,
        "show_average_lines": st.session_state.show_average_lines,
    }


def apply_settings_payload(settings):
    st.session_state.plot_title = settings.get("plot_title", "")
    st.session_state.signal_filter = settings.get("signal_filter", "")
    st.session_state.smoothing_mode = settings.get("smoothing_mode", "None")
    st.session_state.smoothing_window = int(settings.get("smoothing_window", 5))
    st.session_state.x_axis_mode = settings.get("x_axis_mode", "Seconds")
    st.session_state.ignore_first_seconds = float(settings.get("ignore_first_seconds", 60.0))
    st.session_state.selected_columns = settings.get("selected_columns", [])
    st.session_state.scale_factors = settings.get("scale_factors", {})
    st.session_state.show_average_lines = settings.get("show_average_lines", True)


def export_filtered_dataframe(data, selected_columns, ignore_seconds, scale_factors, smoothing_mode, smoothing_window):
    time_numeric = pd.to_numeric(data["Time_sec"], errors="coerce")
    filtered_data = data[time_numeric > ignore_seconds].copy()

    if filtered_data.empty:
        raise ValueError("No rows remain after filtering.")

    export_cols = ["Time_sec"] + [c for c in selected_columns if c != "Time_sec"]
    export_df = filtered_data[export_cols].copy()

    for col in [c for c in selected_columns if c != "Time_sec"]:
        export_df[col] = pd.to_numeric(export_df[col], errors="coerce") * float(scale_factors.get(col, 1.0))
        export_df[f"{col}_smoothed"] = smooth_series(export_df[col], smoothing_mode, smoothing_window)

    return export_df


def make_plotly_figure(filtered_data, selected_columns, scale_factors, smoothing_mode, smoothing_window, x_axis_mode, title, show_average_lines):
    plot_df = downsample_for_plot(filtered_data, max_points=5000)
    x_data, x_label = get_time_axis_values(plot_df, x_axis_mode)

    fig = go.Figure()

    for col in selected_columns:
        if col == "Time_sec":
            continue

        y_data = pd.to_numeric(plot_df[col], errors="coerce") * float(scale_factors.get(col, 1.0))
        y_data = smooth_series(y_data, smoothing_mode, smoothing_window)

        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode="lines",
            name=f"{col} (Scale: {scale_factors.get(col, 1.0)})",
            hovertemplate=f"{col}<br>X=%{{x:.2f}}<br>Y=%{{y:.3f}}<extra></extra>"
        ))

        valid_y = pd.to_numeric(y_data, errors="coerce").dropna()
        if show_average_lines and not valid_y.empty:
            avg_value = valid_y.mean()
            fig.add_hline(y=avg_value, line_dash="dash", opacity=0.45)

    fig.update_layout(
        title=title or "Data Plot",
        xaxis_title=x_label,
        yaxis_title="Value",
        hovermode="x unified",
        legend_title="Signals",
        template="plotly_white",
        height=650,
    )
    return fig


def make_png_bytes(filtered_data, selected_columns, scale_factors, smoothing_mode, smoothing_window, x_axis_mode, title, show_average_lines):
    plot_df = downsample_for_plot(filtered_data, max_points=5000)
    x_data, x_label = get_time_axis_values(plot_df, x_axis_mode)

    fig, ax = plt.subplots(figsize=(12, 6.5))

    for col in selected_columns:
        if col == "Time_sec":
            continue

        y_data = pd.to_numeric(plot_df[col], errors="coerce") * float(scale_factors.get(col, 1.0))
        y_data = smooth_series(y_data, smoothing_mode, smoothing_window)

        ax.plot(x_data, y_data, label=f"{col} (Scale: {scale_factors.get(col, 1.0)})", linewidth=0.9)

        valid_y = pd.to_numeric(y_data, errors="coerce").dropna()
        if show_average_lines and not valid_y.empty:
            avg_value = valid_y.mean()
            ax.axhline(y=avg_value, linestyle="--", linewidth=0.7, alpha=0.6)

    ax.set_title(title or "Data Plot")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.45)
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# =========================
# App
# =========================
init_state()
st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)

with st.sidebar:
    st.header("Data Source")

    uploaded_file = st.file_uploader(
        "Upload CSV or TAR",
        type=["csv", "tar"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.uploaded_file_bytes = uploaded_file.getvalue()

        ext = Path(uploaded_file.name).suffix.lower()
        if ext == ".tar":
            try:
                mapping = scan_tar_bytes(st.session_state.uploaded_file_bytes)
                st.session_state.pair_mapping = mapping
                available_pairs = sorted([s for s, entry in mapping.items() if "RX" in entry and "TX" in entry])
            except Exception as e:
                available_pairs = []
                st.error(str(e))
        else:
            available_pairs = []
            st.session_state.pair_mapping = {}

        selected_pair = ""
        if ext == ".tar" and available_pairs:
            default_index = 0
            if st.session_state.loaded_tar_suffix in available_pairs:
                default_index = available_pairs.index(st.session_state.loaded_tar_suffix)

            selected_pair = st.selectbox("RX/TX Pair", available_pairs, index=default_index)
        else:
            selected_pair = ""

        if st.button("Load Data", use_container_width=True):
            try:
                loaded_df, source_type, tar_suffix = read_source_from_uploaded_file(
                    uploaded_file.name,
                    st.session_state.uploaded_file_bytes,
                    selected_pair if ext == ".tar" else None
                )
                loaded_df = prepare_loaded_dataframe(loaded_df)

                st.session_state.data = loaded_df
                st.session_state.loaded_name = uploaded_file.name
                st.session_state.loaded_source_type = source_type
                st.session_state.loaded_tar_suffix = tar_suffix

                new_title = auto_fill_title_from_file(
                    uploaded_file.name,
                    tar_suffix if source_type == "tar" else ""
                )
                st.session_state.plot_title = new_title

                numeric_columns = sort_columns(get_numeric_columns(loaded_df))
                for col in numeric_columns:
                    st.session_state.scale_factors.setdefault(col, 1.0)

                st.success("Data loaded successfully.")
            except Exception as e:
                st.error(f"Failed to load data: {e}")

    st.divider()
    st.header("Settings")

    settings_payload = build_settings_payload()
    settings_json = json.dumps(settings_payload, indent=2).encode("utf-8")

    st.download_button(
        "Download Settings JSON",
        data=settings_json,
        file_name="settings.json",
        mime="application/json",
        use_container_width=True
    )

    uploaded_settings = st.file_uploader(
        "Load Settings JSON",
        type=["json"],
        key="settings_uploader"
    )
    if uploaded_settings is not None:
        try:
            payload = json.load(uploaded_settings)
            apply_settings_payload(payload)
            st.success("Settings loaded.")
        except Exception as e:
            st.error(f"Failed to load settings: {e}")

data = st.session_state.data

left, right = st.columns([1.15, 2.2])

with left:
    st.subheader("Controls")

    if data is None:
        st.info("Upload a CSV or TAR file and click Load Data.")
    else:
        loaded_info = f"Loaded File: {st.session_state.loaded_name}"
        if st.session_state.loaded_source_type == "tar" and st.session_state.loaded_tar_suffix:
            loaded_info += f" | Pair: {st.session_state.loaded_tar_suffix}"
        st.caption(loaded_info)

        rows = len(data)
        cols = len(data.columns)
        tmin = data["Time_sec"].min() if "Time_sec" in data.columns else None
        tmax = data["Time_sec"].max() if "Time_sec" in data.columns else None

        if tmin is not None and tmax is not None:
            st.caption(f"Data summary: {rows} rows | {cols} columns | Time_sec {tmin:.2f} to {tmax:.2f}")
        else:
            st.caption(f"Data summary: {rows} rows | {cols} columns")

        st.session_state.plot_title = st.text_input("Plot Title", value=st.session_state.plot_title)

        st.markdown("### Smart Controls")
        st.session_state.signal_filter = st.text_input("Signal filter", value=st.session_state.signal_filter)

        col_a, col_b = st.columns(2)
        with col_a:
            st.session_state.smoothing_mode = st.selectbox(
                "Smoothing",
                ["None", "Moving Average", "Median", "EMA"],
                index=["None", "Moving Average", "Median", "EMA"].index(st.session_state.smoothing_mode)
            )
        with col_b:
            st.session_state.smoothing_window = st.number_input(
                "Window",
                min_value=1,
                value=int(st.session_state.smoothing_window),
                step=1
            )

        col_c, col_d = st.columns(2)
        with col_c:
            st.session_state.x_axis_mode = st.selectbox(
                "X axis",
                ["Seconds", "Minutes", "Hours", "Sample Index"],
                index=["Seconds", "Minutes", "Hours", "Sample Index"].index(st.session_state.x_axis_mode)
            )
        with col_d:
            st.session_state.ignore_first_seconds = st.number_input(
                "Ignore first sec",
                min_value=0.0,
                value=float(st.session_state.ignore_first_seconds),
                step=1.0
            )

        st.session_state.show_average_lines = st.checkbox(
            "Show average lines",
            value=st.session_state.show_average_lines
        )

        st.markdown("### Presets")
        preset_cols = st.columns(7)
        preset_names = ["Recommended", "Temp", "Voltage", "Power", "Rx", "Tx", "Clear"]
        numeric_columns = sort_columns(get_numeric_columns(data))

        for idx, name in enumerate(preset_names):
            if preset_cols[idx].button(name, use_container_width=True):
                st.session_state.selected_columns = apply_preset(name, numeric_columns)

        st.markdown("### Signals")

        filtered_columns = [
            c for c in numeric_columns
            if st.session_state.signal_filter.strip().lower() in c.lower()
        ]

        if not filtered_columns:
            st.warning("No signals match the current filter.")
        else:
            selected_set = set(st.session_state.selected_columns)
            updated_selected = []

            for col in filtered_columns:
                row1, row2 = st.columns([1.8, 1.0])
                checked = row1.checkbox(
                    friendly_label(col),
                    value=(col in selected_set),
                    key=f"check_{col}"
                )
                scale_val = row2.number_input(
                    f"Scale {col}",
                    value=float(st.session_state.scale_factors.get(col, 1.0)),
                    key=f"scale_{col}",
                    label_visibility="collapsed"
                )
                st.session_state.scale_factors[col] = scale_val

                if checked:
                    updated_selected.append(col)

            still_selected_hidden = [
                c for c in st.session_state.selected_columns
                if c not in filtered_columns and c in numeric_columns
            ]
            st.session_state.selected_columns = sorted(set(updated_selected + still_selected_hidden))

with right:
    st.subheader("Plot and Statistics")

    if data is None:
        st.info("No data loaded yet.")
    else:
        selected_columns = [c for c in st.session_state.selected_columns if c in data.columns]

        if not selected_columns:
            st.warning("Select at least one signal.")
        else:
            try:
                ignore_seconds = float(st.session_state.ignore_first_seconds)
                time_numeric = pd.to_numeric(data["Time_sec"], errors="coerce")
                filtered_data = data[time_numeric > ignore_seconds].copy()

                if filtered_data.empty:
                    st.error(f"No rows remain after ignoring the first {ignore_seconds} seconds.")
                else:
                    for col in selected_columns:
                        converted = pd.to_numeric(filtered_data[col], errors="coerce")
                        if converted.notna().sum() == 0:
                            raise ValueError(f"The column '{col}' has no numeric data to plot.")

                    fig = make_plotly_figure(
                        filtered_data=filtered_data,
                        selected_columns=selected_columns,
                        scale_factors=st.session_state.scale_factors,
                        smoothing_mode=st.session_state.smoothing_mode,
                        smoothing_window=st.session_state.smoothing_window,
                        x_axis_mode=st.session_state.x_axis_mode,
                        title=st.session_state.plot_title,
                        show_average_lines=st.session_state.show_average_lines,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    stats_df = compute_stats_df(
                        filtered_data,
                        selected_columns,
                        st.session_state.scale_factors
                    )

                    st.markdown("### Selection Statistics")
                    if stats_df.empty:
                        st.info("No numeric stats available for selected columns.")
                    else:
                        st.dataframe(
                            stats_df.style.format({
                                "Min": "{:.3f}",
                                "Max": "{:.3f}",
                                "Avg": "{:.3f}",
                                "Delta": "{:.3f}",
                                "Std": "{:.3f}",
                            }),
                            use_container_width=True
                        )

                    st.markdown("### Export")
                    export_df = export_filtered_dataframe(
                        data=data,
                        selected_columns=selected_columns,
                        ignore_seconds=ignore_seconds,
                        scale_factors=st.session_state.scale_factors,
                        smoothing_mode=st.session_state.smoothing_mode,
                        smoothing_window=st.session_state.smoothing_window
                    )

                    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
                    png_bytes = make_png_bytes(
                        filtered_data=filtered_data,
                        selected_columns=selected_columns,
                        scale_factors=st.session_state.scale_factors,
                        smoothing_mode=st.session_state.smoothing_mode,
                        smoothing_window=st.session_state.smoothing_window,
                        x_axis_mode=st.session_state.x_axis_mode,
                        title=st.session_state.plot_title,
                        show_average_lines=st.session_state.show_average_lines,
                    )

                    file_base = Path(st.session_state.loaded_name).stem or "plot"

                    dl1, dl2 = st.columns(2)
                    dl1.download_button(
                        "Download Filtered CSV",
                        data=csv_bytes,
                        file_name=f"{file_base}_filtered.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    dl2.download_button(
                        "Download Plot PNG",
                        data=png_bytes,
                        file_name=f"{file_base}_plot.png",
                        mime="image/png",
                        use_container_width=True
                    )

                    with st.expander("Preview filtered data"):
                        st.dataframe(export_df, use_container_width=True)

            except Exception as e:
                st.error(f"Failed to process plot: {e}")

import io
import json
import os
import re
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Derated + Arduino Cloud', layout='wide')

PAIR_RE = re.compile(r'^(RX|TX|INF)_(\d{4})\.(CSV|TML)$', re.IGNORECASE)

# -----------------------------
# Shared helpers
# -----------------------------
def scan_tar_bytes(file_bytes: bytes):
    mapping = {}
    with tarfile.open(fileobj=io.BytesIO(file_bytes), mode='r') as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        for member in members:
            name = Path(member.name).name
            match = PAIR_RE.match(name)
            if not match:
                continue
            kind, suffix, _ext = match.groups()
            entry = mapping.setdefault(suffix, {})
            entry[kind.upper()] = member.name
    return mapping


def read_csv_from_tar(tf, member_name):
    with tf.extractfile(member_name) as f:
        return pd.read_csv(f, low_memory=False)


def read_tml_lines(tf, member_name):
    with tf.extractfile(member_name) as f:
        raw = f.read()
    return raw.decode('utf-8', errors='replace').splitlines()


def prefix_columns(df, prefix):
    df = df.copy()
    df.columns = [f'{prefix}{c}' for c in df.columns]
    return df


def build_unified_dataframe(rx_df, tx_df, tml_lines=None):
    max_len = max(len(rx_df), len(tx_df), len(tml_lines or []))
    rx_df = rx_df.reindex(range(max_len))
    tx_df = tx_df.reindex(range(max_len))
    rx_df = prefix_columns(rx_df, 'Rx')
    tx_df = prefix_columns(tx_df, 'Tx')
    tml_series = pd.Series(tml_lines or [], name='tml_info').reindex(range(max_len))
    return pd.concat([tml_series, tx_df, rx_df], axis=1)


def read_source_uploaded(uploaded_file, selected_suffix=None):
    name = uploaded_file.name.lower()
    if name.endswith('.csv'):
        return pd.read_csv(uploaded_file, low_memory=False), 'csv', ''
    if name.endswith('.tar'):
        file_bytes = uploaded_file.getvalue()
        mapping = scan_tar_bytes(file_bytes)
        complete = sorted([s for s, entry in mapping.items() if 'RX' in entry and 'TX' in entry])
        if not complete:
            raise ValueError('No complete RX/TX pairs found in this TAR file.')
        suffix = selected_suffix or complete[0]
        if suffix not in mapping:
            raise ValueError(f'Suffix {suffix} was not found in the TAR file.')
        entry = mapping[suffix]
        with tarfile.open(fileobj=io.BytesIO(file_bytes), mode='r') as tf:
            rx_df = read_csv_from_tar(tf, entry['RX'])
            tx_df = read_csv_from_tar(tf, entry['TX'])
            tml_lines = read_tml_lines(tf, entry['INF']) if 'INF' in entry else None
        return build_unified_dataframe(rx_df, tx_df, tml_lines), 'tar', suffix
    raise ValueError('Please upload a CSV or TAR file.')


def safe_divide(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors='coerce')
    denominator = pd.to_numeric(denominator, errors='coerce')
    result = pd.Series(np.nan, index=numerator.index, dtype='float64')
    valid = denominator.notna() & (denominator != 0) & numerator.notna()
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def add_calculated_columns(data):
    if 'RxVBatt' in data.columns and 'RxIBatt' in data.columns:
        data['RxPower'] = pd.to_numeric(data['RxVBatt'], errors='coerce') * pd.to_numeric(data['RxIBatt'], errors='coerce')
    if 'TxVMonSys' in data.columns and 'TxIMonSys' in data.columns:
        data['TxInPower'] = pd.to_numeric(data['TxVMonSys'], errors='coerce') * pd.to_numeric(data['TxIMonSys'], errors='coerce')
    if 'TxVPA' in data.columns and 'TxIPA' in data.columns:
        data['TxPaPower'] = pd.to_numeric(data['TxVPA'], errors='coerce') * pd.to_numeric(data['TxIPA'], errors='coerce')
    if 'RxPower' in data.columns and 'TxPaPower' in data.columns:
        tx_pa = pd.to_numeric(data['TxPaPower'], errors='coerce')
        rx_p = pd.to_numeric(data['RxPower'], errors='coerce')
        data['WirelessEfficiency'] = safe_divide(rx_p, tx_pa) * 100
        data['PowerLoss'] = tx_pa - rx_p
    if 'TxPaPower' in data.columns and 'TxInPower' in data.columns:
        data['TxDcEfficiency'] = safe_divide(pd.to_numeric(data['TxPaPower'], errors='coerce'), pd.to_numeric(data['TxInPower'], errors='coerce')) * 100
    if 'TxTemp' in data.columns and 'RxTemp' in data.columns:
        data['TempDelta'] = pd.to_numeric(data['TxTemp'], errors='coerce') - pd.to_numeric(data['RxTemp'], errors='coerce')


def prepare_loaded_dataframe(df):
    df = df.copy()
    if 'TxRTC' in df.columns:
        df['TxRTC'] = pd.to_numeric(df['TxRTC'], errors='coerce')
        df = df.dropna(subset=['TxRTC']).reset_index(drop=True)
        if df.empty:
            raise ValueError("'TxRTC' column could not be converted to numeric values.")
        df['Time_sec'] = df['TxRTC'] - df['TxRTC'].iloc[0]
    elif 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.dropna(subset=['Timestamp']).reset_index(drop=True)
        if df.empty:
            raise ValueError("'Timestamp' column could not be converted to valid datetimes.")
        df['Time_sec'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds()
    else:
        raise ValueError("Neither 'TxRTC' nor 'Timestamp' column is available for time calculation.")
    add_calculated_columns(df)
    return df


def preprocess_chamber_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    if 'time' not in df.columns or 'value' not in df.columns:
        raise ValueError("Chamber CSV must contain 'time' and 'value' columns.")
    df['time_utc'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
    df['time_pacific'] = df['time_utc'].dt.tz_convert('America/Los_Angeles')
    df['temperature_c'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['time_pacific', 'temperature_c']).sort_values('time_pacific').reset_index(drop=True)
    if df.empty:
        raise ValueError('No valid chamber rows found after parsing the CSV.')
    return df


def preprocess_arduino_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    if 'time' not in df.columns:
        raise ValueError("Arduino CSV must contain a 'time' column.")
    df['time_utc'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
    df['time_pacific'] = df['time_utc'].dt.tz_convert('America/Los_Angeles')
    df = df.dropna(subset=['time_pacific']).sort_values('time_pacific').reset_index(drop=True)
    return df


def parse_pacific_datetime(text):
    text = (text or '').strip()
    if not text:
        return None
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M'):
        try:
            return pd.to_datetime(text, format=fmt).tz_localize('America/Los_Angeles')
        except Exception:
            pass
    ts = pd.to_datetime(text, errors='raise')
    if ts.tzinfo is None:
        return ts.tz_localize('America/Los_Angeles')
    return ts.tz_convert('America/Los_Angeles')


def apply_chamber_filter(cdf, filter_mode='Moving Average', smooth_seconds=10):
    cdf = cdf.copy().sort_values('time_pacific')
    smooth_seconds = max(1, int(smooth_seconds or 1))
    mode = (filter_mode or 'Moving Average').strip().lower()
    cdf = cdf.set_index('time_pacific')
    if mode in ('none', 'raw') or smooth_seconds <= 1:
        pass
    elif mode in ('moving average', 'rolling mean', 'mean'):
        cdf['temperature_c'] = cdf['temperature_c'].rolling(f'{smooth_seconds}s', min_periods=1).mean()
    elif mode in ('median', 'rolling median'):
        cdf['temperature_c'] = cdf['temperature_c'].rolling(f'{smooth_seconds}s', min_periods=1).median()
    elif mode == 'ema':
        cdf['temperature_c'] = cdf['temperature_c'].ewm(span=max(2, smooth_seconds), adjust=False, min_periods=1).mean()
    else:
        raise ValueError(f'Unsupported chamber filter mode: {filter_mode}')
    return cdf.reset_index()


def align_chamber_to_main_data(main_df, chamber_source_df, start_text, end_text='', smooth_seconds=10, filter_mode='Moving Average'):
    if main_df is None:
        raise ValueError('Load main TAR/CSV data first.')
    cdf = apply_chamber_filter(chamber_source_df, filter_mode=filter_mode, smooth_seconds=smooth_seconds)
    if 'Timestamp' in main_df.columns and pd.to_datetime(main_df['Timestamp'], errors='coerce').notna().sum() > 0:
        left = main_df.copy()
        left['main_time_abs'] = pd.to_datetime(left['Timestamp'], errors='coerce')
        if getattr(left['main_time_abs'].dt, 'tz', None) is None:
            left['main_time_abs'] = left['main_time_abs'].dt.tz_localize('America/Los_Angeles', nonexistent='shift_forward', ambiguous='NaT')
        else:
            left['main_time_abs'] = left['main_time_abs'].dt.tz_convert('America/Los_Angeles')
        merged = pd.merge_asof(
            left.sort_values('main_time_abs'),
            cdf[['time_pacific', 'temperature_c']].sort_values('time_pacific'),
            left_on='main_time_abs',
            right_on='time_pacific',
            direction='nearest',
            tolerance=pd.Timedelta(seconds=max(10, int(smooth_seconds) * 3))
        )
        aligned = merged.sort_index()
        aligned['ChamberTemp'] = aligned['temperature_c']
        return aligned.drop(columns=['temperature_c'], errors='ignore')

    start_ts = parse_pacific_datetime(start_text)
    end_ts = parse_pacific_datetime(end_text) if (end_text or '').strip() else None
    rel = (cdf['time_pacific'] - start_ts).dt.total_seconds()
    cdf = cdf.assign(rel_sec=rel)
    cdf = cdf[cdf['rel_sec'].notna()]
    if end_ts is not None:
        cdf = cdf[cdf['time_pacific'] <= end_ts]
    tmax = pd.to_numeric(main_df['Time_sec'], errors='coerce').max()
    cdf = cdf[(cdf['rel_sec'] >= 0) & (cdf['rel_sec'] <= tmax)]
    if cdf.empty:
        raise ValueError('No overlapping chamber data found in the selected manual time range.')
    x = cdf['rel_sec'].to_numpy(dtype=float)
    y = cdf['temperature_c'].to_numpy(dtype=float)
    target_x = pd.to_numeric(main_df['Time_sec'], errors='coerce').to_numpy(dtype=float)
    interp = np.interp(target_x, x, y, left=np.nan, right=np.nan)
    aligned = main_df.copy()
    aligned['ChamberTemp'] = interp
    return aligned


def generate_derate_artifacts(data, power_col, ignore_seconds, end_window_sec, trim_last_sec, bin_step, title):
    df = data.copy()
    df['Time_sec'] = pd.to_numeric(df['Time_sec'], errors='coerce')
    df[power_col] = pd.to_numeric(df[power_col], errors='coerce')
    df['ChamberTemp'] = pd.to_numeric(df['ChamberTemp'], errors='coerce')
    df = df.dropna(subset=['Time_sec', power_col, 'ChamberTemp']).copy()
    df = df[df['Time_sec'] > ignore_seconds]
    if trim_last_sec > 0 and not df.empty:
        df = df[df['Time_sec'] <= (df['Time_sec'].max() - trim_last_sec)]
    if df.empty:
        raise ValueError('No rows remain after ignore/trim filters.')

    end_start = max(df['Time_sec'].min(), df['Time_sec'].max() - end_window_sec)
    end_df = df[df['Time_sec'] >= end_start].copy()
    if end_df.empty:
        end_df = df.copy()

    summary = {
        'power_signal': power_col,
        'window_start_sec': float(end_df['Time_sec'].min()),
        'window_end_sec': float(end_df['Time_sec'].max()),
        'avg_chamber_temp_c': float(end_df['ChamberTemp'].mean()),
        'median_chamber_temp_c': float(end_df['ChamberTemp'].median()),
        'avg_power_w': float(end_df[power_col].mean()),
        'median_power_w': float(end_df[power_col].median()),
        'min_power_w': float(end_df[power_col].min()),
        'max_power_w': float(end_df[power_col].max()),
        'samples_in_window': int(len(end_df)),
    }

    bins_df = df[['ChamberTemp', power_col]].copy()
    bins_df['temp_bin_c'] = (np.round(bins_df['ChamberTemp'] / bin_step) * bin_step).round(2)
    curve = (
        bins_df.groupby('temp_bin_c')[power_col]
        .agg(mean_power_w='mean', median_power_w='median', count='count')
        .reset_index()
        .sort_values('temp_bin_c')
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 7.2), height_ratios=[2.0, 1.25])
    ax1.plot(df['Time_sec'] / 60.0, df[power_col], label=power_col, linewidth=1.0)
    ax1b = ax1.twinx()
    ax1b.plot(df['Time_sec'] / 60.0, df['ChamberTemp'], label='ChamberTemp', linewidth=1.0)
    ax1.axvspan(end_df['Time_sec'].min() / 60.0, end_df['Time_sec'].max() / 60.0, alpha=0.15)
    ax1.set_title(f"{title} | End-window avg power {summary['avg_power_w']:.2f} W at {summary['avg_chamber_temp_c']:.2f} °C")
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel(f'{power_col} (W)')
    ax1b.set_ylabel('Chamber Temp (°C)')
    ax1.grid(True, alpha=0.35)

    ax2.plot(curve['temp_bin_c'], curve['mean_power_w'], marker='o', linewidth=1.0)
    ax2.set_xlabel('Chamber Temp (°C)')
    ax2.set_ylabel(f'{power_col} mean (W)')
    ax2.set_title('Delivered power vs environment temperature')
    ax2.grid(True, alpha=0.35)
    fig.tight_layout()
    return fig, pd.DataFrame([summary]), curve, end_df[['Time_sec', 'ChamberTemp', power_col]].copy()


# -----------------------------
# UI
# -----------------------------
st.title('Derated + Arduino Cloud')

derate_tab, arduino_tab = st.tabs(['Derated', 'Arduino'])

with derate_tab:
    st.subheader('Derate report')
    left, right = st.columns([1, 1])
    with left:
        main_file = st.file_uploader('Main TAR/CSV', type=['tar', 'csv'], key='main_file')
        suffix = None
        suffix_options = []
        if main_file is not None and main_file.name.lower().endswith('.tar'):
            try:
                suffix_options = sorted([s for s, entry in scan_tar_bytes(main_file.getvalue()).items() if 'RX' in entry and 'TX' in entry])
                suffix = st.selectbox('RX/TX pair', suffix_options, index=0)
            except Exception as e:
                st.error(f'Could not inspect TAR: {e}')
        chamber_file = st.file_uploader('Chamber CSV', type=['csv'], key='chamber_file')
        title = st.text_input('Plot title', value='Derate Report')
        start_text = st.text_input('Manual start Pacific', value='2026-03-19 15:20')
        end_text = st.text_input('Manual end Pacific', value='2026-03-19 16:20')
    with right:
        filter_mode = st.selectbox('Chamber filter', ['Moving Average', 'Median', 'EMA', 'None'], index=0)
        smooth_seconds = st.number_input('Chamber smooth seconds', min_value=1, value=10, step=1)
        ignore_seconds = st.number_input('Ignore first seconds', min_value=0.0, value=60.0, step=10.0)
        end_window_sec = st.number_input('End window seconds', min_value=30.0, value=120.0, step=30.0)
        trim_last_sec = st.number_input('Trim last seconds', min_value=0.0, value=0.0, step=10.0)
        bin_step = st.number_input('Temperature bin step', min_value=0.25, value=0.5, step=0.25)

    prepared_df = None
    if main_file is not None:
        try:
            raw_df, source_type, chosen_suffix = read_source_uploaded(main_file, suffix)
            prepared_df = prepare_loaded_dataframe(raw_df)
            st.caption(f'Loaded {main_file.name} | source={source_type} | pair={chosen_suffix}')
            power_choices = [c for c in prepared_df.columns if c.lower().endswith('power') or 'efficiency' in c.lower() or c in ('RxPower', 'TxPaPower')]
            if 'RxPower' in prepared_df.columns:
                default_idx = power_choices.index('RxPower') if 'RxPower' in power_choices else 0
            else:
                default_idx = 0 if power_choices else None
        except Exception as e:
            st.error(f'Failed to load main file: {e}')
            power_choices = []
            default_idx = None
    else:
        power_choices = []
        default_idx = None

    power_col = st.selectbox('Power signal', power_choices, index=default_idx if default_idx is not None and power_choices else 0)

    if st.button('Generate derate report', type='primary'):
        try:
            if prepared_df is None:
                raise ValueError('Upload the main TAR/CSV first.')
            if chamber_file is None:
                raise ValueError('Upload the chamber CSV first.')
            chamber_df = preprocess_chamber_csv(chamber_file)
            aligned_df = align_chamber_to_main_data(prepared_df, chamber_df, start_text, end_text, int(smooth_seconds), filter_mode)
            if 'ChamberTemp' not in aligned_df.columns:
                raise ValueError('ChamberTemp could not be aligned.')
            fig, summary_df, curve_df, window_df = generate_derate_artifacts(
                aligned_df, power_col, float(ignore_seconds), float(end_window_sec), float(trim_last_sec), float(bin_step), title
            )
            st.pyplot(fig, clear_figure=False)
            s = summary_df.iloc[0]
            st.success(f"Avg power {s['avg_power_w']:.2f} W at {s['avg_chamber_temp_c']:.2f} °C")
            c1, c2, c3 = st.columns(3)
            c1.dataframe(summary_df, use_container_width=True)
            c2.dataframe(curve_df, use_container_width=True, height=260)
            c3.dataframe(window_df.head(200), use_container_width=True, height=260)
            st.download_button('Download summary CSV', summary_df.to_csv(index=False).encode('utf-8'), file_name='derate_summary.csv', mime='text/csv')
            st.download_button('Download curve CSV', curve_df.to_csv(index=False).encode('utf-8'), file_name='derate_curve.csv', mime='text/csv')
            st.download_button('Download window CSV', window_df.to_csv(index=False).encode('utf-8'), file_name='derate_window.csv', mime='text/csv')
        except Exception as e:
            st.error(str(e))

with arduino_tab:
    st.subheader('Arduino Cloud viewer')
    arduino_file = st.file_uploader('Arduino CSV', type=['csv'], key='arduino_file')
    if arduino_file is not None:
        try:
            adf = preprocess_arduino_csv(arduino_file)
            st.caption(f"Rows: {len(adf)} | Pacific range: {adf['time_pacific'].min()} to {adf['time_pacific'].max()}")
            pacific_dates = sorted(adf['time_pacific'].dt.date.astype(str).unique().tolist())
            selected_date = st.selectbox('Pacific date', pacific_dates)
            day_df = adf[adf['time_pacific'].dt.date.astype(str) == selected_date].copy()
            numeric_cols = [c for c in day_df.columns if c not in ['time', 'time_utc', 'time_pacific'] and pd.to_numeric(day_df[c], errors='coerce').notna().sum() > 0]
            if not numeric_cols:
                st.warning('No numeric value columns found to plot.')
            else:
                y_col = st.selectbox('Signal', numeric_cols)
                mode = st.radio('Time selection mode', ['Available times from file', 'Manual HH:MM entry'], horizontal=True)
                if mode == 'Available times from file':
                    times = day_df['time_pacific'].dt.strftime('%H:%M').drop_duplicates().tolist()
                    start_hhmm = st.selectbox('Start time', times, index=0)
                    end_hhmm = st.selectbox('End time', times, index=len(times)-1)
                else:
                    c1, c2 = st.columns(2)
                    start_hhmm = c1.text_input('Start HH:MM', value='10:15')
                    end_hhmm = c2.text_input('End HH:MM', value='11:00')
                start_ts = parse_pacific_datetime(f'{selected_date} {start_hhmm}')
                end_ts = parse_pacific_datetime(f'{selected_date} {end_hhmm}')
                plot_df = day_df[(day_df['time_pacific'] >= start_ts) & (day_df['time_pacific'] <= end_ts)].copy()
                plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors='coerce')
                plot_df = plot_df.dropna(subset=[y_col])
                if plot_df.empty:
                    st.warning('No rows in that range.')
                else:
                    fig = plt.figure(figsize=(10, 4.8))
                    ax = fig.add_subplot(111)
                    ax.plot(plot_df['time_pacific'], plot_df[y_col], linewidth=1.2)
                    ax.set_title(f'{y_col} | {selected_date} | {start_hhmm} to {end_hhmm} Pacific')
                    ax.set_xlabel('Pacific time')
                    ax.set_ylabel(y_col)
                    ax.grid(True, alpha=0.35)
                    fig.autofmt_xdate()
                    st.pyplot(fig, clear_figure=False)
                    st.dataframe(plot_df[['time_pacific', y_col]].head(500), use_container_width=True, height=260)
        except Exception as e:
            st.error(f'Failed to load Arduino CSV: {e}')
    else:
        st.info('Upload an Arduino Cloud CSV to use this tab.')

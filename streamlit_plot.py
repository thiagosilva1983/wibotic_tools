import io
import base64
import re
import math
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import inch
from reportlab.graphics.barcode import code39
from PIL import Image, ImageDraw, ImageFont
import csv
import tempfile
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests

st.set_page_config(page_title='Derated + Arduino + Plot', layout='wide')

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
            raise ValueError(f'Suffix {suffix} was not found in this TAR file.')
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


def friendly_label(col):
    return 'TxTemp (DC-DC)' if col == 'TxTemp' else 'TxAmbTemp (PA)' if col == 'TxAmbTemp' else col


def get_plot_columns(df):
    cols = []
    for c in df.columns:
        if c == 'Time_sec':
            continue
        if pd.to_numeric(df[c], errors='coerce').notna().sum() > 0:
            cols.append(c)
    return cols


def apply_preset(columns, preset):
    p = preset.lower()
    if p == 'clear':
        return []
    if p == 'temp':
        return [c for c in columns if 'temp' in c.lower()]
    if p == 'voltage':
        return [c for c in columns if any(k in c.lower() for k in ['vbat', 'vpa', 'vmon', 'vcharge', 'vrect', 'volt'])]
    if p == 'power':
        return [c for c in columns if 'power' in c.lower() or 'efficiency' in c.lower() or 'loss' in c.lower()]
    if p == 'rx':
        return [c for c in columns if c.startswith('Rx')]
    if p == 'tx':
        return [c for c in columns if c.startswith('Tx')]
    if p == 'recommended':
        preferred = {'RxTemp', 'TxTemp', 'RxVBatt', 'RxPower', 'TxPaPower', 'WirelessEfficiency', 'TempDelta'}
        return [c for c in columns if c in preferred]
    return []


def smooth_series(series, mode, window):
    mode = (mode or 'None').strip().lower()
    window = max(1, int(window or 1))
    if mode == 'none':
        return series
    if mode == 'moving average':
        return series.rolling(window=window, min_periods=1).mean()
    if mode == 'median':
        return series.rolling(window=window, min_periods=1).median()
    if mode == 'ema':
        return series.ewm(span=window, adjust=False).mean()
    return series


def get_time_axis_values(time_sec, mode):
    mode = (mode or 'Seconds').strip().lower()
    time_sec = pd.to_numeric(time_sec, errors='coerce')
    if mode == 'seconds':
        return time_sec.to_numpy(dtype=float), 'Time (seconds)'
    if mode == 'minutes':
        return (time_sec / 60.0).to_numpy(dtype=float), 'Time (minutes)'
    if mode == 'hours':
        return (time_sec / 3600.0).to_numpy(dtype=float), 'Time (hours)'
    return np.arange(len(time_sec), dtype=float), 'Sample index'


def compute_stats_text(df, selected_columns, scale_map):
    rows = []
    for col in selected_columns:
        s = pd.to_numeric(df[col], errors='coerce').dropna()
        if s.empty:
            continue
        scaled = s * float(scale_map.get(col, 1.0))
        rows.append({
            'signal': col,
            'min': float(scaled.min()),
            'max': float(scaled.max()),
            'avg': float(scaled.mean()),
            'delta': float(scaled.max() - scaled.min()),
            'std': float(scaled.std() if len(scaled) > 1 else 0.0),
        })
    return pd.DataFrame(rows)



# -----------------------------
# Label helpers
# -----------------------------
LOGO_B64 = "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"




JABIL_CUSTOMER_PN_MAP = {
    "130-000071": "JBLTR1000ACST",
    "140-000059": "JBLOC100036ST",
    "140-000060": "JBRC1501000SL",
    "130-000077": "JBLTC2001000SL",
}

def get_logo_image():
    try:
        return Image.open(io.BytesIO(base64.b64decode(LOGO_B64)))
    except Exception:
        return None


def clean_part_name(text: str) -> str:
    t = re.sub(r'\b(?:S\/?N|SN|MAC)\b\s*[:\-]?\s*[^|\r\n]+', '', str(text or ''), flags=re.IGNORECASE)
    t = re.sub(r'\s*\|\s*', ' ', t)
    t = re.sub(r'\s{2,}', ' ', t).strip(' -|')
    return t.strip()

def parse_nabtesco_labels(uploaded_file):
    uploaded_file.seek(0)
    rows = []
    processed_serials = set()
    added_unserialized_parts = set()
    text_io = io.StringIO(uploaded_file.getvalue().decode('utf-8-sig', errors='replace'))
    reader = csv.DictReader(text_io)
    for row in reader:
        part_number = (row.get('Item') or row.get('Part Number') or 'Unknown').strip()
        desc_raw = (row.get('Description') or row.get('Part Name') or 'Unknown').strip()
        quantity_str = (row.get('Shipped') or row.get('Quantity') or '1').strip()

        m_mn = re.search(r'\bMN\b\s*[:\-]?\s*([^|\s,;]+)', desc_raw, re.IGNORECASE)
        part_name = m_mn.group(1).strip() if m_mn else clean_part_name(desc_raw)

        serialized = False
        sn_blocks = re.findall(r'(?:\bS\/?N\b|\bSN\b)\s*[:\-]?\s*([^|\r\n]+)', desc_raw, flags=re.IGNORECASE)
        for block in sn_blocks:
            for serial_number in re.split(r'[,\s;|]+', block.strip()):
                serial_number = serial_number.strip()
                if serial_number and serial_number not in processed_serials:
                    processed_serials.add(serial_number)
                    rows.append({
                        'Part Number': part_number,
                        'Part Name': part_name,
                        'Serial Number': serial_number,
                        'Quantity': '1',
                    })
                    serialized = True

        if not serialized:
            raw_sn = (row.get('Serial Number') or row.get('S/N') or row.get('SN') or '').strip()
            if raw_sn:
                for serial_number in re.split(r'[,\s;|]+', raw_sn):
                    serial_number = serial_number.strip()
                    if serial_number and serial_number not in processed_serials:
                        processed_serials.add(serial_number)
                        rows.append({
                            'Part Number': part_number,
                            'Part Name': part_name,
                            'Serial Number': serial_number,
                            'Quantity': '1',
                        })
                        serialized = True

        if not serialized and part_number not in added_unserialized_parts:
            rows.append({
                'Part Number': part_number,
                'Part Name': part_name,
                'Serial Number': 'N/A',
                'Quantity': quantity_str or '1',
            })
            added_unserialized_parts.add(part_number)

    if not rows:
        raise ValueError('No labels could be created from this CSV.')
    return pd.DataFrame(rows)

def _label_settings_dict(label_width_in, label_height_in, font_size, x_offset_in=0.0, y_offset_in=0.0, line_spacing_in=0.30, logo_scale=1.0):
    return {
        'label_width_in': float(label_width_in),
        'label_height_in': float(label_height_in),
        'font_size': int(font_size),
        'x_offset_in': float(x_offset_in),
        'y_offset_in': float(y_offset_in),
        'line_spacing_in': float(line_spacing_in),
        'logo_scale': float(logo_scale),
    }


def draw_label_on_canvas(c, label, label_width, label_height, settings, sales_order, customer_po):
    font_size = int(settings['font_size'])
    x_offset = settings['x_offset_in'] * inch
    y_offset = settings['y_offset_in'] * inch
    line_spacing = settings['line_spacing_in'] * inch
    logo_scale = settings['logo_scale']

    c.setFont('Helvetica-Bold', font_size)
    y = label_height - 0.3 * inch - y_offset
    x = 0.5 * inch + x_offset

    logo = get_logo_image()
    if logo is not None:
        try:
            img_width, img_height = logo.size
            max_width = 0.6 * inch * logo_scale
            max_height = 0.6 * inch * logo_scale
            ratio = min(max_width / img_width, max_height / img_height)
            new_width = img_width * ratio
            new_height = img_height * ratio
            c.drawInlineImage(
                logo,
                label_width - new_width - 0.2 * inch + x_offset,
                label_height - new_height - 0.2 * inch - y_offset,
                width=new_width,
                height=new_height,
            )
        except Exception:
            pass

    c.drawString(x, y, f'Sales Order: {sales_order}')
    y -= line_spacing
    c.drawString(x, y, f'Customer Purchase Order: {customer_po}')
    y -= line_spacing

    for key in ['Part Number', 'Part Name', 'Serial Number', 'Quantity']:
        c.drawString(x, y, f'{key}: {label.get(key, '')}')
        y -= line_spacing


def _load_preview_font(size):
    for candidate in [
        '/System/Library/Fonts/Supplemental/Arial.ttf',
        '/System/Library/Fonts/Supplemental/Helvetica.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    ]:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def build_label_preview_image(label, sales_order, customer_po, settings, scale=2):
    width_px = max(300, int(settings['label_width_in'] * 96 * scale))
    height_px = max(180, int(settings['label_height_in'] * 96 * scale))
    img = Image.new('RGB', (width_px, height_px), 'white')
    draw = ImageDraw.Draw(img)

    font = _load_preview_font(max(10, int(settings['font_size'] * scale * 1.35)))
    x = int((0.5 + settings['x_offset_in']) * 96 * scale)
    y = int((0.3 + settings['y_offset_in']) * 96 * scale)
    line_px = max(16, int(settings['line_spacing_in'] * 96 * scale))

    logo = get_logo_image()
    if logo is not None:
        try:
            logo = logo.copy().convert('RGBA')
            max_w = int(0.6 * settings['logo_scale'] * 96 * scale)
            max_h = int(0.6 * settings['logo_scale'] * 96 * scale)
            ratio = min(max_w / logo.width, max_h / logo.height)
            new_size = (max(1, int(logo.width * ratio)), max(1, int(logo.height * ratio)))
            logo = logo.resize(new_size)
            img.paste(logo, (width_px - new_size[0] - int(0.2 * 96 * scale) + int(settings['x_offset_in'] * 96 * scale), int(0.2 * 96 * scale) + int(settings['y_offset_in'] * 96 * scale)), logo)
        except Exception:
            pass

    lines = [
        f'Sales Order: {sales_order}',
        f'Customer Purchase Order: {customer_po}',
        f"Part Number: {label.get('Part Number', '')}",
        f"Part Name: {label.get('Part Name', '')}",
        f"Serial Number: {label.get('Serial Number', '')}",
        f"Quantity: {label.get('Quantity', '')}",
    ]
    for line in lines:
        draw.text((x, y), line, fill='black', font=font)
        y += line_px

    draw.rectangle([(0, 0), (width_px - 1, height_px - 1)], outline='black', width=max(1, scale))
    return img


def build_page_preview_image(labels_df, sales_order, customer_po, settings, max_labels=4, scale=1):
    previews = [build_label_preview_image(row, sales_order, customer_po, settings, scale=scale) for row in labels_df.head(max_labels).to_dict(orient='records')]
    if not previews:
        return None
    gap = 16
    width = max(img.width for img in previews)
    height = sum(img.height for img in previews) + gap * (len(previews) - 1)
    sheet = Image.new('RGB', (width, height), '#f4f4f4')
    y = 0
    for img in previews:
        sheet.paste(img, (0, y))
        y += img.height + gap
    return sheet


def build_label_pdf(labels_df, sales_order, customer_po, settings):
    output = io.BytesIO()
    label_width = settings['label_width_in'] * inch
    label_height = settings['label_height_in'] * inch
    c = canvas.Canvas(output, pagesize=(label_width, label_height))
    for row in labels_df.to_dict(orient='records'):
        draw_label_on_canvas(c, row, label_width, label_height, settings, sales_order, customer_po)
        c.showPage()
    c.save()
    output.seek(0)
    return output.getvalue()



def parse_jabil_labels(uploaded_file):
    uploaded_file.seek(0)
    rows = []
    processed_serials = set()
    part_header_info = {}
    parts_with_sn = set()
    text_io = io.StringIO(uploaded_file.getvalue().decode('utf-8-sig', errors='replace'))
    reader = csv.DictReader(text_io)
    for row in reader:
        part_number = (row.get('Item') or row.get('Part Number') or 'Unknown').strip()
        desc_raw = (row.get('Description') or row.get('Part Name') or 'Unknown').strip()
        quantity_str = (row.get('Shipped') or row.get('Quantity') or '1').strip()
        cust_pn_csv = (row.get('Customer Part Number') or row.get('Jabil Part Number') or row.get('Customer PN') or '').strip()
        base_cust_pn = JABIL_CUSTOMER_PN_MAP.get(part_number) or cust_pn_csv or part_number

        m_mn = re.search(r'\bMN\b\s*[:\-]?\s*([^|\s,;]+)', desc_raw, re.IGNORECASE)
        part_name = m_mn.group(1).strip() if m_mn else clean_part_name(desc_raw)
        if part_number not in part_header_info:
            part_header_info[part_number] = (part_name, quantity_str or '1', base_cust_pn)

        sn_blocks = re.findall(r'(?:\bS\/?N\b|\bSN\b)\s*[:\-]?\s*([^|\r\n]+)', desc_raw, flags=re.IGNORECASE)
        for block in sn_blocks:
            for serial_number in re.split(r'[,\s;|]+', block.strip()):
                serial_number = serial_number.strip()
                if serial_number and serial_number not in processed_serials:
                    processed_serials.add(serial_number)
                    rows.append({
                        'Part Number': part_number,
                        'Part Name': part_name,
                        'Serial Number': serial_number,
                        'Quantity': '1',
                        'Customer Part Number': base_cust_pn,
                    })
                    parts_with_sn.add(part_number)

        raw_sn = (row.get('Serial Number') or row.get('S/N') or row.get('SN') or '').strip()
        if raw_sn:
            for serial_number in re.split(r'[,\s;|]+', raw_sn):
                serial_number = serial_number.strip()
                if serial_number and serial_number not in processed_serials:
                    processed_serials.add(serial_number)
                    rows.append({
                        'Part Number': part_number,
                        'Part Name': part_name,
                        'Serial Number': serial_number,
                        'Quantity': '1',
                        'Customer Part Number': base_cust_pn,
                    })
                    parts_with_sn.add(part_number)

    for pn, (part_name, qty, base_cust_pn) in part_header_info.items():
        if pn in parts_with_sn:
            continue
        rows.append({
            'Part Number': pn,
            'Part Name': part_name,
            'Serial Number': 'N/A',
            'Quantity': qty,
            'Customer Part Number': base_cust_pn,
        })

    if not rows:
        raise ValueError('No Jabil labels could be created from this CSV.')
    return pd.DataFrame(rows)


def _clean_code39(value: str) -> str:
    if not value:
        return ''
    v = str(value).upper().strip()
    return re.sub(r"[^A-Z0-9\-\. \$\/\+\%]", "", v)


def _draw_text_and_barcode(c, label_text, text_value, x, y, bar_height=0.45 * inch, code_value=None):
    c.setFont('Helvetica-Bold', 9)
    text_value = text_value or ''
    code_str = _clean_code39(code_value if code_value is not None else text_value)
    c.drawString(x, y, f"{label_text} {text_value}".rstrip())
    if not code_str:
        return y - 0.3 * inch
    barcode = code39.Standard39(code_str, barHeight=bar_height, stop=1, checksum=0, barWidth=0.01 * inch)
    barcode_y = y - bar_height - 4
    barcode.drawOn(c, x, barcode_y)
    return barcode_y - 0.12 * inch


def draw_jabil_label_on_canvas(c, label, label_width, label_height, settings, sales_order, customer_po, date_code=''):
    logo = get_logo_image()
    margin_left = 0.35 * inch
    margin_right = 0.35 * inch
    margin_top = 0.35 * inch
    right_col_x = 3.2 * inch
    right_col_width = label_width - right_col_x - margin_right

    logo_max_w = right_col_width
    logo_max_h = 0.9 * inch
    logo_y = label_height - margin_top - logo_max_h
    if logo is not None:
        try:
            img_w, img_h = logo.size
            ratio = min(logo_max_w / img_w, logo_max_h / img_h) * max(0.2, settings.get('logo_scale',1.0))
            w = img_w * ratio
            h = img_h * ratio
            logo_x = right_col_x + (right_col_width - w) / 2.0 + 0.25 * inch
            c.drawInlineImage(logo, logo_x, logo_y, width=w, height=h)
        except Exception:
            pass

    mn_text = label.get('Part Name', '') or label.get('Part Number', '')
    box_w = right_col_width * 0.9
    box_h = 0.7 * inch
    box_x = right_col_x + (right_col_width - box_w) / 2.0
    box_y = logo_y - box_h - 0.15 * inch
    c.rect(box_x, box_y, box_w, box_h, stroke=1, fill=0)
    c.setFont('Helvetica-Bold', 14)
    c.drawCentredString(box_x + box_w / 2.0, box_y + box_h / 2.0 - 5, mn_text)

    right_data_width = 1.6 * inch
    right_data_x = label_width - margin_right - right_data_width
    y_right = box_y - 0.5 * inch
    y_right = _draw_text_and_barcode(c, '(D) DATE CODE:', date_code or '', right_data_x, y_right, bar_height=0.35 * inch, code_value=date_code or '')
    _draw_text_and_barcode(c, '(Q) QUANTITY:', str(label.get('Quantity', '1')).strip(), right_data_x, y_right, bar_height=0.35 * inch, code_value=str(label.get('Quantity', '1')).strip())

    left_x = margin_left
    y = label_height - margin_top - 0.4 * inch
    wibotic_pn = str(label.get('Part Number', ''))
    cust_pn_base = str(label.get('Customer Part Number', '')).strip() or wibotic_pn
    serial = str(label.get('Serial Number', ''))
    text_po = customer_po or ''

    y = _draw_text_and_barcode(c, '(P) CUSTOMER PART NO.:', cust_pn_base, left_x, y, bar_height=0.45 * inch, code_value=f'P{cust_pn_base}')
    y = _draw_text_and_barcode(c, '(K) CUSTOMER PO NO.:', text_po, left_x, y, bar_height=0.45 * inch, code_value=f'K{text_po}' if text_po else '')
    y = _draw_text_and_barcode(c, '(1P) WIBOTIC PART NO.:', wibotic_pn, left_x, y, bar_height=0.45 * inch, code_value=f'1P{wibotic_pn}')
    _draw_text_and_barcode(c, '(S) WIBOTIC SERIAL NO.:', serial, left_x, y, bar_height=0.45 * inch, code_value=f'S{serial}')


def build_jabil_preview_image(label, sales_order, customer_po, settings, date_code='', scale=2):
    width_px = max(500, int(settings['label_width_in'] * 96 * scale))
    height_px = max(300, int(settings['label_height_in'] * 96 * scale))
    img = Image.new('RGB', (width_px, height_px), 'white')
    draw = ImageDraw.Draw(img)
    title_font = _load_preview_font(max(14, int(11 * scale)))
    text_font = _load_preview_font(max(10, int(8 * scale)))

    # left content
    left_x = int(0.35 * 96 * scale)
    y = int(0.35 * 96 * scale)
    line_gap = int(0.68 * 96 * scale)
    fields = [
        ('(P) CUSTOMER PART NO.', label.get('Customer Part Number', '')),
        ('(K) CUSTOMER PO NO.', customer_po),
        ('(1P) WIBOTIC PART NO.', label.get('Part Number', '')),
        ('(S) WIBOTIC SERIAL NO.', label.get('Serial Number', '')),
    ]
    for title, value in fields:
        draw.text((left_x, y), f'{title}: {value}', fill='black', font=text_font)
        y += int(0.20 * 96 * scale)
        draw.rectangle((left_x, y, left_x + int(2.4 * 96 * scale), y + int(0.34 * 96 * scale)), outline='black', width=max(1, scale))
        draw.line((left_x + 6, y + int(0.17 * 96 * scale), left_x + int(2.4 * 96 * scale) - 6, y + int(0.17 * 96 * scale)), fill='black', width=max(1, scale))
        y += line_gap

    # right top logo and model box
    logo = get_logo_image()
    if logo is not None:
        try:
            logo = logo.copy().convert('RGBA')
            max_w = int(1.6 * 96 * scale)
            max_h = int(0.8 * 96 * scale)
            ratio = min(max_w / logo.width, max_h / logo.height)
            new_size = (max(1, int(logo.width * ratio)), max(1, int(logo.height * ratio)))
            logo = logo.resize(new_size)
            img.paste(logo, (width_px - new_size[0] - int(0.45 * 96 * scale), int(0.35 * 96 * scale)), logo)
        except Exception:
            pass

    box_w = int(1.9 * 96 * scale)
    box_h = int(0.6 * 96 * scale)
    box_x = width_px - box_w - int(0.45 * 96 * scale)
    box_y = int(1.25 * 96 * scale)
    draw.rectangle((box_x, box_y, box_x + box_w, box_y + box_h), outline='black', width=max(1, scale))
    mn_text = str(label.get('Part Name', '') or label.get('Part Number', ''))
    bbox = draw.textbbox((0,0), mn_text, font=title_font)
    tx = box_x + (box_w - (bbox[2]-bbox[0]))/2
    ty = box_y + (box_h - (bbox[3]-bbox[1]))/2 - 2
    draw.text((tx, ty), mn_text, fill='black', font=title_font)

    rd_x = width_px - int(1.95 * 96 * scale)
    rd_y = int(2.2 * 96 * scale)
    for title, value in [('(D) DATE CODE', date_code), ('(Q) QUANTITY', str(label.get('Quantity','1')) )]:
        draw.text((rd_x, rd_y), f'{title}: {value}', fill='black', font=text_font)
        rd_y += int(0.20 * 96 * scale)
        draw.rectangle((rd_x, rd_y, rd_x + int(1.4 * 96 * scale), rd_y + int(0.26 * 96 * scale)), outline='black', width=max(1, scale))
        draw.line((rd_x + 6, rd_y + int(0.13 * 96 * scale), rd_x + int(1.4 * 96 * scale) - 6, rd_y + int(0.13 * 96 * scale)), fill='black', width=max(1, scale))
        rd_y += int(0.58 * 96 * scale)

    draw.rectangle([(0, 0), (width_px - 1, height_px - 1)], outline='black', width=max(1, scale))
    return img


def build_jabil_pdf(labels_df, sales_order, customer_po, settings, date_code=''):
    output = io.BytesIO()
    label_width = settings['label_width_in'] * inch
    label_height = settings['label_height_in'] * inch
    c = canvas.Canvas(output, pagesize=(label_width, label_height))
    for row in labels_df.to_dict(orient='records'):
        draw_jabil_label_on_canvas(c, row, label_width, label_height, settings, sales_order, customer_po, date_code=date_code)
        c.showPage()
    c.save()
    output.seek(0)
    return output.getvalue()




# ---------- Label help / templates ----------


def build_nabtesco_csv_from_shipment_detail(sh_detail: Dict[str, Any]) -> bytes:
    lines = SOSReadonlyClient.extract_shipment_lines(sh_detail)
    out_rows: List[Dict[str, Any]] = []
    for ln in lines:
        item_obj = ln.get("item") or {}
        part_number = str(item_obj.get("name") or item_obj.get("fullname") or ln.get("itemName") or ln.get("name") or "").strip()
        description = str(ln.get("description") or "").strip()
        shipped = ln.get("quantity") or ln.get("qty") or ln.get("quantityShipped") or ln.get("shippedQuantity") or 1
        try:
            shipped_val = int(float(shipped))
        except Exception:
            shipped_val = 1
        out_rows.append({
            "Item": part_number,
            "Description": description,
            "Shipped": shipped_val,
        })
    if not out_rows:
        return b"Item,Description,Shipped\n"
    return pd.DataFrame(out_rows).to_csv(index=False).encode("utf-8")

try:
    NABTESCO_TEMPLATE_BYTES = Path("/mnt/data/CSV Example.csv").read_bytes()
except Exception:
    NABTESCO_TEMPLATE_BYTES = b"Item,Description,Shipped\n130-000058,TR-302-AC-ST-JP | SN:WX123456789,1\n"


def _build_jabil_template_bytes():
    try:
        df = pd.read_csv(io.BytesIO(NABTESCO_TEMPLATE_BYTES))
        if 'Customer Part Number' not in df.columns:
            df['Customer Part Number'] = ''
        return df.head(6).to_csv(index=False).encode('utf-8')
    except Exception:
        sample = pd.DataFrame([
            {
                'Item': '130-000058',
                'Description': 'TR-302-AC-ST-JP | SN:WX123456789',
                'Shipped': 1,
                'Customer Part Number': 'JABIL-EXAMPLE-001',
            }
        ])
        return sample.to_csv(index=False).encode('utf-8')


JABIL_TEMPLATE_BYTES = _build_jabil_template_bytes()


def _label_help_table(rows):
    return pd.DataFrame(rows, columns=['Field', 'Required', 'Example', 'How it is used'])


def _show_columns_found(uploaded_file, required_cols, optional_cols, key_prefix):
    if uploaded_file is None:
        return
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, nrows=5)
        uploaded_file.seek(0)
        found = list(df.columns)
        required_found = [c for c in required_cols if c in found]
        optional_found = [c for c in optional_cols if c in found]
        missing = [c for c in required_cols if c not in found]
        st.markdown('##### CSV check')
        c1, c2, c3 = st.columns(3)
        c1.metric('Columns found', len(found))
        c2.metric('Required matched', len(required_found))
        c3.metric('Missing required', len(missing))
        st.caption('Detected columns: ' + ', '.join(found))
        if optional_found:
            st.caption('Optional columns found: ' + ', '.join(optional_found))
        if missing:
            st.warning('Missing required columns for this pattern: ' + ', '.join(missing))
        else:
            st.success('Required columns look good for this pattern.')
    except Exception as e:
        st.warning(f'Could not inspect CSV columns: {e}')


def _render_nabtesco_help(label_csv=None):
    required = ['Item', 'Description', 'Shipped']
    optional = ['Part Number', 'Part Name', 'Quantity', 'Serial Number', 'S/N', 'SN']
    with st.expander('Help for Nabtesco', expanded=False):
        st.markdown('**Overview**')
        st.write('Upload the shipment CSV used for Nabtesco labels. The parser reads the shipment rows and creates one label per serial number when serials are present.')
        d1, d2 = st.columns([1, 1])
        d1.download_button('Download Nabtesco template CSV', data=NABTESCO_TEMPLATE_BYTES, file_name='nabtesco_template.csv', mime='text/csv', use_container_width=True, key='nab_help_dl')
        d2.caption('Best results come from the same CSV style you already use for shipment exports.')
        st.markdown('**Required CSV fields**')
        st.dataframe(_label_help_table([
            ['Item', 'Yes', '130-000058', 'Part number printed on the label'],
            ['Description', 'Yes', 'TR-302-AC-ST-JP | SN:WX683F7D3C', 'Used to find part name and any serial numbers embedded in the text'],
            ['Shipped', 'Yes', '10', 'Used as quantity when a row is not serialized'],
            ['Serial Number / S/N / SN', 'Optional', 'WX683F7D3C', 'Used if serials are not embedded inside Description'],
        ]), use_container_width=True, hide_index=True)
        st.markdown('**Example CSV**')
        try:
            example_df = pd.read_csv(io.BytesIO(NABTESCO_TEMPLATE_BYTES)).head(5)
            st.dataframe(example_df, use_container_width=True, height=180)
        except Exception:
            st.caption('Template preview unavailable.')
        st.markdown('**Common issues**')
        st.write('• Missing `Description` or `Item` columns.  • Serial numbers not present in either Description or Serial Number fields.  • Duplicate serials collapse into one label.')
        _show_columns_found(label_csv, required, optional, 'nab_help')


def _render_jabil_help(label_csv=None):
    required = ['Item', 'Description', 'Shipped']
    optional = ['Customer Part Number', 'Jabil Part Number', 'Customer PN', 'Serial Number', 'S/N', 'SN']
    with st.expander('Help for Jabil', expanded=False):
        st.markdown('**Overview**')
        st.write('Upload the shipment CSV for Jabil labels. The app can use a customer part number column when present, or fall back to the built-in part-number map.')
        d1, d2 = st.columns([1, 1])
        d1.download_button('Download Jabil template CSV', data=JABIL_TEMPLATE_BYTES, file_name='jabil_template.csv', mime='text/csv', use_container_width=True, key='jabil_help_dl')
        d2.caption('Sales Order, Customer Purchase Order, and Date Code are entered in the app, not in the CSV.')
        st.markdown('**Required CSV fields**')
        st.dataframe(_label_help_table([
            ['Item', 'Yes', '130-000058', 'Internal part number used for the label and customer-PN mapping'],
            ['Description', 'Yes', 'TR-302-AC-ST-JP | SN:WX683F7D3C', 'Used to find part name and serial numbers'],
            ['Shipped', 'Yes', '10', 'Used as quantity when a row is not serialized'],
            ['Customer Part Number / Jabil Part Number / Customer PN', 'Optional', 'JBL-12345', 'Overrides the built-in customer part number map for the printed label'],
        ]), use_container_width=True, hide_index=True)
        st.markdown('**Example CSV**')
        try:
            example_df = pd.read_csv(io.BytesIO(JABIL_TEMPLATE_BYTES)).head(5)
            st.dataframe(example_df, use_container_width=True, height=180)
        except Exception:
            st.caption('Template preview unavailable.')
        st.markdown('**Common issues**')
        st.write('• Missing `Item` or `Description`.  • No serial number and no shipped quantity.  • Customer part number mismatch when you expected an override column.')
        _show_columns_found(label_csv, required, optional, 'jabil_help')



def render_nabtesco_editor():
    st.markdown('### Nabtesco')
    editor_col, preview_col = st.columns([0.95, 1.35], gap='large')
    with editor_col:
        _render_nabtesco_help(st.session_state.get('nabtesco_csv'))
        st.markdown('#### Edit label')
        label_csv = st.file_uploader('Shipment CSV', type=['csv'], key='nabtesco_csv')
        sales_order = st.text_input('Sales Order', key='nabtesco_sales_order')
        fetch_col, clear_col = st.columns([1.2, 0.8])
        fetch_clicked = fetch_col.button('Fetch Shipment from SOS', use_container_width=True, key='nab_fetch_shipment')
        if clear_col.button('Clear fetched shipment', use_container_width=True, key='nab_clear_fetched'):
            for k in ['nab_fetched_csv_bytes','nab_shipment_options','nab_selected_shipment_number','nab_fetch_status']:
                st.session_state.pop(k, None)
            st.rerun()

        customer_po = st.text_input('Customer Purchase Order', key='nabtesco_customer_po')
        preset = st.radio('Label size preset', ['4.00 × 2.32', '6.00 × 2.32', 'Custom'], horizontal=True, key='nab_size_preset', label_visibility='collapsed')
        preset_map = {'4.00 × 2.32': (4.0, 2.32), '6.00 × 2.32': (6.0, 2.32)}

        def labeled_number(label, key, value, min_value=None, step=None, fmt=None):
            lcol, wcol = st.columns([0.92, 1.18], gap='small')
            with lcol:
                st.markdown(f"<div style='padding-top:0.45rem;font-weight:600;color:#445066'>{label}</div>", unsafe_allow_html=True)
            with wcol:
                if key not in st.session_state:
                    st.session_state[key] = value
                kwargs = dict(label=label, key=key, label_visibility='collapsed')
                if min_value is not None: kwargs['min_value']=min_value
                if step is not None: kwargs['step']=step
                if fmt is not None: kwargs['format']=fmt
                return st.number_input(**kwargs)

        custom_size = preset == 'Custom'
        if custom_size:
            label_width = labeled_number('Label width', 'nab_custom_width', st.session_state.get('nab_custom_width', 4.0), min_value=1.0, step=0.1, fmt='%.2f')
            label_height = labeled_number('Label height', 'nab_custom_height', st.session_state.get('nab_custom_height', 2.32), min_value=1.0, step=0.01, fmt='%.2f')
        else:
            label_width, label_height = preset_map[preset]
            st.markdown(f"<div style='font-weight:700;color:#445066;margin-top:0.15rem;margin-bottom:0.35rem;'>Label size</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='padding:0.72rem 0.9rem;border-radius:14px;background:rgba(255,255,255,0.84);border:1px solid rgba(23,168,255,0.16);font-weight:700;color:#0b2942;margin-bottom:0.75rem;'>{label_width:.2f} in × {label_height:.2f} in</div>", unsafe_allow_html=True)
        st.markdown('##### Layout tuning')
        font_size = labeled_number('Font size', 'nab_font_size', 10, min_value=6, step=1)
        x_offset = labeled_number('Text X offset', 'nab_x_offset', 0.0, step=0.02, fmt='%.2f')
        y_offset = labeled_number('Text Y offset', 'nab_y_offset', 0.0, step=0.02, fmt='%.2f')
        line_spacing = labeled_number('Line spacing', 'nab_line_spacing', 0.30, min_value=0.15, step=0.01, fmt='%.2f')
        logo_scale = labeled_number('Logo scale', 'nab_logo_scale', 1.0, min_value=0.2, step=0.1, fmt='%.1f')
        preview_count = int(labeled_number('Page preview count', 'nab_preview_count_num', 3, min_value=1, step=1))

        if fetch_clicked:
            if not sales_order.strip():
                st.warning('Enter Sales Order first.')
            else:
                client, _status = sos_get_authenticated_client()
                if client is None:
                    st.error('SOS is not connected. Add SOS secrets first.')
                else:
                    progress = st.progress(0, text='Searching shipment(s) in SOS...')
                    try:
                        shipments = client.get_shipments_for_sales_order(sales_order.strip(), maxresults=50)
                        progress.progress(35, text='Loading shipment detail(s)...')
                        options = []
                        details_map = {}
                        for idx, sh in enumerate(shipments):
                            sh_id = sh.get('id')
                            if not sh_id:
                                continue
                            detail = client.get_shipment_detail(int(sh_id))
                            num = str(detail.get('number') or sh.get('number') or f'Shipment {idx+1}')
                            date = str(detail.get('date') or sh.get('date') or '')
                            label = f"{num} | {date}"
                            options.append(label)
                            details_map[label] = detail
                        progress.progress(70, text='Converting shipment to label rows...')
                        st.session_state['nab_shipment_options'] = options
                        st.session_state['nab_shipment_detail_map'] = details_map
                        if options:
                            st.session_state['nab_selected_shipment_label'] = options[0]
                            detail = details_map[options[0]]
                            st.session_state['nab_fetched_csv_bytes'] = build_nabtesco_csv_from_shipment_detail(detail)
                            st.session_state['nab_selected_shipment_number'] = str(detail.get('number') or '')
                            st.session_state['nab_fetch_status'] = f"Fetched {len(options)} shipment(s) from SOS."
                        else:
                            st.session_state['nab_fetched_csv_bytes'] = None
                            st.session_state['nab_fetch_status'] = 'No shipment found for that Sales Order.'
                        progress.progress(100, text='Done.')
                    except Exception as e:
                        st.session_state['nab_fetch_status'] = f'Fetch failed: {e}'
                    st.rerun()

        if st.session_state.get('nab_fetch_status'):
            st.info(st.session_state['nab_fetch_status'])

        options = st.session_state.get('nab_shipment_options') or []
        if options:
            selected_label = st.selectbox('Fetched shipment', options, key='nab_selected_shipment_label')
            details_map = st.session_state.get('nab_shipment_detail_map') or {}
            selected_detail = details_map.get(selected_label)
            if selected_detail is not None:
                st.session_state['nab_fetched_csv_bytes'] = build_nabtesco_csv_from_shipment_detail(selected_detail)
                st.session_state['nab_selected_shipment_number'] = str(selected_detail.get('number') or '')

    active_csv = label_csv
    fetched_csv_bytes = st.session_state.get('nab_fetched_csv_bytes')
    if active_csv is None and fetched_csv_bytes:
        active_csv = io.BytesIO(fetched_csv_bytes)
        active_csv.name = 'shipment_from_sos.csv'

    if active_csv is None:
        with preview_col:
            st.info('Upload a CSV or fetch a shipment from SOS to open the label preview.')
        return
    try:
        labels_df = parse_nabtesco_labels(active_csv)
        settings = _label_settings_dict(label_width, label_height, font_size, x_offset, y_offset, line_spacing, logo_scale)
        so_for_label = (st.session_state.get('nab_selected_shipment_number') or sales_order or '').strip()
        with editor_col:
            st.success(f'{len(labels_df)} labels created.')
        with preview_col:
            export_col, mode_col = st.columns([1.0, 1.2], gap='small')
            with export_col:
                st.markdown('#### Export')
                if so_for_label and customer_po.strip():
                    pdf_bytes = build_label_pdf(labels_df, so_for_label, customer_po.strip(), settings)
                    st.download_button('Download label PDF', pdf_bytes, file_name='nabtesco_labels.pdf', mime='application/pdf', use_container_width=True)
                else:
                    missing = []
                    if not so_for_label:
                        missing.append('Sales Order/Shipment')
                    if not customer_po.strip():
                        missing.append('Customer Purchase Order')
                    st.warning('Missing: ' + ', '.join(missing))
            with mode_col:
                st.markdown('#### Preview')
                preview_choice = st.radio('Preview mode', ['Single label', 'Page view'], horizontal=True, key='nab_preview_mode')
            first_label = labels_df.iloc[0].to_dict()
            if preview_choice == 'Single label':
                st.image(build_label_preview_image(first_label, so_for_label, customer_po.strip(), settings, scale=2), use_container_width=True)
            else:
                st.image(build_page_preview_image(labels_df, so_for_label, customer_po.strip(), settings, max_labels=preview_count, scale=1), use_container_width=True)
            with st.expander('Label data', expanded=False):
                st.dataframe(labels_df, use_container_width=True, height=240)
    except Exception as e:
        with preview_col:
            st.error(f'Failed to build labels: {e}')

def render_jabil_editor():
    st.markdown('### Jabil')
    editor_col, preview_col = st.columns([0.95, 1.35], gap='large')
    with editor_col:
        _render_jabil_help(st.session_state.get('jabil_csv'))
        st.markdown('#### Edit label')
        label_csv = st.file_uploader('Shipment CSV', type=['csv'], key='jabil_csv')
        sales_order = st.text_input('Sales Order', key='jabil_sales_order')
        customer_po = st.text_input('Customer Purchase Order', key='jabil_customer_po')
        date_code = st.text_input('Date Code', key='jabil_date_code')
        st.markdown(f"<div style='font-weight:700;color:#445066;margin-top:0.15rem;margin-bottom:0.35rem;'>Label size</div>", unsafe_allow_html=True)
        st.markdown("<div style='padding:0.72rem 0.9rem;border-radius:14px;background:rgba(255,255,255,0.84);border:1px solid rgba(23,168,255,0.16);font-weight:700;color:#0b2942;margin-bottom:0.75rem;'>6.00 in × 4.00 in</div>", unsafe_allow_html=True)

        def jn(label, key, value, min_value=None, step=None, fmt=None):
            lcol, wcol = st.columns([0.92, 1.18], gap='small')
            with lcol:
                st.markdown(f"<div style='padding-top:0.45rem;font-weight:600;color:#445066'>{label}</div>", unsafe_allow_html=True)
            with wcol:
                if key not in st.session_state:
                    st.session_state[key]=value
                kwargs=dict(label=label, key=key, label_visibility='collapsed')
                if min_value is not None: kwargs['min_value']=min_value
                if step is not None: kwargs['step']=step
                if fmt is not None: kwargs['format']=fmt
                return st.number_input(**kwargs)

        st.markdown('##### Layout tuning')
        logo_scale = jn('Logo scale', 'jabil_logo_scale', 1.0, min_value=0.2, step=0.1, fmt='%.1f')
        preview_count = int(jn('Page preview count', 'jabil_preview_count_num', 3, min_value=1, step=1))
    if label_csv is None:
        with preview_col:
            st.info('Upload a CSV to open the Jabil label preview.')
        return
    try:
        labels_df = parse_jabil_labels(label_csv)
        settings = _label_settings_dict(6.0, 4.0, 10, 0.0, 0.0, 0.30, logo_scale)
        with editor_col:
            st.success(f'{len(labels_df)} labels created from the CSV.')
            with st.expander('Label data', expanded=False):
                st.dataframe(labels_df, use_container_width=True, height=240)
            if sales_order.strip() and customer_po.strip():
                pdf_bytes = build_jabil_pdf(labels_df, sales_order.strip(), customer_po.strip(), settings, date_code=date_code.strip())
                st.download_button('Download Jabil label PDF', pdf_bytes, file_name='jabil_labels.pdf', mime='application/pdf', use_container_width=True)
            else:
                st.warning('Enter Sales Order and Customer Purchase Order to enable the final PDF export.')
        with preview_col:
            st.markdown('#### Preview')
            preview_choice = st.radio('Preview mode', ['Single label', 'Page view'], horizontal=True, key='jabil_preview_mode')
            first_label = labels_df.iloc[0].to_dict()
            if preview_choice == 'Single label':
                st.image(build_jabil_preview_image(first_label, sales_order.strip(), customer_po.strip(), settings, date_code=date_code.strip(), scale=2), use_container_width=True)
            else:
                previews = [build_jabil_preview_image(row, sales_order.strip(), customer_po.strip(), settings, date_code=date_code.strip(), scale=1) for row in labels_df.head(preview_count).to_dict(orient='records')]
                width = max(img.width for img in previews)
                gap = 16
                height = sum(img.height for img in previews) + gap * (len(previews) - 1)
                sheet = Image.new('RGB', (width, height), '#f4f4f4')
                y = 0
                for img in previews:
                    sheet.paste(img, (0, y))
                    y += img.height + gap
                st.image(sheet, use_container_width=True)
    except Exception as e:
        with preview_col:
            st.error(f'Failed to build Jabil labels: {e}')




# -----------------------------
# SOS read-only workspace
# -----------------------------
SOS_BASE_URL = "https://api.sosinventory.com/api/v2/"
SOS_AUTH_URL = "https://api.sosinventory.com/oauth2/authorize"
SOS_TOKEN_URL = "https://api.sosinventory.com/oauth2/token"
SOS_SAFE_FILENAME = re.compile(r"[^a-zA-Z0-9_-]+")
SOS_LOCATION_RE = re.compile(r"(Aisle\s*[^,|]+?(?:,\s*Shelf\s*[^,|]+)?)", re.IGNORECASE)


@dataclass
class SOSLineItem:
    item_id: int
    on_hand: int
    type: str
    quantity: float
    fullname: str
    description: str
    has_serial: bool
    serial: Optional[str] = None
    notes: str = ""
    purchase_cost: float = 0.0


class SOSAuthError(Exception):
    pass


def sos_safe_default_filename(title: str) -> str:
    safe = SOS_SAFE_FILENAME.sub("_", (title or "").strip()).strip("_")
    safe = safe[:60] if safe else "report"
    return safe + ".csv"


def sos_extract_location(notes: str) -> str:
    if not notes:
        return ""
    m = SOS_LOCATION_RE.search(notes)
    return m.group(1).strip() if m else ""


def sos_trim_text(s: str, n: int = 45) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def sos_to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def sos_buildability_summary(parent_item: SOSLineItem, rows: List[List[str]]) -> Dict[str, Any]:
    assembly_on_hand = int(getattr(parent_item, "on_hand", 0) or 0)
    assembly_location = sos_extract_location(getattr(parent_item, "notes", "") or "")
    assembly_notes = (getattr(parent_item, "notes", "") or "").strip()
    assembly_location_display = assembly_location or (assembly_notes if assembly_notes else "Not found in notes")
    if not rows:
        return {
            "assembly_on_hand": assembly_on_hand,
            "assembly_location": assembly_location_display,
            "assembly_purchase_cost": sos_to_float(getattr(parent_item, "purchase_cost", 0), 0.0),
            "buildable_from_parts": 0,
            "potential_total": assembly_on_hand,
            "limiting_parts": "",
        }

    buildable_values = []
    for row in rows:
        try:
            part_number = str(row[0])
            needed = max(int(float(row[2])), 0)
            on_hand = max(int(float(row[3])), 0)
        except Exception:
            continue
        if needed <= 0:
            continue
        buildable_values.append((part_number, on_hand // needed))

    if not buildable_values:
        return {
            "assembly_on_hand": assembly_on_hand,
            "assembly_location": assembly_location_display,
            "assembly_purchase_cost": sos_to_float(getattr(parent_item, "purchase_cost", 0), 0.0),
            "buildable_from_parts": 0,
            "potential_total": assembly_on_hand,
            "limiting_parts": "",
        }

    min_buildable = min(v for _, v in buildable_values)
    limiting_parts = [pn for pn, v in buildable_values if v == min_buildable]
    return {
        "assembly_on_hand": assembly_on_hand,
        "assembly_location": assembly_location_display,
        "assembly_purchase_cost": sos_to_float(getattr(parent_item, "purchase_cost", 0), 0.0),
        "buildable_from_parts": int(min_buildable),
        "potential_total": int(assembly_on_hand + min_buildable),
        "limiting_parts": ", ".join(limiting_parts),
    }


def sos_normalize_item_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).lower()


def sos_item_choice_label(item: SOSLineItem) -> str:
    desc = sos_trim_text(item.description or '', 55)
    return f"{item.fullname} | {item.type} | On hand: {int(item.on_hand)} | {desc}"


def sos_serialize_item(item: SOSLineItem) -> Dict[str, Any]:
    return {
        'item_id': int(item.item_id),
        'on_hand': int(item.on_hand),
        'type': item.type,
        'quantity': float(item.quantity),
        'fullname': item.fullname,
        'description': item.description,
        'has_serial': bool(item.has_serial),
        'serial': item.serial,
        'notes': item.notes,
    }


def sos_deserialize_item(data: Dict[str, Any]) -> SOSLineItem:
    return SOSLineItem(
        item_id=int(data['item_id']),
        on_hand=int(data.get('on_hand', 0)),
        type=str(data.get('type', '')),
        quantity=float(data.get('quantity', 1)),
        fullname=str(data.get('fullname', '')),
        description=str(data.get('description', '')),
        has_serial=bool(data.get('has_serial', False)),
        serial=data.get('serial'),
        notes=str(data.get('notes', '')),
    )


def sos_pick_best_item(items: List[SOSLineItem], query_name: str) -> Optional[SOSLineItem]:
    if not items:
        return None

    q = sos_normalize_item_text(query_name)

    def score(item: SOSLineItem) -> tuple:
        full = sos_normalize_item_text(item.fullname)
        desc = sos_normalize_item_text(item.description)
        first_token = full.split(' ')[0] if full else ''
        labor_penalty = 1 if 'labor' in full or 'labor' in desc else 0
        rank = 0
        if full == q:
            rank = 100
        elif first_token == q:
            rank = 90
        elif full.startswith(q + ' '):
            rank = 80
        elif full.startswith(q):
            rank = 70
        elif q in full:
            rank = 60
        elif desc == q:
            rank = 50
        elif q in desc:
            rank = 40
        return (rank, -labor_penalty, -len(full))

    ranked = sorted(items, key=score, reverse=True)
    return ranked[0]


def sos_load_requests_from_csv(uploaded_file) -> List[Tuple[str, int]]:
    uploaded_file.seek(0)
    wrapper = io.TextIOWrapper(uploaded_file, encoding="utf-8", newline="")
    reader = csv.DictReader(wrapper)
    header_map = {h.lower().strip(): h for h in (reader.fieldnames or [])}

    def find_header(*candidates: str) -> Optional[str]:
        for c in candidates:
            key = c.lower().strip()
            if key in header_map:
                return header_map[key]
        return None

    name_col = find_header("Item Name", "ItemName", "Part Name", "Name", "Item Number", "Part Number", "PN")
    qty_col = find_header("quantity", "qty", "QTY", "Quantity")
    if not name_col or not qty_col:
        raise ValueError(f"CSV headers missing. Found: {list(header_map.keys())}")

    rows: List[Tuple[str, int]] = []
    for row in reader:
        name = (row.get(name_col) or "").strip()
        if not name:
            continue
        try:
            qty = int(row.get(qty_col) or 1)
        except Exception:
            qty = 1
        rows.append((name, qty))

    uploaded_file.seek(0)
    return rows


def sos_get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        value = st.secrets[name]
        return str(value) if value is not None else default
    except Exception:
        return os.environ.get(name, default)


def sos_build_auth_url() -> Optional[str]:
    client_id = sos_get_secret("SOS_CLIENT_ID")
    redirect_uri = sos_get_secret("SOS_REDIRECT_URI")
    if not client_id or not redirect_uri:
        return None
    params = {"response_type": "code", "client_id": client_id, "redirect_uri": redirect_uri}
    return f"{SOS_AUTH_URL}?{urlencode(params)}"


def sos_exchange_code_for_token(code: str) -> dict:
    client_id = sos_get_secret("SOS_CLIENT_ID")
    client_secret = sos_get_secret("SOS_CLIENT_SECRET")
    redirect_uri = sos_get_secret("SOS_REDIRECT_URI")
    if not client_id or not client_secret or not redirect_uri:
        raise SOSAuthError("Missing SOS_CLIENT_ID, SOS_CLIENT_SECRET, or SOS_REDIRECT_URI in secrets.")

    data = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": redirect_uri,
    }
    resp = requests.post(
        SOS_TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Host": "api.sosinventory.com"},
        data=data,
        timeout=30,
    )
    payload = resp.json()
    if not resp.ok or "access_token" not in payload:
        raise SOSAuthError(f"Token exchange failed: {payload}")
    return payload


def sos_refresh_token(refresh_token_value: str) -> dict:
    data = {"grant_type": "refresh_token", "refresh_token": refresh_token_value}
    resp = requests.post(
        SOS_TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Host": "api.sosinventory.com"},
        data=data,
        timeout=30,
    )
    payload = resp.json()
    if not resp.ok or "access_token" not in payload:
        raise SOSAuthError(f"Refresh failed: {payload}")
    return payload


class SOSReadonlyClient:
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
            resp = requests.get(SOS_BASE_URL + endpoint, headers=self.headers, params=params, timeout=50)
            try:
                last_json = resp.json()
            except Exception:
                last_json = {"status": "error", "message": resp.text or "", "code": resp.status_code}

            if "throttle" in str(last_json.get("message", "")).lower():
                continue
            if resp.ok:
                return last_json
            if resp.status_code == 400:
                if "throttle" in str(last_json.get("message", "")).lower():
                    continue
                if last_json.get("status", "") == "invalid":
                    return last_json
            if resp.status_code == 404:
                raise ValueError(404)
            raise RuntimeError(f"SOS GET failed ({resp.status_code}): {last_json.get('message', resp.text)}")
        return last_json

    def get_sales_orders(self, query: str, maxresults: int = 20) -> List[Dict[str, Any]]:
        resp = self._get("salesorder", params={"query": query, "maxresults": maxresults})
        return resp.get("data", []) or []

    def get_sales_order_by_number(self, so_number: str) -> Optional[Dict[str, Any]]:
        matches = self.get_sales_orders(so_number, maxresults=50)
        if not matches:
            return None
        so_number_l = so_number.strip().lower()
        for so in matches:
            num = (so.get("number") or "").strip().lower()
            if num == so_number_l:
                return so
        return matches[0]

    def get_sales_order_detail(self, so_id: int) -> Dict[str, Any]:
        resp = self._get(f"salesorder/{int(so_id)}")
        return resp.get("data", {}) or {}

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
            name = item_obj.get("name") or item_obj.get("fullname") or ln.get("itemName") or ln.get("name") or ""
            name = str(name).strip()
            if not name:
                name = str(ln.get("description") or "").strip()
            if name:
                reqs.append((name, qty_i))
        return reqs


    def get_shipments(self, query: str, maxresults: int = 50) -> List[Dict[str, Any]]:
        resp = self._get("shipment", params={"query": query, "maxresults": maxresults})
        return resp.get("data", []) or []

    def get_shipment_detail(self, shipment_id: int) -> Dict[str, Any]:
        resp = self._get(f"shipment/{int(shipment_id)}")
        return resp.get("data", {}) or {}

    def get_shipments_for_sales_order(self, so_number: str, maxresults: int = 50) -> List[Dict[str, Any]]:
        matches = self.get_shipments(so_number, maxresults=maxresults)
        so_lower = (so_number or "").strip().lower()
        filtered: List[Dict[str, Any]] = []
        for sh in matches:
            num = str(sh.get("number") or "").strip().lower()
            refs = " ".join([
                str((sh.get("salesOrder") or {}).get("number") or ""),
                str(sh.get("referenceNumber") or ""),
                str(sh.get("memo") or ""),
                str(sh.get("description") or ""),
            ]).strip().lower()
            if so_lower in num or so_lower in refs:
                filtered.append(sh)
        return filtered if filtered else matches

    @staticmethod
    def extract_shipment_lines(sh_detail: Dict[str, Any]) -> List[Dict[str, Any]]:
        for key in ("lines", "lineItems", "items", "shipmentLines"):
            if isinstance(sh_detail.get(key), list):
                return sh_detail[key]
        return []

    def get_items_by_id(self, ids: List[int]) -> List[SOSLineItem]:
        if not ids:
            return []
        as_str = ",".join(str(i) for i in ids)
        items = self._get("item", params={"ids": as_str}).get("data", [])
        return [SOSLineItem(item_id=item["id"], on_hand=item["onhand"], type=item["type"], quantity=1, fullname=item["fullname"], description=item.get("description", ""), has_serial=item.get("serialTracking", False), notes=item.get("notes", ""), purchase_cost=sos_to_float(item.get("purchaseCost", item.get("cost", 0)), 0.0)) for item in items]

    def get_items_by_name(self, name: str) -> List[SOSLineItem]:
        items = self._get("item", params={"query": name}).get("data", [])
        return [SOSLineItem(item_id=item["id"], on_hand=item["onhand"], type=item["type"], quantity=1, fullname=item["fullname"], description=item.get("description", ""), has_serial=item.get("serialTracking", False), notes=item.get("notes", ""), purchase_cost=sos_to_float(item.get("purchaseCost", item.get("cost", 0)), 0.0)) for item in items]

    def get_single_level_bom(self, item_id: int, assembly_quantity: int) -> List[SOSLineItem]:
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
            bom_item.quantity = item_build_data["quantity"] * assembly_quantity
        return bom_data

    def bom_lookup(self, item: SOSLineItem, quantity: int = 1, always_explode: bool = True) -> List[SOSLineItem]:
        line_items: List[SOSLineItem] = []
        assemblies = [{"item_id": item.item_id, "quantity": quantity}]

        def has_subassembly(e: SOSLineItem) -> bool:
            return (e.type or "").lower() in ["assembly", "item group"]

        while assemblies:
            assembly = assemblies.pop()
            bom_data = self.get_single_level_bom(int(assembly["item_id"]), int(assembly["quantity"]))
            if not bom_data:
                item_self = self.get_items_by_id([int(assembly["item_id"])])[0]
                item_self.quantity = float(assembly["quantity"])
                existing = next((x for x in line_items if x.fullname == item_self.fullname), None)
                if existing:
                    existing.quantity += item_self.quantity
                else:
                    line_items.append(item_self)
                continue
            for bom_item in bom_data:
                if has_subassembly(bom_item) and always_explode:
                    assemblies.append({"item_id": bom_item.item_id, "quantity": bom_item.quantity})
                    continue
                existing = next((x for x in line_items if x.fullname == bom_item.fullname), None)
                if existing:
                    existing.quantity += bom_item.quantity
                else:
                    line_items.append(bom_item)
        return line_items


def sos_get_authenticated_client() -> Tuple[Optional[SOSReadonlyClient], str]:
    if "sos_access_token" in st.session_state:
        return SOSReadonlyClient(st.session_state["sos_access_token"]), "Connected (session token)"

    refresh_value = sos_get_secret("SOS_REFRESH_TOKEN")
    if refresh_value:
        payload = sos_refresh_token(refresh_value)
        st.session_state["sos_access_token"] = payload["access_token"]
        if payload.get("refresh_token"):
            st.session_state["sos_refresh_token_latest"] = payload["refresh_token"]
        return SOSReadonlyClient(payload["access_token"]), "Connected (refreshed from secrets)"

    access_value = sos_get_secret("SOS_ACCESS_TOKEN")
    if access_value:
        st.session_state["sos_access_token"] = access_value
        return SOSReadonlyClient(access_value), "Connected (access token from secrets)"

    qp = st.query_params
    code = qp.get("code")
    if code:
        payload = sos_exchange_code_for_token(code)
        st.session_state["sos_access_token"] = payload["access_token"]
        if payload.get("refresh_token"):
            st.session_state["sos_refresh_token_latest"] = payload["refresh_token"]
        try:
            del st.query_params["code"]
        except Exception:
            pass
        return SOSReadonlyClient(payload["access_token"]), "Connected (OAuth code exchange)"

    return None, "Not connected"


def sos_bom_rows_from_selected_item(client: SOSReadonlyClient, item: SOSLineItem, qty: int, explode: bool = True) -> List[List[str]]:
    if item is None:
        return []
    bom = client.bom_lookup(item, quantity=qty, always_explode=True) if explode else [item]
    rows: List[List[str]] = []
    for bi in bom:
        needed = int(bi.quantity)
        onhand = int(bi.on_hand)
        short = max(needed - onhand, 0)
        enough = "✅" if short == 0 else "❌"
        buildable_qty = onhand // needed if needed > 0 else 0
        rows.append([bi.fullname, enough, str(needed), str(onhand), str(buildable_qty), str(short), bi.type, bi.description, sos_extract_location(bi.notes or ""), sos_trim_text(bi.notes or "", 45)])
    return rows


def sos_bom_rows_from_item(client: SOSReadonlyClient, name: str, qty: int, explode: bool = True) -> List[List[str]]:
    items = client.get_items_by_name(name)
    if not items:
        return []
    item = sos_pick_best_item(items, name)
    if item is None:
        return []
    return sos_bom_rows_from_selected_item(client, item, qty, explode=explode)


def sos_rows_to_dataframe(rows: List[List[str]], headers: List[str], source_label: Optional[str] = None) -> pd.DataFrame:
    data_rows = []
    for row in rows:
        data_rows.append(row if source_label is None else [source_label] + row)
    if source_label is None:
        return pd.DataFrame(data_rows, columns=headers)
    return pd.DataFrame(data_rows, columns=["Assembly / SO Line"] + headers)


def sos_grouped_sales_order_dataframe(client: SOSReadonlyClient, so_number: str, explode: bool) -> pd.DataFrame:
    so_obj = client.get_sales_order_by_number(so_number)
    if not so_obj:
        raise ValueError(f"Sales order not found: {so_number}")
    so_id = so_obj.get("id")
    if not so_id:
        raise ValueError(f"Sales order missing id: {so_obj}")
    so_detail = client.get_sales_order_detail(int(so_id))
    reqs = client.sales_order_to_requests(so_detail)
    headers = ["Part Number", "Enough", "Needed", "On Hand", "Buildable Qty", "Short", "Type", "Name/Description", "Location", "Notes"]
    frames: List[pd.DataFrame] = []
    for name, qty in reqs:
        rows = sos_bom_rows_from_item(client, name, qty, explode=explode)
        if rows:
            frames.append(sos_rows_to_dataframe(rows, headers, source_label=f"{name} x{qty}"))
    if not frames:
        return pd.DataFrame(columns=["Assembly / SO Line"] + headers)
    return pd.concat(frames, ignore_index=True)


def render_sos_dashboard_viewer():
    st.subheader('Dashboard / CSV viewer')
    df = st.session_state.get('sos_last_df')
    label = st.session_state.get('sos_last_label', 'Current result')
    if df is None or getattr(df, 'empty', True):
        st.info('Run a Single, Batch CSV, or Sales Order check first. The result will appear here for filtering, summaries, and export.')
        return

    st.caption(f'Viewing: {label}')
    work = df.copy()

    fc1, fc2, fc3, fc4 = st.columns([1.2, 1.2, 1.4, 1.0])
    with fc1:
        short_only = st.checkbox('Shortages only', value=False, key='sos_dash_short_only')
    with fc2:
        search_text = st.text_input('Search PN / description', key='sos_dash_search')
    with fc3:
        locations = ['(all)'] + sorted([str(x) for x in work.get('Location', pd.Series(dtype=str)).fillna('').replace('', '(blank)').unique().tolist()]) if 'Location' in work.columns else ['(all)']
        selected_location = st.selectbox('Location', locations, key='sos_dash_location')
    with fc4:
        source_options = ['(all)'] + sorted([str(x) for x in work.get('Assembly / SO Line', pd.Series(dtype=str)).fillna('').unique().tolist()]) if 'Assembly / SO Line' in work.columns else ['(all)']
        selected_source = st.selectbox('Source', source_options, key='sos_dash_source')

    if short_only and 'Short' in work.columns:
        short_num = pd.to_numeric(work['Short'], errors='coerce').fillna(0)
        work = work[short_num > 0].copy()
    if search_text:
        mask = pd.Series(False, index=work.index)
        for col in [c for c in ['Part Number', 'Name/Description', 'Notes', 'Assembly / SO Line'] if c in work.columns]:
            mask = mask | work[col].astype(str).str.contains(search_text, case=False, na=False)
        work = work[mask].copy()
    if selected_location != '(all)' and 'Location' in work.columns:
        loc_series = work['Location'].fillna('').replace('', '(blank)')
        work = work[loc_series == selected_location].copy()
    if selected_source != '(all)' and 'Assembly / SO Line' in work.columns:
        work = work[work['Assembly / SO Line'].astype(str) == selected_source].copy()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric('Rows', len(work))
    with m2:
        if 'Part Number' in work.columns:
            st.metric('Unique parts', int(work['Part Number'].astype(str).nunique()))
    with m3:
        if 'Short' in work.columns:
            st.metric('Total short', int(pd.to_numeric(work['Short'], errors='coerce').fillna(0).sum()))
    with m4:
        if 'Assembly / SO Line' in work.columns:
            st.metric('Sources', int(work['Assembly / SO Line'].astype(str).nunique()))

    if work.empty:
        st.warning('No rows match the current filters.')
        return

    st.dataframe(work, use_container_width=True, hide_index=True, height=420)
    st.download_button(
        'Download filtered CSV',
        data=work.to_csv(index=False).encode('utf-8'),
        file_name=sos_safe_default_filename(f'{label}_filtered'),
        mime='text/csv',
        key='sos_dash_dl',
    )

    s1, s2 = st.columns(2)
    with s1:
        if 'Part Number' in work.columns and 'Short' in work.columns:
            by_part = (
                work.assign(Short_num=pd.to_numeric(work['Short'], errors='coerce').fillna(0), Needed_num=pd.to_numeric(work.get('Needed', 0), errors='coerce').fillna(0), Buildable_num=pd.to_numeric(work.get('Buildable Qty', 0), errors='coerce').fillna(0))
                .groupby('Part Number', as_index=False)
                .agg(Short_Total=('Short_num', 'sum'), Needed_Total=('Needed_num', 'sum'), Min_Buildable=('Buildable_num', 'min'), Lines=('Part Number', 'count'))
                .sort_values(['Short_Total', 'Needed_Total'], ascending=False)
            )
            st.markdown('**Summary by part**')
            st.dataframe(by_part, use_container_width=True, hide_index=True, height=260)
    with s2:
        if 'Location' in work.columns and 'Short' in work.columns:
            loc = work.copy()
            loc['Location'] = loc['Location'].fillna('').replace('', '(blank)')
            by_loc = (
                loc.assign(Short_num=pd.to_numeric(loc['Short'], errors='coerce').fillna(0))
                .groupby('Location', as_index=False)
                .agg(Short_Total=('Short_num', 'sum'), UniqueParts=('Part Number', 'nunique') if 'Part Number' in loc.columns else ('Location', 'count'), Lines=('Location', 'count'))
                .sort_values(['Short_Total', 'Lines'], ascending=False)
            )
            st.markdown('**Summary by location**')
            st.dataframe(by_loc, use_container_width=True, hide_index=True, height=260)




def render_sos_help_tab():
    st.subheader('Help / How to use SOS')
    st.markdown("""
Use this workspace to check **live SOS inventory and BOM availability** without changing anything in SOS.

**Single**
- Enter one exact item or part name.
- Choose a quantity.
- Click **Run single check**.
- The app looks up the item in SOS and explodes the BOM.

**Batch CSV**
- Upload a CSV with an item column and a quantity column.
- Supported item headers: `Item Name`, `ItemName`, `Part Name`, `Name`, `Item Number`, `Part Number`, `PN`
- Supported quantity headers: `Quantity`, `quantity`, `qty`, `QTY`
- Click **Run batch check**.

**Sales Order**
- Enter the SOS sales order number.
- Leave **Explode subassemblies** checked to sync full BOM demand.
- Click **Run sales order check**.

**Dashboard**
- Review the last result on-page.
- Filter shortages, search part numbers, and export the filtered CSV.

**Safety**
- This workspace is read-only.
- It checks live SOS data but does not create builds and does not write back to SOS.
""")

    template_df = pd.DataFrame([
        {'Part Number': '130-000101 Rev C', 'Quantity': 2},
        {'Part Number': '330-000038', 'Quantity': 12},
        {'Part Number': '720-000141', 'Quantity': 4},
    ])
    template_bytes = template_df.to_csv(index=False).encode('utf-8')

    st.markdown('**CSV template example**')
    st.dataframe(template_df, use_container_width=True, hide_index=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.download_button(
            'Download CSV template',
            data=template_bytes,
            file_name='sos_batch_template.csv',
            mime='text/csv',
            key='sos_template_dl',
            use_container_width=True,
        )
    with c2:
        st.caption('You can use `Part Number` or any of the supported item-name headers. Quantities should be whole numbers.')

def render_sos_workspace():
    st.subheader('SOS Inventory')
    st.caption('Read-only SOS inventory, BOM, and sales-order checks. No POST, no PUT, no build creation.')

    def _sos_mark_live_fetch(source: str, rows_count: int, shortage_count: int, note: str = ''):
        st.session_state['sos_last_fetch_time'] = pd.Timestamp.now().strftime('%Y-%m-%d %I:%M:%S %p')
        st.session_state['sos_last_source'] = source
        st.session_state['sos_last_rows'] = int(rows_count)
        st.session_state['sos_last_shortages'] = int(shortage_count)
        st.session_state['sos_last_note'] = note

    auth_url = sos_build_auth_url()
    with st.expander('SOS connection', expanded=True):
        c1, c2 = st.columns([1, 2])
        client: Optional[SOSReadonlyClient] = None
        status_text = 'Not connected'
        try:
            client, status_text = sos_get_authenticated_client()
        except Exception as exc:
            st.error(f'SOS authentication failed: {exc}')
        with c1:
            st.metric('Connection', 'Ready' if client else 'Not ready')
        with c2:
            st.write(status_text)
            if auth_url:
                st.markdown(f'[Connect to SOS]({auth_url})')
            else:
                st.info('OAuth link appears after SOS_CLIENT_ID and SOS_REDIRECT_URI are added to secrets.')
            if 'sos_refresh_token_latest' in st.session_state:
                latest = st.session_state['sos_refresh_token_latest']
                masked = latest[:8] + '...' + latest[-8:] if isinstance(latest, str) and len(latest) > 20 else 'Hidden'
                st.info('A new refresh token was returned. Replace the old SOS_REFRESH_TOKEN in your secrets after testing.')
                st.caption(f'New token received: {masked}')
        cc1, cc2 = st.columns([1, 3])
        if cc1.button('Forget SOS session token', key='forget_sos_session_btn'):
            for key in ['sos_access_token', 'sos_refresh_token_latest']:
                st.session_state.pop(key, None)
            st.rerun()
        with cc2.expander('Hosted secrets template'):
            st.code("""SOS_REFRESH_TOKEN="..."
SOS_CLIENT_ID="..."
SOS_CLIENT_SECRET="..."
SOS_REDIRECT_URI="https://your-app.streamlit.app/""" , language='toml')

    if client is None:
        st.warning('Add SOS secrets first. The workspace is ready, but it cannot talk to SOS until a refresh token or OAuth secrets are available.')
        return

    st.markdown('#### Live SOS status')
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Source', st.session_state.get('sos_last_source', 'Connected to live SOS'))
    m2.metric('Last sync', st.session_state.get('sos_last_fetch_time', 'Not run yet'))
    m3.metric('Rows returned', st.session_state.get('sos_last_rows', 0))
    m4.metric('Shortages', st.session_state.get('sos_last_shortages', 0))
    st.caption(st.session_state.get('sos_last_note', 'Run a check to confirm live SOS fetch and BOM explosion activity.'))

    headers = ['Part Number', 'Enough', 'Needed', 'On Hand', 'Buildable Qty', 'Short', 'Type', 'Name/Description', 'Location', 'Notes']
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['Single', 'Batch CSV', 'Sales Order', 'Dashboard', 'Help'])

    with tab1:
        st.subheader('Single item check')
        single_name = st.text_input('Item name or part number', key='sos_single_name')
        single_qty = st.number_input('Quantity', min_value=1, value=1, step=1, key='sos_single_qty')

        cfind1, cfind2 = st.columns([1, 1])
        if cfind1.button('Find items', key='sos_find_single_items'):
            try:
                query = single_name.strip()
                if not query:
                    st.warning('Enter an item name or part number first.')
                else:
                    with st.spinner('Talking to SOS live API and searching matching items...'):
                        matches = client.get_items_by_name(query)
                    st.session_state['sos_single_query'] = query
                    st.session_state['sos_single_candidates'] = [sos_serialize_item(x) for x in matches]
                    auto = sos_pick_best_item(matches, query)
                    st.session_state['sos_single_auto_item_id'] = int(auto.item_id) if auto else None
            except Exception as exc:
                st.error(str(exc))
        if cfind2.button('Clear item search', key='sos_clear_single_items'):
            for key in ['sos_single_query', 'sos_single_candidates', 'sos_single_auto_item_id']:
                st.session_state.pop(key, None)
            st.rerun()

        selected_item = None
        stored_query = st.session_state.get('sos_single_query', '')
        stored_candidates = st.session_state.get('sos_single_candidates', [])
        if stored_query and stored_query == single_name.strip():
            candidates = [sos_deserialize_item(x) for x in stored_candidates]
            if not candidates:
                st.warning('No SOS items matched that search.')
            else:
                exact_matches = [x for x in candidates if sos_normalize_item_text(x.fullname) == sos_normalize_item_text(single_name)]
                if len(exact_matches) == 1:
                    selected_item = exact_matches[0]
                    st.success(f'Exact SOS match found: {selected_item.fullname}')
                else:
                    label_map = {sos_item_choice_label(x): x for x in candidates}
                    options = list(label_map.keys())
                    default_item_id = st.session_state.get('sos_single_auto_item_id')
                    default_index = 0
                    for idx, opt in enumerate(options):
                        if label_map[opt].item_id == default_item_id:
                            default_index = idx
                            break
                    chosen_label = st.selectbox('Multiple SOS items found. Choose which one to fetch:', options, index=default_index, key='sos_single_choice')
                    selected_item = label_map[chosen_label]
                    st.caption(f'Selected: {selected_item.fullname}')

        if st.button('Run single check', key='sos_run_single'):
            try:
                query = single_name.strip()
                if not query:
                    st.warning('Enter an item name or part number first.')
                else:
                    if selected_item is None:
                        with st.spinner('Talking to SOS live API, finding matching items, and exploding BOM...'):
                            matches = client.get_items_by_name(query)
                        if not matches:
                            _sos_mark_live_fetch('Single item', 0, 0, 'Live SOS lookup ran, but no matching item was found.')
                            st.warning('No item found.')
                            return
                        st.session_state['sos_single_query'] = query
                        st.session_state['sos_single_candidates'] = [sos_serialize_item(x) for x in matches]
                        auto = sos_pick_best_item(matches, query)
                        st.session_state['sos_single_auto_item_id'] = int(auto.item_id) if auto else None
                        if len(matches) > 1 and not any(sos_normalize_item_text(x.fullname) == sos_normalize_item_text(query) for x in matches):
                            st.info('Multiple SOS items were found. Choose the correct one from the dropdown, then click Run single check again.')
                            st.rerun()
                        selected_item = auto

                    with st.spinner('Talking to SOS live API, syncing the selected item, and exploding BOM...'):
                        rows = sos_bom_rows_from_selected_item(client, selected_item, int(single_qty), explode=True)
                        build_summary = sos_buildability_summary(selected_item, rows)
                    if not rows:
                        _sos_mark_live_fetch('Single item', 0, 0, 'Live SOS lookup ran, but the selected item returned no BOM rows.')
                        st.warning('No BOM rows returned for the selected item.')
                    else:
                        df = sos_rows_to_dataframe(rows, headers)
                        shortage_count = int(pd.to_numeric(df['Short'], errors='coerce').fillna(0).gt(0).sum()) if 'Short' in df.columns else 0
                        _sos_mark_live_fetch('Single item', len(df), shortage_count, f'Live SOS fetch complete. BOM exploded for {selected_item.fullname} x{int(single_qty)}.')
                        st.success(f'Live SOS fetch complete for {selected_item.fullname}. BOM rows returned: {len(df)}. Shortage rows: {shortage_count}.')
                        st.caption(f'Data source: SOS live API • Fetched at: {st.session_state.get("sos_last_fetch_time")}')
                        st.session_state['sos_last_df'] = df.copy()
                        st.session_state['sos_last_label'] = f'Single_{selected_item.fullname}_x{single_qty}'
                        csum1, csum2 = st.columns([1, 2])
                        csum1.metric('Assembly on hand', build_summary['assembly_on_hand'])
                        csum2.metric('Assembly location', build_summary['assembly_location'])

                        csum3, csum4, csum5 = st.columns(3)
                        csum3.metric('Buildable from parts', build_summary['buildable_from_parts'])
                        csum4.metric('Potential total', build_summary['potential_total'])
                        csum5.metric('Purchase cost', f"{build_summary['assembly_purchase_cost']:.2f}")

                        if build_summary['limiting_parts']:
                            st.caption(f"Limiting part(s): {build_summary['limiting_parts']}")
                        st.dataframe(df, use_container_width=True, hide_index=True, height=460)
                        st.download_button('Download CSV', data=df.to_csv(index=False).encode('utf-8'), file_name=sos_safe_default_filename(f'Single_{selected_item.fullname}_x{single_qty}'), mime='text/csv', key='sos_single_dl')
            except Exception as exc:
                st.error(str(exc))

    with tab2:
        st.subheader('Batch CSV check')
        uploaded = st.file_uploader('Upload CSV', type=['csv'], key='sos_batch_csv')
        if st.button('Run batch check', key='sos_run_batch'):
            if uploaded is None:
                st.warning('Upload a CSV first.')
            else:
                try:
                    with st.spinner('Talking to SOS live API, checking CSV items, and exploding BOMs...'):
                        reqs = sos_load_requests_from_csv(uploaded)
                        frames: List[pd.DataFrame] = []
                        for name, qty in reqs:
                            rows = sos_bom_rows_from_item(client, name, qty, explode=True)
                            if rows:
                                frames.append(sos_rows_to_dataframe(rows, headers, source_label=f'{name} x{qty}'))
                    if not frames:
                        _sos_mark_live_fetch('Batch CSV', 0, 0, 'Live SOS batch lookup ran, but no matching items were found from the CSV.')
                        st.warning('No matching items found from the CSV.')
                    else:
                        df = pd.concat(frames, ignore_index=True)
                        shortage_count = int(pd.to_numeric(df['Short'], errors='coerce').fillna(0).gt(0).sum()) if 'Short' in df.columns else 0
                        _sos_mark_live_fetch('Batch CSV', len(df), shortage_count, f'Live SOS batch fetch complete. Checked {len(reqs)} CSV lines and returned {len(df)} BOM rows.')
                        st.success(f'Live SOS batch fetch complete. Rows returned: {len(df)}. Shortage rows: {shortage_count}.')
                        st.caption(f'Data source: SOS live API • Fetched at: {st.session_state.get("sos_last_fetch_time")}')
                        st.session_state['sos_last_df'] = df.copy()
                        st.session_state['sos_last_label'] = 'batch_inventory_check'
                        st.dataframe(df, use_container_width=True, hide_index=True, height=520)
                        st.download_button('Download CSV', data=df.to_csv(index=False).encode('utf-8'), file_name='batch_inventory_check.csv', mime='text/csv', key='sos_batch_dl')
                except Exception as exc:
                    st.error(str(exc))

    with tab3:
        st.subheader('Sales Order check')
        so_number = st.text_input('Sales order number', key='sos_so_number')
        explode = st.checkbox('Explode subassemblies', value=True, key='sos_so_explode')
        if st.button('Run sales order check', key='sos_run_so'):
            try:
                with st.spinner('Talking to SOS live API, fetching sales order, and syncing BOM lines...'):
                    df = sos_grouped_sales_order_dataframe(client, so_number.strip(), explode=explode)
                if df.empty:
                    _sos_mark_live_fetch('Sales Order', 0, 0, f'Live SOS sales order lookup ran for {so_number.strip()}, but no lines were returned.')
                    st.warning('No lines found for that sales order.')
                else:
                    shortage_count = int(pd.to_numeric(df['Short'], errors='coerce').fillna(0).gt(0).sum()) if 'Short' in df.columns else 0
                    mode_note = 'BOM exploded from SOS assemblies.' if explode else 'Sales order lines fetched without subassembly explosion.'
                    _sos_mark_live_fetch('Sales Order', len(df), shortage_count, f'Live SOS sales order fetch complete for {so_number.strip()}. {mode_note}')
                    st.success(f'Live SOS sales order fetch complete. Rows returned: {len(df)}. Shortage rows: {shortage_count}.')
                    st.caption(f'Data source: SOS live API • Fetched at: {st.session_state.get("sos_last_fetch_time")}')
                    st.session_state['sos_last_df'] = df.copy()
                    st.session_state['sos_last_label'] = f'SO_{so_number}_inventory_check'
                    if 'Assembly / SO Line' in df.columns and 'Buildable Qty' in df.columns:
                        summary_rows = []
                        for source_name, grp in df.groupby('Assembly / SO Line'):
                            buildable = int(pd.to_numeric(grp['Buildable Qty'], errors='coerce').fillna(0).min()) if not grp.empty else 0
                            limiting_parts = ', '.join(grp.loc[pd.to_numeric(grp['Buildable Qty'], errors='coerce').fillna(0) == buildable, 'Part Number'].astype(str).tolist())
                            requested_qty = int(pd.to_numeric(grp['Needed'], errors='coerce').fillna(0).max()) if not grp.empty else 0
                            summary_rows.append({
                                'Assembly / SO Line': source_name,
                                'Requested Qty': requested_qty,
                                'Buildable From Parts': buildable,
                                'Enough For Requested Qty': '✅' if buildable >= requested_qty else '❌',
                                'Limiting Part(s)': limiting_parts,
                            })
                        if summary_rows:
                            st.markdown('**Buildability summary**')
                            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True, height=220)
                    st.dataframe(df, use_container_width=True, hide_index=True, height=520)
                    st.download_button('Download CSV', data=df.to_csv(index=False).encode('utf-8'), file_name=sos_safe_default_filename(f'SO_{so_number}_inventory_check'), mime='text/csv', key='sos_so_dl')
            except Exception as exc:
                st.error(str(exc))

    with tab4:
        render_sos_dashboard_viewer()

    with tab5:
        render_sos_help_tab()

def render_workspace_selector():
    options = ['Derate Reports', 'Arduino Viewer', 'Plot Explorer', 'Label Studio', 'RF Calculator', 'SOS Inventory']
    selected = st.radio(
        'Workspace',
        options,
        index=options.index(st.session_state.get('active_workspace', 'RF Calculator')) if st.session_state.get('active_workspace', 'RF Calculator') in options else 0,
        horizontal=True,
        key='active_workspace',
        label_visibility='collapsed',
    )
    return selected





def render_label_tab():
    st.subheader('Label')
    st.caption('Choose a customer label pattern, edit the layout on the left, and review the live preview on the right before exporting the PDF.')
    tab_nab, tab_jabil = st.tabs(['Nabtesco', 'Jabil'])
    with tab_nab:
        render_nabtesco_editor()
    with tab_jabil:
        render_jabil_editor()

def render_plot_tab():
    st.subheader('Plot')
    st.caption('Use TAR file from Wibotic Transmitter log.')

    left, right = st.columns([1.05, 1.35])

    with left:
        plot_file = st.file_uploader('Plot CSV/TAR', type=['csv', 'tar'], key='plot_file')
        plot_suffix = None
        if plot_file is not None and plot_file.name.lower().endswith('.tar'):
            try:
                plot_suffix_options = sorted([s for s, entry in scan_tar_bytes(plot_file.getvalue()).items() if 'RX' in entry and 'TX' in entry])
                plot_suffix = st.selectbox('RX/TX pair', plot_suffix_options, index=0, key='plot_suffix')
            except Exception as e:
                st.error(f'Could not inspect TAR: {e}')

        plot_title = st.text_input('Plot title', value='Data Plot', key='plot_title_input')
        signal_filter = st.text_input('Signal filter', value='', key='plot_signal_filter').strip().lower()

        c1, c2 = st.columns(2)
        smoothing_mode = c1.selectbox('Smoothing', ['None', 'Moving Average', 'Median', 'EMA'], index=0, key='plot_smoothing')
        x_axis_mode = c2.selectbox('X axis', ['Seconds', 'Minutes', 'Hours', 'Sample Index'], index=0, key='plot_xaxis')

        c3, c4 = st.columns(2)
        smoothing_window = c3.number_input('Window', min_value=1, value=5, step=1, key='plot_window')
        ignore_seconds = c4.number_input('Ignore first sec', min_value=0.0, value=60.0, step=10.0, key='plot_ignore')

    prepared_df = None
    source_caption = None
    all_plot_cols = []

    if plot_file is not None:
        try:
            raw_df, source_type, chosen_suffix = read_source_uploaded(plot_file, plot_suffix)
            prepared_df = prepare_loaded_dataframe(raw_df)
            all_plot_cols = get_plot_columns(prepared_df)
            source_caption = f"Loaded {plot_file.name} | source={source_type}" + (f" | pair={chosen_suffix}" if chosen_suffix else '')
            st.caption(source_caption)
        except Exception as e:
            st.error(f'Failed to load plot file: {e}')

    with right:
        if prepared_df is None:
            st.info('Upload a CSV or TAR file to use the Plot tab.')
            return

        filtered_cols = [c for c in all_plot_cols if signal_filter in c.lower()]
        if not filtered_cols:
            st.warning('No numeric signals match the current filter.')
            return

        st.write('Preset selection')
        p1, p2, p3, p4, p5, p6, p7 = st.columns(7)
        if p1.button('Recommended', key='preset_rec'):
            st.session_state['plot_selected_columns'] = apply_preset(filtered_cols, 'Recommended')
        if p2.button('Temp', key='preset_temp'):
            st.session_state['plot_selected_columns'] = apply_preset(filtered_cols, 'Temp')
        if p3.button('Voltage', key='preset_volt'):
            st.session_state['plot_selected_columns'] = apply_preset(filtered_cols, 'Voltage')
        if p4.button('Power', key='preset_power'):
            st.session_state['plot_selected_columns'] = apply_preset(filtered_cols, 'Power')
        if p5.button('Rx', key='preset_rx'):
            st.session_state['plot_selected_columns'] = apply_preset(filtered_cols, 'Rx')
        if p6.button('Tx', key='preset_tx'):
            st.session_state['plot_selected_columns'] = apply_preset(filtered_cols, 'Tx')
        if p7.button('Clear', key='preset_clear'):
            st.session_state['plot_selected_columns'] = []

        default_selection = st.session_state.get('plot_selected_columns')
        valid_default = [c for c in (default_selection or []) if c in filtered_cols]
        if not valid_default:
            valid_default = [c for c in apply_preset(filtered_cols, 'Recommended') if c in filtered_cols]

        selected_columns = st.multiselect(
            'Signals to plot',
            options=filtered_cols,
            default=valid_default,
            format_func=friendly_label,
            key='plot_selected_columns'
        )

        st.write('Scale factors')
        scale_map = {}
        show_cols = selected_columns[:24]
        more_count = max(0, len(selected_columns) - len(show_cols))
        grid_cols = st.columns(3)
        for i, col in enumerate(show_cols):
            scale_map[col] = grid_cols[i % 3].number_input(
                f'{col} scale',
                value=1.0,
                step=0.1,
                format='%.3f',
                key=f'scale_{col}'
            )
        if more_count:
            st.caption(f'{more_count} more selected signals are using scale 1.0 and are hidden to keep the page manageable.')
        for col in selected_columns[24:]:
            scale_map[col] = 1.0

        if not selected_columns:
            st.warning('Select one or more signals to plot.')
            return

        filtered_df = prepared_df.copy()
        filtered_df['Time_sec'] = pd.to_numeric(filtered_df['Time_sec'], errors='coerce')
        filtered_df = filtered_df[filtered_df['Time_sec'] > float(ignore_seconds)].copy()
        if filtered_df.empty:
            st.warning('No rows remain after the ignore-first-seconds filter.')
            return

        x_data, x_label = get_time_axis_values(filtered_df['Time_sec'], x_axis_mode)
        fig = plt.figure(figsize=(11.5, 6.2))
        ax = fig.add_subplot(111)
        export_df = pd.DataFrame({'Time_sec': filtered_df['Time_sec'].reset_index(drop=True)})

        for col in selected_columns:
            y_data = pd.to_numeric(filtered_df[col], errors='coerce') * float(scale_map.get(col, 1.0))
            y_data = smooth_series(y_data, smoothing_mode, smoothing_window)
            valid = y_data.notna()
            if valid.sum() == 0:
                continue
            ax.plot(x_data[valid.to_numpy()], y_data[valid], label=f'{col} (Scale: {float(scale_map.get(col, 1.0))})', linewidth=0.9)
            avg_value = float(y_data[valid].mean())
            ax.axhline(y=avg_value, linestyle='--', linewidth=0.7, alpha=0.6)
            export_df[col] = (pd.to_numeric(filtered_df[col], errors='coerce') * float(scale_map.get(col, 1.0))).reset_index(drop=True)
            export_df[f'{col}_smoothed'] = y_data.reset_index(drop=True)

        ax.set_xlabel(x_label)
        ax.set_ylabel('Value')
        ax.set_title(plot_title)
        ax.grid(True, alpha=0.45)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=False)

        stats_df = compute_stats_text(filtered_df, selected_columns, scale_map)
        cstats1, cstats2 = st.columns([1, 1])
        cstats1.dataframe(stats_df, use_container_width=True, height=260)
        preview_cols = ['Time_sec'] + selected_columns[:6]
        cstats2.dataframe(filtered_df[preview_cols].head(250), use_container_width=True, height=260)

        png_bytes = io.BytesIO()
        fig.savefig(png_bytes, format='png', dpi=150, bbox_inches='tight')
        png_bytes.seek(0)
        st.download_button('Download plot PNG', png_bytes.getvalue(), file_name='plot.png', mime='image/png')
        st.download_button('Download filtered CSV', export_df.to_csv(index=False).encode('utf-8'), file_name='filtered_plot_data.csv', mime='text/csv')




def _metric_cards(items):
    """Simple metric card helper used across workspaces."""
    if not items:
        return
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        if isinstance(item, (list, tuple)):
            label = item[0] if len(item) > 0 else ''
            value = item[1] if len(item) > 1 else ''
            help_text = item[2] if len(item) > 2 else None
        else:
            label, value, help_text = str(item), '', None
        with col:
            st.metric(label, value, help=help_text)


def _workspace_intro(title, description=''):
    st.markdown(f"### {title}")
    if description:
        st.caption(description)


def render_app_header():
    st.markdown("## WiBotic Engineering Toolkit")
    st.caption("Derate, Arduino, Plot, Label, RF, and SOS inventory tools in one Streamlit app.")


# -----------------------------
# UI
# -----------------------------
def inject_branding():
    st.markdown(
        f"""
        <style>
        :root {{
            --wb-blue: #17a8ff;
            --wb-blue-deep: #0c5ea8;
            --wb-blue-soft: #eaf6ff;
            --wb-text: #0f172a;
            --wb-muted: #5b6472;
            --wb-border: rgba(23, 168, 255, 0.18);
            --wb-shadow: 0 14px 36px rgba(15, 23, 42, 0.08);
        }}
        .stApp {{
            background:
                radial-gradient(circle at top right, rgba(23,168,255,0.09), transparent 30%),
                linear-gradient(180deg, #f8fbff 0%, #f4f8fc 100%);
            color: var(--wb-text);
        }}
        [data-testid="stHeader"] {{
            background: rgba(248, 251, 255, 0.7);
            backdrop-filter: blur(10px);
        }}
        [data-testid="stToolbar"] {{ right: 1rem; }}
        .block-container {{
            padding-top: 1.35rem;
            padding-bottom: 2rem;
            max-width: 1380px;
        }}
        .wibotic-watermark {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -42%);
            width: min(44vw, 620px);
            height: min(44vw, 620px);
            background-image: url("data:image/png;base64,{LOGO_B64}");
            background-repeat: no-repeat;
            background-size: contain;
            background-position: center;
            opacity: 0.07;
            pointer-events: none;
            z-index: 0;
            filter: saturate(1.15) hue-rotate(4deg);
        }}
        .wibotic-watermark::after {{
            content: "";
            position: absolute;
            inset: -12%;
            background: radial-gradient(circle, rgba(23,168,255,0.10) 0%, rgba(23,168,255,0.04) 42%, transparent 72%);
            z-index: -1;
            filter: blur(8px);
        }}
        .wibotic-hero {{
            position: relative;
            z-index: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            gap: 0.95rem;
            padding: 2rem 2rem 1.6rem 2rem;
            margin: 0 auto 1.15rem auto;
            background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(245,250,255,0.90));
            border: 1px solid rgba(23,168,255,0.20);
            border-radius: 28px;
            box-shadow: 0 18px 48px rgba(15, 23, 42, 0.08);
            max-width: 1120px;
            overflow: hidden;
        }}
        .wibotic-hero::before {{
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(180deg, rgba(23,168,255,0.06), transparent 30%, transparent 70%, rgba(12,94,168,0.04));
            pointer-events: none;
        }}
        .wibotic-hero__logo {{
            width: 132px;
            height: 132px;
            border-radius: 34px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: radial-gradient(circle at 50% 35%, rgba(255,255,255,0.98) 0%, rgba(232,244,255,0.96) 55%, rgba(191,228,255,0.72) 100%);
            border: 1px solid rgba(23,168,255,0.22);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.9), 0 16px 36px rgba(23,168,255,0.16);
            flex: 0 0 auto;
        }}
        .wibotic-hero__logo img {{
            width: 92px;
            height: 92px;
            object-fit: contain;
            filter: drop-shadow(0 6px 10px rgba(12,94,168,0.10));
        }}
        .wibotic-hero__title {{
            margin: 0;
            font-size: clamp(2.4rem, 5vw, 4.2rem);
            line-height: 1.02;
            font-weight: 900;
            color: #071a3d;
            letter-spacing: -0.04em;
        }}
        .wibotic-hero__subtitle {{
            margin: 0.2rem 0 0 0;
            color: var(--wb-muted);
            font-size: clamp(1rem, 1.55vw, 1.35rem);
            max-width: 900px;
        }}
        .wibotic-badges {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 0.55rem;
            margin-top: 0.85rem;
        }}
        .wibotic-badge {{
            padding: 0.38rem 0.72rem;
            border-radius: 999px;
            font-size: 0.84rem;
            font-weight: 700;
            color: var(--wb-blue-deep);
            background: rgba(23,168,255,0.1);
            border: 1px solid rgba(23,168,255,0.18);
        }}
        div[role="radiogroup"] {{
            gap: 0.55rem;
            flex-wrap: wrap;
        }}
        div[role="radiogroup"] > label[data-baseweb="radio"] {{
            margin-right: 0 !important;
            background: rgba(255,255,255,0.78);
            border: 1px solid var(--wb-border);
            border-radius: 18px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
            padding: 0.58rem 0.95rem;
            min-height: 48px;
        }}
        div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {{
            display: none;
        }}
        div[role="radiogroup"] > label[data-baseweb="radio"] p {{
            font-weight: 700 !important;
            color: #476072 !important;
        }}
        div[role="radiogroup"] > label[data-baseweb="radio"][aria-checked="true"] {{
            background: linear-gradient(135deg, rgba(23,168,255,0.22), rgba(12,94,168,0.18)) !important;
            border-color: rgba(23,168,255,0.42) !important;
        }}
        div[role="radiogroup"] > label[data-baseweb="radio"][aria-checked="true"] p {{
            color: #0b2942 !important;
        }}
        div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stFileUploader"]),
        div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stDataFrame"]),
        div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] > div:has(> div.stDownloadButton) {{
            position: relative;
            z-index: 1;
        }}
        div[data-testid="stFileUploader"],
        div[data-testid="stDataFrame"],
        div[data-testid="stMetric"],
        .stAlert,
        .stPlotlyChart,
        .stDownloadButton > button,
        .stButton > button,
        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea,
        .stSelectbox div[data-baseweb="select"],
        .stMultiSelect div[data-baseweb="select"] {{
            border-radius: 14px;
        }}
        .stButton > button,
        .stDownloadButton > button {{
            border: 1px solid rgba(23,168,255,0.2);
            background: linear-gradient(180deg, #ffffff, #f3faff);
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
            font-weight: 700;
        }}
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, #17a8ff, #0c5ea8);
            color: white;
            border: none;
        }}
        h1, h2, h3 {{ color: #0b2942; letter-spacing: -0.02em; }}
        p, label, .stCaption {{ color: var(--wb-muted); }}
        @media (max-width: 900px) {{
            .wibotic-hero {{ padding: 1.4rem 1rem 1.2rem 1rem; }}
            .wibotic-hero__logo {{ width: 108px; height: 108px; border-radius: 28px; }}
            .wibotic-hero__logo img {{ width: 78px; height: 78px; }}
            .wibotic-watermark {{ width: min(70vw, 360px); height: min(70vw, 360px); opacity: 0.05; transform: translate(-50%, -34%); }}
        }}
        </style>
        <div class="wibotic-watermark"></div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# RF calculator helpers
# -----------------------------
def rf_resonance_calculate(freq_mhz=None, inductance_uh=None, capacitance_pf=None):
    values = [freq_mhz is not None and freq_mhz > 0, inductance_uh is not None and inductance_uh > 0, capacitance_pf is not None and capacitance_pf > 0]
    if sum(values) < 2:
        raise ValueError('Enter at least two positive values.')

    freq_hz = freq_mhz * 1e6 if freq_mhz is not None and freq_mhz > 0 else None
    inductance_h = inductance_uh * 1e-6 if inductance_uh is not None and inductance_uh > 0 else None
    capacitance_f = capacitance_pf * 1e-12 if capacitance_pf is not None and capacitance_pf > 0 else None

    if freq_hz is not None and inductance_h is not None and capacitance_f is None:
        capacitance_f = 1 / (4 * math.pi**2 * freq_hz**2 * inductance_h)
    elif freq_hz is not None and capacitance_f is not None and inductance_h is None:
        inductance_h = 1 / (4 * math.pi**2 * freq_hz**2 * capacitance_f)
    elif inductance_h is not None and capacitance_f is not None and freq_hz is None:
        freq_hz = 1 / (2 * math.pi * math.sqrt(inductance_h * capacitance_f))
    elif freq_hz is not None and inductance_h is not None and capacitance_f is not None:
        pass
    else:
        raise ValueError('Unable to solve with the provided values.')

    return {
        'frequency_mhz': freq_hz * 1e-6,
        'inductance_uh': inductance_h * 1e6,
        'capacitance_pf': capacitance_f * 1e12,
    }


def bank_capacitance_calculate(bank1_vals_pf, bank2_vals_pf, output_unit='pF', tolerance_pct=None):
    bank1_pf = sum(v for v in bank1_vals_pf if v is not None and v > 0)
    bank2_pf = sum(v for v in bank2_vals_pf if v is not None and v > 0)
    total_pf = 0.0 if bank1_pf <= 0 or bank2_pf <= 0 else (bank1_pf * bank2_pf) / (bank1_pf + bank2_pf)

    div = 1e6 if output_unit == 'µF' else 1.0
    result = {
        'bank1': bank1_pf / div,
        'bank2': bank2_pf / div,
        'total': total_pf / div,
        'unit': output_unit,
    }

    if tolerance_pct is not None and tolerance_pct > 0:
        tol = result['total'] * (tolerance_pct / 100.0)
        result['min_total'] = result['total'] - tol
        result['max_total'] = result['total'] + tol
    return result


COIL_PRESETS = {
    "215-000086 rev C — RC-1K Loop": {
        "part_number": "215-000086 rev C",
        "coil_name": "RC-1K Loop",
        "topology": "single_parallel",
        "bank1_label": "Single parallel bank",
        "bank2_label": "Not used",
        "bank1": ["C1", "C2", "C3", "C4", "C5", "C6", "C7"],
        "bank2": [],
        "notes": "All populated capacitors are in parallel. In your reference sheet, Bank 2 is marked with x to indicate this single-bank parallel build.",
    },
    "215-000091 rev B — TC-1K-G2 Loop": {
        "part_number": "215-000091 rev B",
        "coil_name": "TC-1K-G2 Loop",
        "topology": "single_parallel",
        "bank1_label": "Single parallel bank",
        "bank2_label": "Not used",
        "bank1": ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"],
        "bank2": [],
        "notes": "All populated capacitors are in parallel. Use this for the loop build where one effective bank sets the total tuning capacitance.",
    },
    "215-000092 rev C — TC-1K Tank": {
        "part_number": "215-000092 rev C",
        "coil_name": "TC-1K Tank",
        "topology": "two_parallel_series",
        "bank1_label": "Bank 1 (parallel)",
        "bank2_label": "Bank 2 (parallel)",
        "bank1": ["C9", "C10", "C11", "C12", "C13"],
        "bank2": ["C4", "C5", "C6", "C7", "C8"],
        "notes": "C9 through C13 form one parallel bank. C4 through C8 form the second parallel bank. The two banks are then in series.",
    },
}


def coil_preset_capacitance_calculate(preset_name, cap_map_pf, output_unit='pF', tolerance_pct=None):
    preset = COIL_PRESETS[preset_name]
    bank1_pf = sum(float(cap_map_pf.get(ref, 0) or 0) for ref in preset['bank1'])
    bank2_pf = sum(float(cap_map_pf.get(ref, 0) or 0) for ref in preset['bank2']) if preset['bank2'] else 0.0

    if preset['topology'] == 'single_parallel':
        total_pf = bank1_pf
    else:
        total_pf = 0.0 if bank1_pf <= 0 or bank2_pf <= 0 else (bank1_pf * bank2_pf) / (bank1_pf + bank2_pf)

    div = 1e6 if output_unit == 'µF' else 1.0
    result = {
        'bank1_pf': bank1_pf,
        'bank2_pf': bank2_pf,
        'total_pf': total_pf,
        'bank1': bank1_pf / div,
        'bank2': bank2_pf / div,
        'total': total_pf / div,
        'unit': output_unit,
        'topology': preset['topology'],
        'populated_bank1': [ref for ref in preset['bank1'] if float(cap_map_pf.get(ref, 0) or 0) > 0],
        'populated_bank2': [ref for ref in preset['bank2'] if float(cap_map_pf.get(ref, 0) or 0) > 0],
    }
    if tolerance_pct is not None and tolerance_pct > 0:
        tol = result['total'] * (tolerance_pct / 100.0)
        result['min_total'] = result['total'] - tol
        result['max_total'] = result['total'] + tol
    return result


def render_rf_tab():
    _workspace_intro('RF Calculator', 'Capacitance-bank and RF resonance tools with quick presets, coil-aware bank topology, and result cards for common lab work.')

    top_left, top_right = st.columns([1.0, 1.25], gap='large')
    with top_left:
        st.markdown('#### RF resonance')
        preset_cols = st.columns(4)
        presets = [('13.56 MHz', 13.56, None, None), ('6.78 MHz', 6.78, None, None), ('125 kHz', 0.125, None, None), ('Clear', 0.0, 0.0, 0.0)]
        for col, (label, fval, lval, cval) in zip(preset_cols, presets):
            if col.button(label, key=f'rf_preset_{label}', use_container_width=True):
                st.session_state['rf_freq_mhz_v8'] = fval or 0.0
                st.session_state['rf_inductance_uh_v8'] = lval or 0.0
                st.session_state['rf_capacitance_pf_v8'] = cval or 0.0

        rf1, rf2, rf3 = st.columns(3)
        freq_mhz = rf1.number_input('Frequency (MHz)', min_value=0.0, value=float(st.session_state.get('rf_freq_mhz_v8', 0.0)), step=0.1, format='%.3f', key='rf_freq_mhz_v8')
        inductance_uh = rf2.number_input('Inductance (µH)', min_value=0.0, value=float(st.session_state.get('rf_inductance_uh_v8', 0.0)), step=0.1, format='%.3f', key='rf_inductance_uh_v8')
        capacitance_pf = rf3.number_input('Capacitance (pF)', min_value=0.0, value=float(st.session_state.get('rf_capacitance_pf_v8', 0.0)), step=0.1, format='%.3f', key='rf_capacitance_pf_v8')
        if st.button('Calculate RF resonance', use_container_width=True, key='rf_calc_btn_v8'):
            try:
                st.session_state['rf_result_v8'] = rf_resonance_calculate(
                    freq_mhz=freq_mhz if freq_mhz > 0 else None,
                    inductance_uh=inductance_uh if inductance_uh > 0 else None,
                    capacitance_pf=capacitance_pf if capacitance_pf > 0 else None,
                )
                st.session_state.pop('rf_error_v8', None)
            except Exception as e:
                st.session_state['rf_error_v8'] = str(e)
        if st.session_state.get('rf_error_v8'):
            st.error(st.session_state['rf_error_v8'])
        if st.session_state.get('rf_result_v8'):
            res = st.session_state['rf_result_v8']
            _metric_cards([
                ('Frequency', f"{res['frequency_mhz']:.3f} MHz", None),
                ('Inductance', f"{res['inductance_uh']:.3f} µH", None),
                ('Capacitance', f"{res['capacitance_pf']:.3f} pF", None),
            ])

    with top_right:
        st.markdown('#### Coil presets')
        with st.expander('Help', expanded=False):
            st.write('Choose a preset coil, enter the populated capacitor values by reference designator, and the app will calculate the effective bank and total capacitance based on the coil topology.')
            st.write('For the single-bank builds, the x marks in your reference sheet mean the build acts as one parallel bank. For the tank build, Bank 1 and Bank 2 are parallel internally and then placed in series.')

        preset_name = st.selectbox('Reference designator / coil preset', list(COIL_PRESETS.keys()), key='rf_coil_preset_v10')
        preset = COIL_PRESETS[preset_name]
        st.caption(f"{preset['part_number']} | {preset['coil_name']}")
        _metric_cards([
            ('Topology', 'Single parallel bank' if preset['topology'] == 'single_parallel' else 'Two parallel banks in series', None),
            ('Bank 1 refs', ', '.join(preset['bank1']), None),
            ('Bank 2 refs', ', '.join(preset['bank2']) if preset['bank2'] else 'Not used', None),
        ])
        st.info(preset['notes'])

        setting_col1, setting_col2, setting_col3 = st.columns(3)
        preset_unit = setting_col1.selectbox('Result unit', ['pF', 'µF'], index=0, key='coil_bank_unit_v10')
        preset_tolerance = setting_col2.number_input('Tolerance (%)', min_value=0.0, value=0.0, step=0.5, format='%.2f', key='coil_bank_tol_v10')
        preset_l_uh = setting_col3.number_input('Inductance (µH)', min_value=0.0, value=float(st.session_state.get('coil_preset_l_v10', 0.0)), step=0.1, format='%.3f', key='coil_preset_l_v10')
        target_freq_mhz = st.number_input('Target frequency (MHz) optional', min_value=0.0, value=float(st.session_state.get('coil_target_freq_v10', 0.0)), step=0.01, format='%.3f', key='coil_target_freq_v10')

        input_col, preview_col = st.columns([1.15, 0.85], gap='large')
        cap_map = {}
        with input_col:
            if preset['bank2']:
                bank_col1, bank_col2 = st.columns(2, gap='large')
                with bank_col1:
                    st.markdown(f"##### {preset['bank1_label']}")
                    for ref in preset['bank1']:
                        cap_map[ref] = st.number_input(f'{ref} (pF)', min_value=0.0, value=float(st.session_state.get(f'coilcap::{preset_name}::{ref}', 0.0)), step=0.1, format='%.2f', key=f'coilcap::{preset_name}::{ref}')
                with bank_col2:
                    st.markdown(f"##### {preset['bank2_label']}")
                    for ref in preset['bank2']:
                        cap_map[ref] = st.number_input(f'{ref} (pF)', min_value=0.0, value=float(st.session_state.get(f'coilcap::{preset_name}::{ref}', 0.0)), step=0.1, format='%.2f', key=f'coilcap::{preset_name}::{ref}')
            else:
                st.markdown('##### Single parallel bank')
                for ref in preset['bank1']:
                    cap_map[ref] = st.number_input(f'{ref} (pF)', min_value=0.0, value=float(st.session_state.get(f'coilcap::{preset_name}::{ref}', 0.0)), step=0.1, format='%.2f', key=f'coilcap::{preset_name}::{ref}')

            if st.button('Calculate preset coil capacitance', type='primary', use_container_width=True, key='coil_bank_calc_btn_v10'):
                try:
                    res = coil_preset_capacitance_calculate(preset_name, cap_map, output_unit=preset_unit, tolerance_pct=preset_tolerance)
                    st.session_state['coil_bank_result_v10'] = res
                    st.session_state['coil_bank_result_name_v10'] = preset_name
                    st.session_state.pop('coil_bank_error_v10', None)
                    if preset_l_uh > 0 and res['total_pf'] > 0:
                        st.session_state['coil_resonance_v10'] = rf_resonance_calculate(inductance_uh=preset_l_uh, capacitance_pf=res['total_pf'])
                    else:
                        st.session_state.pop('coil_resonance_v10', None)
                    if target_freq_mhz > 0 and preset_l_uh > 0:
                        st.session_state['coil_target_result_v10'] = rf_resonance_calculate(freq_mhz=target_freq_mhz, inductance_uh=preset_l_uh)
                    else:
                        st.session_state.pop('coil_target_result_v10', None)
                except Exception as e:
                    st.session_state['coil_bank_error_v10'] = str(e)

        with preview_col:
            st.markdown('##### Preset summary')
            st.write(f"**Part number:** {preset['part_number']}")
            st.write(f"**Coil:** {preset['coil_name']}")
            st.write(f"**Topology:** {'Single parallel bank' if preset['topology'] == 'single_parallel' else 'Two parallel banks in series'}")
            st.write(f"**Bank 1:** {', '.join(preset['bank1'])}")
            st.write(f"**Bank 2:** {', '.join(preset['bank2']) if preset['bank2'] else 'Not used'}")
            populated_refs = [ref for ref, val in cap_map.items() if float(val) > 0]
            st.write(f"**Populated refs entered:** {', '.join(populated_refs) if populated_refs else 'None yet'}")
            if preset['topology'] == 'single_parallel':
                st.caption('This preset acts as one parallel bank. Enter only the populated capacitor values.')
            else:
                st.caption('Each bank is parallel internally. Bank 1 and Bank 2 are then combined in series.')

        if st.session_state.get('coil_bank_error_v10'):
            st.error(st.session_state['coil_bank_error_v10'])

        stored_name = st.session_state.get('coil_bank_result_name_v10')
        if stored_name == preset_name and st.session_state.get('coil_bank_result_v10'):
            res = st.session_state['coil_bank_result_v10']
            cards = [('Bank 1', f"{res['bank1']:.3f} {res['unit']}", None)]
            if preset['bank2']:
                cards.append(('Bank 2', f"{res['bank2']:.3f} {res['unit']}", None))
            cards.append(('Total', f"{res['total']:.3f} {res['unit']}", None))
            _metric_cards(cards)
            if res['populated_bank1']:
                st.caption('Populated Bank 1 refs: ' + ', '.join(res['populated_bank1']))
            if preset['bank2'] and res['populated_bank2']:
                st.caption('Populated Bank 2 refs: ' + ', '.join(res['populated_bank2']))
            if 'min_total' in res:
                st.info(f"Tolerance range: {res['min_total']:.3f} to {res['max_total']:.3f} {res['unit']}")

        if stored_name == preset_name and st.session_state.get('coil_resonance_v10'):
            rr = st.session_state['coil_resonance_v10']
            st.markdown('##### Resonance from preset capacitance')
            _metric_cards([
                ('Resonant frequency', f"{rr['frequency_mhz']:.3f} MHz", None),
                ('Inductance used', f"{rr['inductance_uh']:.3f} µH", None),
                ('Effective C', f"{rr['capacitance_pf']:.3f} pF", None),
            ])
        if stored_name == preset_name and st.session_state.get('coil_target_result_v10'):
            tr = st.session_state['coil_target_result_v10']
            st.markdown('##### Required total capacitance for target frequency')
            _metric_cards([
                ('Target frequency', f"{tr['frequency_mhz']:.3f} MHz", None),
                ('Inductance used', f"{tr['inductance_uh']:.3f} µH", None),
                ('Required total C', f"{tr['capacitance_pf']:.3f} pF", None),
            ])

def render_derate_workspace():
    _workspace_intro('Derate Reports', 'Align a transmitter TAR/CSV with chamber data, generate the derate chart, and export summary tables.')
    left, right = st.columns([1, 1], gap='large')
    with left:
        with st.expander('Help', expanded=False):
            st.write('Upload the main transmitter TAR/CSV and the chamber CSV. Then choose the power signal and generate the report.')
        main_file = st.file_uploader('Main TAR/CSV', type=['tar', 'csv'], key='main_file_v8')
        suffix = None
        if main_file is not None and main_file.name.lower().endswith('.tar'):
            try:
                suffix_options = sorted([s for s, entry in scan_tar_bytes(main_file.getvalue()).items() if 'RX' in entry and 'TX' in entry])
                if suffix_options:
                    suffix = st.selectbox('RX/TX pair', suffix_options, index=0, key='derate_suffix_v8')
            except Exception as e:
                st.error(f'Could not inspect TAR: {e}')
        chamber_file = st.file_uploader('Chamber CSV', type=['csv'], key='chamber_file_v8')
        title = st.text_input('Plot title', value='Derate Report', key='derate_title_v8')
        start_text = st.text_input('Manual start Pacific', value='2026-03-19 15:20', key='derate_start_v8')
        end_text = st.text_input('Manual end Pacific', value='2026-03-19 16:20', key='derate_end_v8')
    with right:
        filter_mode = st.selectbox('Chamber filter', ['Moving Average', 'Median', 'EMA', 'None'], index=0, key='derate_filter_v8')
        smooth_seconds = st.number_input('Chamber smooth seconds', min_value=1, value=10, step=1, key='derate_smooth_v8')
        ignore_seconds = st.number_input('Ignore first seconds', min_value=0.0, value=60.0, step=10.0, key='derate_ignore_v8')
        end_window_sec = st.number_input('End window seconds', min_value=30.0, value=120.0, step=30.0, key='derate_end_window_v8')
        trim_last_sec = st.number_input('Trim last seconds', min_value=0.0, value=0.0, step=10.0, key='derate_trim_v8')
        bin_step = st.number_input('Temperature bin step', min_value=0.25, value=0.5, step=0.25, key='derate_bin_v8')
    prepared_df = None
    power_choices = []
    default_idx = 0
    if main_file is not None:
        try:
            raw_df, source_type, chosen_suffix = read_source_uploaded(main_file, suffix)
            prepared_df = prepare_loaded_dataframe(raw_df)
            power_choices = [c for c in prepared_df.columns if c.lower().endswith('power') or 'efficiency' in c.lower() or c in ('RxPower', 'TxPaPower')]
            if 'RxPower' in power_choices:
                default_idx = power_choices.index('RxPower')
            _metric_cards([
                ('Source', source_type.upper(), None),
                ('Rows', f'{len(prepared_df):,}', None),
                ('Power options', f'{len(power_choices)}', None),
            ])
            st.caption(f'Loaded {main_file.name}' + (f' | pair={chosen_suffix}' if chosen_suffix else ''))
        except Exception as e:
            st.error(f'Failed to load main file: {e}')
    power_col = st.selectbox('Power signal', power_choices, index=default_idx if power_choices else None, key='derate_power_v8')
    if st.button('Generate derate report', type='primary', key='derate_generate_v8'):
        try:
            if prepared_df is None:
                raise ValueError('Upload the main TAR/CSV first.')
            if chamber_file is None:
                raise ValueError('Upload the chamber CSV first.')
            chamber_df = preprocess_chamber_csv(chamber_file)
            aligned_df = align_chamber_to_main_data(prepared_df, chamber_df, start_text, end_text, int(smooth_seconds), filter_mode)
            if 'ChamberTemp' not in aligned_df.columns:
                raise ValueError('ChamberTemp could not be aligned.')
            fig, summary_df, curve_df, window_df = generate_derate_artifacts(aligned_df, power_col, float(ignore_seconds), float(end_window_sec), float(trim_last_sec), float(bin_step), title)
            s = summary_df.iloc[0]
            _metric_cards([
                ('Avg power', f"{s['avg_power_w']:.2f} W", None),
                ('Avg chamber temp', f"{s['avg_chamber_temp_c']:.2f} °C", None),
                ('Curve points', f'{len(curve_df)}', None),
            ])
            st.pyplot(fig, clear_figure=False)
            c1, c2, c3 = st.columns(3)
            c1.dataframe(summary_df, use_container_width=True)
            c2.dataframe(curve_df, use_container_width=True, height=260)
            c3.dataframe(window_df.head(200), use_container_width=True, height=260)
            st.download_button('Download summary CSV', summary_df.to_csv(index=False).encode('utf-8'), file_name='derate_summary.csv', mime='text/csv')
            st.download_button('Download curve CSV', curve_df.to_csv(index=False).encode('utf-8'), file_name='derate_curve.csv', mime='text/csv')
            st.download_button('Download window CSV', window_df.to_csv(index=False).encode('utf-8'), file_name='derate_window.csv', mime='text/csv')
        except Exception as e:
            st.error(str(e))


def render_arduino_workspace():
    _workspace_intro('Arduino Viewer', 'Load the chamber CSV, select a day and time range, then inspect a single signal with a clean quick-look chart.')
    arduino_file = st.file_uploader('Arduino CSV', type=['csv'], key='arduino_file_v8')
    if arduino_file is None:
        st.info('Upload an Arduino Cloud CSV to use this tab.')
        return
    try:
        adf = preprocess_arduino_csv(arduino_file)
        _metric_cards([
            ('Rows', f'{len(adf):,}', None),
            ('Start', str(adf['time_pacific'].min())[:16], None),
            ('End', str(adf['time_pacific'].max())[:16], None),
        ])
        pacific_dates = sorted(adf['time_pacific'].dt.date.astype(str).unique().tolist())
        c1, c2 = st.columns([0.8, 1.2], gap='large')
        with c1:
            selected_date = st.selectbox('Pacific date', pacific_dates, key='arduino_date_v8')
            day_df = adf[adf['time_pacific'].dt.date.astype(str) == selected_date].copy()
            numeric_cols = [c for c in day_df.columns if c not in ['time', 'time_utc', 'time_pacific'] and pd.to_numeric(day_df[c], errors='coerce').notna().sum() > 0]
            y_col = st.selectbox('Signal', numeric_cols, key='arduino_signal_v8')
            mode = st.radio('Time selection mode', ['Available times from file', 'Manual HH:MM entry'], horizontal=True, key='arduino_mode_v8')
            if mode == 'Available times from file':
                times = day_df['time_pacific'].dt.strftime('%H:%M').drop_duplicates().tolist()
                start_hhmm = st.selectbox('Start time', times, index=0, key='arduino_start_v8')
                end_hhmm = st.selectbox('End time', times, index=len(times)-1, key='arduino_end_v8')
            else:
                t1, t2 = st.columns(2)
                start_hhmm = t1.text_input('Start HH:MM', value='10:15', key='arduino_start_manual_v8')
                end_hhmm = t2.text_input('End HH:MM', value='11:00', key='arduino_end_manual_v8')
        with c2:
            start_ts = parse_pacific_datetime(f'{selected_date} {start_hhmm}')
            end_ts = parse_pacific_datetime(f'{selected_date} {end_hhmm}')
            plot_df = day_df[(day_df['time_pacific'] >= start_ts) & (day_df['time_pacific'] <= end_ts)].copy()
            plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors='coerce')
            plot_df = plot_df.dropna(subset=[y_col])
            if plot_df.empty:
                st.warning('No rows in that range.')
                return
            _metric_cards([
                ('Rows in range', f'{len(plot_df):,}', None),
                ('Min', f"{plot_df[y_col].min():.2f}", None),
                ('Max', f"{plot_df[y_col].max():.2f}", None),
            ])
            fig = plt.figure(figsize=(10.5, 4.8))
            ax = fig.add_subplot(111)
            ax.plot(plot_df['time_pacific'], plot_df[y_col], linewidth=1.2)
            ax.set_title(f'{y_col} | {selected_date} | {start_hhmm} to {end_hhmm} Pacific')
            ax.set_xlabel('Pacific time')
            ax.set_ylabel(y_col)
            ax.grid(True, alpha=0.35)
            fig.autofmt_xdate()
            st.pyplot(fig, clear_figure=False)
            with st.expander('Preview data', expanded=False):
                st.dataframe(plot_df[['time_pacific', y_col]].head(500), use_container_width=True, height=260)
    except Exception as e:
        st.error(f'Failed to load Arduino CSV: {e}')


inject_branding()
render_app_header()
active_workspace = render_workspace_selector()

if active_workspace == 'Derate Reports':
    render_derate_workspace()
elif active_workspace == 'Arduino Viewer':
    render_arduino_workspace()
elif active_workspace == 'Plot Explorer':
    render_plot_tab()
elif active_workspace == 'Label Studio':
    render_label_tab()
elif active_workspace == 'SOS Inventory':
    render_sos_workspace()
else:
    render_rf_tab()

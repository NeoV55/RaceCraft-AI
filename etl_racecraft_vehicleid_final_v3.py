#!/usr/bin/env python3
"""
ETL FINAL v3 for Racecraft (vehicle_id based)
- Robust handling of smashed headers from PDF extraction
- Chunked telemetry aggregation (safe for very large telemetry files)
- Uses vehicle_id / vehicle_number; never uses car_number
- Writes combined outputs to processed/combined

Notes:
- This file is an improved, hardened version of the script you provided.
- Key fixes made:
  * Never use .get(..., '').astype(...) on presumed Series — use safe checks and fallbacks
  * Ensure lap_number and vehicle_number types match before merges
  * Coerce lap_number to numeric in both analysis and lap windows
  * Ensure vehicle_id is present (fall back to vehicle_number or synthetic id)
  * Defensive merges: coerce join keys to consistent dtypes
  * Telemetry chunk aggregation robust to very large telemetry CSVs
"""

import os
import re
import json
import traceback
from glob import glob
from datetime import datetime
from dateutil import parser as dateparser
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

# PDF libraries (optional; used if PDFs are present)
try:
    import camelot
except Exception:
    camelot = None
try:
    import pdfplumber
except Exception:
    pdfplumber = None

# ---------- CONFIG ----------
ROOT = os.path.abspath(os.path.dirname(__file__))
RAW_DIR = os.path.join(ROOT, "raw")
PROCESSED_CLEANED = os.path.join(ROOT, "processed", "cleaned")
PROCESSED_COMBINED = os.path.join(ROOT, "processed", "combined")
METADATA_DIR = os.path.join(ROOT, "metadata")
DOCS_PDF_EXTRACT = os.path.join(ROOT, "docs", "pdf_extracted_tables")
LOG_FILE = os.path.join(METADATA_DIR, "etl_log.json")

os.makedirs(PROCESSED_CLEANED, exist_ok=True)
os.makedirs(PROCESSED_COMBINED, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(DOCS_PDF_EXTRACT, exist_ok=True)

# Telemetry aggregation config
TELE_VARS = ['speed','ath','aps','pbrake_f','pbrake_r','accx_can','accy_can','steering_angle']
TELE_CHUNKSIZE = 500_000  # adjust down if memory constrained

# ---------- Utilities ----------
def detect_delimiter(sample_text: str):
    counts = {',': sample_text.count(','), ';': sample_text.count(';'), '\t': sample_text.count('\t'), '|': sample_text.count('|')}
    best = max(counts.items(), key=lambda kv: kv[1])[0]
    return best if counts[best] > 0 else ','


def safe_read_csv_with_delimiter(path):
    # Try pandas autodetect first, then try common separators, then fallback to manual splitting
    for sep in (None, ',', ';', '\t', '|'):
        try:
            kwargs = {'low_memory': False}
            if sep is not None:
                kwargs['sep'] = sep
            df = pd.read_csv(path, **kwargs)
            return df, sep
        except Exception:
            continue
    # fallback: detect from sample and attempt python engine
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
            sample = fh.read(8192)
    except Exception:
        with open(path, 'r', encoding='latin1', errors='ignore') as fh:
            sample = fh.read(8192)
    sep = detect_delimiter(sample)
    try:
        df = pd.read_csv(path, sep=sep, engine='python', low_memory=False)
        return df, sep
    except Exception:
        # last resort: treat file as one-column text and return that
        lines = [l.rstrip('\n') for l in sample.splitlines() if l.strip()]
        if not lines:
            return pd.DataFrame(), sep
        header_line = lines[0]
        return pd.DataFrame([l.split(sep) for l in lines[1:]]), sep


def sanitize_columns(df: pd.DataFrame):
    df = df.rename(columns=lambda c: re.sub(r"[^\w\s\-\.:]", "", str(c)).strip().lower().replace(" ", "_"))
    return df


def time_to_millis(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return int(val)
    s = str(val).strip()
    # pattern M:SS.mmm or MM:SS.mmm
    m = re.match(r'^(?:(\d+):)?(\d{1,2})(?:[:\.](\d{1,3}))?$', s)
    if m:
        mins = int(m.group(1)) if m.group(1) else 0
        secs = int(m.group(2))
        frac = m.group(3) or '0'
        frac = (frac + '000')[:3]
        ms = int(frac)
        return int((mins*60 + secs)*1000 + ms)
    # try parse as datetime
    try:
        dt = dateparser.parse(s)
        if dt:
            return int(dt.timestamp() * 1000)
    except Exception:
        pass
    digits = re.sub(r'[^\d]', '', s)
    if digits:
        try:
            return int(digits)
        except Exception:
            return np.nan
    return np.nan

# ---------- Fix smashed headers logic ----------
def fix_smashed_headers(df: pd.DataFrame):
    if df is None or df.shape[0] == 0:
        return df

    # If dataframe has one column but entries likely are delim-separated rows, attempt to split
    if df.shape[1] == 1:
        col0 = df.columns[0]
        sample = str(df.iloc[0, 0])
        sep = detect_delimiter(sample)
        if any(k in sample.lower() for k in ['lap', 'number', 's1', 's2', 's3', 'kph', 'driver']):
            lines = [str(x) for x in df.iloc[:,0].tolist()]
            headers = re.split(rf'[{sep}]', lines[0])
            headers = [re.sub(r'\s+', '_', h.strip()).lower() for h in headers]
            rows = []
            for line in lines[1:]:
                parts = re.split(rf'[{sep}]', line)
                if len(parts) < len(headers):
                    parts += [''] * (len(headers) - len(parts))
                rows.append(parts[:len(headers)])
            if rows:
                return pd.DataFrame(rows, columns=headers)

    # For multi-column df, check for smashed header strings in column names
    new_cols = []
    changed = False
    for col in df.columns:
        col_s = str(col)
        if any(k in col_s.lower() for k in ['lap_number', 'driver_number', 'lap_time']) and ('_' in col_s or ' ' in col_s):
            parts = [re.sub(r'\s+', '_', p.strip()).lower() for p in re.split(r'[_;,]+', col_s) if p.strip()]
            if len(parts) > 1:
                new_cols.extend(parts)
                changed = True
                continue
        new_cols.append(re.sub(r'\s+', '_', col_s).lower())
    if changed and len(new_cols) == df.shape[1]:
        df.columns = new_cols
    return df

# ---------- PDF extraction ----------
def extract_tables_pdf(pdf_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    basename = os.path.basename(pdf_path).replace('.pdf', '')
    extracted = []

    if camelot is not None:
        for flavor in ('lattice', 'stream'):
            try:
                tables = camelot.read_pdf(pdf_path, pages='all', flavor=flavor)
            except Exception:
                tables = []
            for i, t in enumerate(tables):
                try:
                    df = t.df.copy()
                    df = sanitize_columns(df)
                    out_csv = os.path.join(out_dir, f"{basename}_camelot_{flavor}_p{i+1}.csv")
                    df.to_csv(out_csv, index=False)
                    extracted.append((out_csv, df))
                except Exception:
                    continue

    if not extracted and pdfplumber is not None:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, p in enumerate(pdf.pages):
                    tbl = p.extract_table()
                    if not tbl or len(tbl) <= 1:
                        continue
                    df = pd.DataFrame(tbl[1:], columns=tbl[0])
                    df = sanitize_columns(df)
                    out_csv = os.path.join(out_dir, f"{basename}_plumber_p{i+1}.csv")
                    df.to_csv(out_csv, index=False)
                    extracted.append((out_csv, df))
        except Exception:
            pass

    return extracted

# ---------- Scan & Process ----------
def scan_and_index(raw_dir=RAW_DIR):
    index = []
    for session in sorted(os.listdir(raw_dir)):
        sp = os.path.join(raw_dir, session)
        if not os.path.isdir(sp):
            continue
        for fname in sorted(os.listdir(sp)):
            fp = os.path.join(sp, fname)
            if os.path.isdir(fp):
                for sub in glob(os.path.join(fp, '*')):
                    index.append({'session': session, 'path': sub})
            else:
                index.append({'session': session, 'path': fp})
    with open(os.path.join(METADATA_DIR, 'file_index.json'), 'w', encoding='utf8') as fh:
        json.dump(index, fh, indent=2)
    print(f"Indexed {len(index)} files.")
    return index


def ensure_header_row(df):
    if df is None or df.shape[0] == 0:
        return df
    if df.shape[1] == 1:
        col = df.columns[0]
        sample = str(df.iloc[0, 0])
        sep = detect_delimiter(sample)
        if any(k in sample.lower() for k in ['lap', 'number', 'driver', 'time']):
            lines = [str(x) for x in df.iloc[:,0].tolist()]
            headers = re.split(rf'[{sep}]', lines[0])
            headers = [re.sub(r'\s+','_',h.strip()).lower() for h in headers]
            rows = []
            for line in lines[1:]:
                parts = re.split(rf'[{sep}]', line)
                if len(parts) < len(headers):
                    parts += ['']*(len(headers)-len(parts))
                rows.append(parts[:len(headers)])
            if rows:
                return pd.DataFrame(rows, columns=headers)
    df.columns = [re.sub(r'\s+','_',str(c).strip()).lower() for c in df.columns]
    return df


def detect_and_set_core(df):
    df = df.copy()
    # vehicle_id detection
    if 'vehicle_id' not in df.columns:
        for c in df.columns:
            if re.search(r'original_vehicle_id|vehicle_id|ecm_car_id|ecm.*car', c):
                df['vehicle_id'] = df[c].astype(str)
                break
    # vehicle_number
    if 'vehicle_number' not in df.columns:
        for c in df.columns:
            if re.fullmatch(r'number|nr|no|vehicle_number|vehicle_no|pos|position', c):
                try:
                    df['vehicle_number'] = pd.to_numeric(df[c], errors='coerce')
                except Exception:
                    df['vehicle_number'] = df[c].astype(str).str.extract(r'(\d+)')
                break
    # lap_number
    if 'lap_number' not in df.columns:
        for c in df.columns:
            if 'lap_number' in c or re.fullmatch(r'lap|lapno|lap_number', c):
                df['lap_number'] = pd.to_numeric(df[c], errors='coerce')
                break
    # lap_time
    if 'lap_time' not in df.columns:
        for c in df.columns:
            if 'lap_time' in c or re.fullmatch(r'lap_time|laptime|fl_time', c):
                df['lap_time'] = df[c].astype(str)
                break
    # lap_time_ms
    if 'lap_time' in df.columns and 'lap_time_ms' not in df.columns:
        df['lap_time_ms'] = df['lap_time'].apply(lambda x: time_to_millis(x) if pd.notna(x) else np.nan)
    return df


def process_session_files(index):
    sessions = {}
    metadata = {'indexed_count': len(index), 'extracted_pdf': [], 'read_errors': []}
    for it in tqdm(index, desc="Loading files"):
        session = it['session']
        path = it['path']
        sessions.setdefault(session, {})
        try:
            if path.lower().endswith('.pdf'):
                extracted = extract_tables_pdf(path, os.path.join(DOCS_PDF_EXTRACT, session))
                for fpath, df in extracted:
                    df = ensure_header_row(df)
                    df = fix_smashed_headers(df)
                    df = sanitize_columns(df)
                    df = detect_and_set_core(df)
                    sessions[session].setdefault('pdf_extracted', []).append((fpath, df))
                    metadata['extracted_pdf'].append(fpath)
                continue
            df, sep = safe_read_csv_with_delimiter(path)
            if df is None:
                metadata['read_errors'].append(path)
                continue
            df = ensure_header_row(df)
            df = fix_smashed_headers(df)
            df = sanitize_columns(df)
            df = detect_and_set_core(df)
            if 'lap_number' not in df.columns:
                for c in df.columns:
                    if 'lap_number' in c:
                        df['lap_number'] = pd.to_numeric(df[c], errors='coerce')
                        break
            out_dir = os.path.join(PROCESSED_CLEANED, session)
            os.makedirs(out_dir, exist_ok=True)
            cleaned_path = os.path.join(out_dir, os.path.basename(path))
            try:
                df.to_csv(cleaned_path, index=False)
            except Exception:
                df.to_csv(cleaned_path, index=False, encoding='utf-8')
            lower = os.path.basename(path).lower()
            key = 'csv_unknown'
            if 'analysis' in lower or 'analysisendurance' in lower or '23_' in lower:
                key = 'analysis_sections'
            elif 'telemetry' in lower or 'sonoma_telemetry' in lower:
                key = 'telemetry'
            elif 'lap_start' in lower or 'lap_start_time' in lower or 'sonoma_lap_start' in lower:
                key = 'lap_start'
            elif 'lap_end' in lower or 'sonoma_lap_end' in lower:
                key = 'lap_end'
            elif 'lap_time' in lower or 'sonoma_lap_time' in lower:
                key = 'lap_time'
            elif 'results_by_class' in lower or '05_' in lower:
                key = 'results_by_class'
            elif 'results' in lower or '_results' in lower or '03_' in lower:
                key = 'results'
            elif 'best 10' in lower or '99_' in lower:
                key = 'best_10_laps'
            elif 'weather' in lower or '26_' in lower:
                key = 'weather'
            sessions[session].setdefault(key, []).append((cleaned_path, df))
        except Exception as e:
            metadata.setdefault('exceptions', []).append({'path': path, 'error': str(e), 'trace': traceback.format_exc()})
    with open(LOG_FILE, 'w', encoding='utf8') as fh:
        json.dump(metadata, fh, indent=2, default=str)
    return sessions, metadata

# ---------- Synthetic lap windows ----------
def compute_synthetic_lap_windows(analysis_df, tele_df):
    if analysis_df is None or analysis_df.empty:
        return pd.DataFrame()

    adf = analysis_df.copy()
    adf = fix_smashed_headers(adf)
    adf = detect_and_set_core(adf)

    # Standardize keys
    if 'vehicle_id' in adf.columns:
        adf['vehicle_id'] = adf['vehicle_id'].astype(str)
    elif 'vehicle_number' in adf.columns:
        # create a synthetic vehicle_id string based on number
        adf['vehicle_id'] = adf['vehicle_number'].astype(str)
    else:
        adf['vehicle_id'] = adf.index.astype(str)

    if 'lap_number' not in adf.columns:
        # try to locate any col containing 'lap'
        for c in adf.columns:
            if 'lap' in c and c != 'lap_number':
                adf['lap_number'] = pd.to_numeric(adf[c], errors='coerce')
                break
    adf['lap_number'] = pd.to_numeric(adf.get('lap_number', pd.Series([np.nan]*len(adf))), errors='coerce')

    # sort safely
    sort_keys = []
    if 'vehicle_id' in adf.columns:
        sort_keys.append('vehicle_id')
    if 'lap_number' in adf.columns:
        sort_keys.append('lap_number')
    if not sort_keys:
        adf['__dummy__'] = range(len(adf))
        sort_keys = ['__dummy__']
    adf = adf.sort_values(sort_keys, na_position='last').reset_index(drop=True)

    # compute hour_ms if hour exists
    if 'hour' in adf.columns:
        adf['hour_ms'] = adf['hour'].apply(lambda x: time_to_millis(x) if pd.notna(x) else np.nan)

    # prepare telemetry timestamps if available
    tele = None
    if tele_df is not None and not tele_df.empty:
        tele = tele_df.copy()
        tele = sanitize_columns(tele)
        if 'timestamp' in tele.columns:
            tele['ts_ms'] = tele['timestamp'].apply(lambda x: time_to_millis(x) if pd.notna(x) else np.nan)
        elif 'meta_time' in tele.columns:
            tele['ts_ms'] = tele['meta_time'].apply(lambda x: time_to_millis(x) if pd.notna(x) else np.nan)
        if 'vehicle_id' not in tele.columns and 'original_vehicle_id' in tele.columns:
            tele['vehicle_id'] = tele['original_vehicle_id'].astype(str)
        if 'vehicle_id' in tele.columns:
            tele['vehicle_id'] = tele['vehicle_id'].astype(str)

    rows = []
    group_key = 'vehicle_id' if 'vehicle_id' in adf.columns else ('vehicle_number' if 'vehicle_number' in adf.columns else None)
    if group_key is None:
        adf['__group__'] = 1
        group_key = '__group__'

    for key, g in adf.groupby(group_key, dropna=True):
        g = g.sort_values(['lap_number'] if 'lap_number' in g.columns else [], na_position='last').reset_index(drop=True)
        veh_num = int(g['vehicle_number'].dropna().iloc[0]) if 'vehicle_number' in g.columns and not g['vehicle_number'].dropna().empty else None
        veh_id = str(g['vehicle_id'].dropna().iloc[0]) if 'vehicle_id' in g.columns and not g['vehicle_id'].dropna().empty else (str(veh_num) if veh_num is not None else str(key))

        if 'hour_ms' in g.columns and g['hour_ms'].notnull().any() and tele is not None and not tele.empty:
            if veh_id is not None and 'vehicle_id' in tele.columns:
                tele_v = tele[tele['vehicle_id'] == veh_id]
            elif veh_num is not None and 'vehicle_number' in tele.columns:
                tele_v = tele[tele['vehicle_number'] == veh_num]
            else:
                tele_v = tele
            tele_med = tele_v['ts_ms'].median() if not tele_v.empty else np.nan
            adf_med = g['hour_ms'].median() if not g['hour_ms'].dropna().empty else np.nan
            offset = tele_med - adf_med if not (np.isnan(tele_med) or np.isnan(adf_med)) else 0
            for _, r in g.iterrows():
                end_ms = r.get('hour_ms')
                if pd.isna(end_ms):
                    continue
                end_aligned = int(end_ms + (offset if not pd.isna(offset) else 0))
                lap_ms = r.get('lap_time_ms') if 'lap_time_ms' in r else None
                if lap_ms is None or pd.isna(lap_ms):
                    start_aligned = end_aligned - 1000
                else:
                    start_aligned = end_aligned - int(lap_ms)
                rows.append({'vehicle_id': veh_id, 'vehicle_number': veh_num, 'lap_number': int(r.get('lap_number')) if not pd.isna(r.get('lap_number')) else None, 'lap_start_ms': int(start_aligned), 'lap_end_ms': int(end_aligned)})
        else:
            cum = 0
            for _, r in g.iterrows():
                lm = int(r['lap_time_ms']) if 'lap_time_ms' in r and not pd.isna(r['lap_time_ms']) else 1000
                start_ms = cum
                end_ms = cum + lm
                rows.append({'vehicle_id': veh_id, 'vehicle_number': veh_num, 'lap_number': int(r.get('lap_number')) if not pd.isna(r.get('lap_number')) else None, 'lap_start_ms': int(start_ms), 'lap_end_ms': int(end_ms)})
                cum = end_ms

    lw = pd.DataFrame(rows)
    # ensure dtypes
    if not lw.empty:
        lw['lap_number'] = pd.to_numeric(lw['lap_number'], errors='coerce')
        lw['vehicle_id'] = lw['vehicle_id'].astype(str)
    return lw

# ---------- Chunked telemetry aggregator ----------
def aggregate_telemetry_chunked(tele_csv_path, lap_windows_df, tele_vars=TELE_VARS, chunksize=TELE_CHUNKSIZE):
    if lap_windows_df is None or lap_windows_df.empty:
        return pd.DataFrame()

    lw = lap_windows_df.copy()
    lw['vehicle_id'] = lw['vehicle_id'].astype(str)
    lw['lap_number'] = pd.to_numeric(lw['lap_number'], errors='coerce')
    vehicle_set = set(lw['vehicle_id'].dropna().unique())

    windows_by_vehicle = defaultdict(list)
    for _, r in lw.iterrows():
        if pd.isna(r['lap_number']):
            continue
        windows_by_vehicle[str(r['vehicle_id'])].append((int(r['lap_number']), int(r['lap_start_ms']), int(r['lap_end_ms'])))

    accum = {}
    def make_key(vid, lap, var): return (str(vid), int(lap), var)

    total_rows = 0
    for chunk in pd.read_csv(tele_csv_path, chunksize=chunksize, low_memory=False):
        total_rows += len(chunk)
        chunk = sanitize_columns(chunk)
        if 'vehicle_id' not in chunk.columns and 'original_vehicle_id' in chunk.columns:
            chunk['vehicle_id'] = chunk['original_vehicle_id'].astype(str)
        if 'timestamp' in chunk.columns:
            chunk['ts_ms'] = chunk['timestamp'].apply(lambda x: time_to_millis(x) if pd.notna(x) else np.nan)
        elif 'meta_time' in chunk.columns:
            chunk['ts_ms'] = chunk['meta_time'].apply(lambda x: time_to_millis(x) if pd.notna(x) else np.nan)
        else:
            chunk['ts_ms'] = np.nan
        chunk['vehicle_id'] = chunk['vehicle_id'].astype(str)
        chunk = chunk[chunk['vehicle_id'].isin(vehicle_set)]
        if chunk.empty:
            continue
        for v in tele_vars:
            if v in chunk.columns:
                chunk[v] = pd.to_numeric(chunk[v], errors='coerce')
        for vid in chunk['vehicle_id'].unique():
            sub = chunk[chunk['vehicle_id'] == vid]
            if sub.empty:
                continue
            windows = windows_by_vehicle.get(vid, [])
            if not windows:
                continue
            for lapnum, start, end in windows:
                sel = sub[(sub['ts_ms'] >= start) & (sub['ts_ms'] <= end)]
                if sel.empty:
                    sel = sub[(sub['ts_ms'] >= (start - 2000)) & (sub['ts_ms'] <= (end + 2000))]
                if sel.empty:
                    continue
                for v in tele_vars:
                    if v in sel.columns and not sel[v].dropna().empty:
                        arr = sel[v].dropna().astype(float)
                        key = make_key(vid, lapnum, v)
                        if key not in accum:
                            accum[key] = {'count': 0, 'sum': 0.0, 'sum_sq': 0.0, 'max': -np.inf}
                        a = accum[key]
                        cnt = int(arr.size)
                        a['count'] += cnt
                        a['sum'] += float(arr.sum())
                        a['sum_sq'] += float((arr**2).sum())
                        a['max'] = max(a['max'], float(arr.max()))
    if not accum:
        return pd.DataFrame()
    rows = []
    keys_by_pair = defaultdict(list)
    for (vid, lap, var), stats in accum.items():
        keys_by_pair[(vid, lap)].append((var, stats))
    for (vid, lap), varstats in keys_by_pair.items():
        rec = {'vehicle_id': vid, 'lap_number': int(lap)}
        total_samples = 0
        for var, st in varstats:
            cnt = st['count']
            mean = (st['sum'] / cnt) if cnt > 0 else np.nan
            varval = (st['sum_sq'] / cnt - mean*mean) if cnt > 0 else np.nan
            std = np.sqrt(varval) if cnt > 0 else np.nan
            mx = st['max'] if st['max'] != -np.inf else np.nan
            rec[f"{var}_mean"] = mean
            rec[f"{var}_std"] = std
            rec[f"{var}_max"] = mx
            total_samples += cnt
        rec['samples'] = total_samples
        rows.append(rec)
    return pd.DataFrame(rows)

# ---------- Master tables builder ----------
def build_master_tables(sessions, metadata):
    master_laps = []
    summary_profiles = []
    optimal_rows = []
    session_summaries = []

    for session, cats in sessions.items():
        print(f"\nBuilding session: {session}")
        analysis_entries = cats.get('analysis_sections', []) or cats.get('analysis_sections'.lower(), [])
        if not analysis_entries:
            print(f"  - No analysis sections for {session}; skipping.")
            continue

        analysis_path, analysis_df = analysis_entries[0]
        analysis_df = ensure_header_row(analysis_df)
        analysis_df = fix_smashed_headers(analysis_df)
        analysis_df = sanitize_columns(analysis_df)
        analysis_df = detect_and_set_core(analysis_df)

        if 'lap_number' in analysis_df.columns:
            analysis_df['lap_number'] = pd.to_numeric(analysis_df['lap_number'], errors='coerce')
        if 'vehicle_number' in analysis_df.columns:
            analysis_df['vehicle_number'] = pd.to_numeric(analysis_df['vehicle_number'], errors='coerce')
        if 'lap_time' in analysis_df.columns and 'lap_time_ms' not in analysis_df.columns:
            analysis_df['lap_time_ms'] = analysis_df['lap_time'].apply(lambda x: time_to_millis(x) if pd.notna(x) else np.nan)

        out_dir = os.path.join(PROCESSED_CLEANED, session)
        os.makedirs(out_dir, exist_ok=True)
        cleaned_path = os.path.join(out_dir, os.path.basename(analysis_path))
        analysis_df.to_csv(cleaned_path, index=False)

        tele_entries = cats.get('telemetry', [])
        tele_csv_path = None
        tele_df_sample = None
        if tele_entries:
            tele_csv_path, tele_df_sample = tele_entries[0]
            tele_df_sample = ensure_header_row(tele_df_sample)
            tele_df_sample = sanitize_columns(tele_df_sample)
            tele_df_sample = detect_and_set_core(tele_df_sample)

        lap_windows = compute_synthetic_lap_windows(analysis_df, tele_df_sample)

        merged = analysis_df.copy()
        if not lap_windows.empty:
            # align types for merge keys
            if 'vehicle_number' in lap_windows.columns and 'vehicle_number' in merged.columns:
                merged['vehicle_number'] = pd.to_numeric(merged['vehicle_number'], errors='coerce')
                lap_windows['vehicle_number'] = pd.to_numeric(lap_windows['vehicle_number'], errors='coerce')
            if 'vehicle_id' in lap_windows.columns and 'vehicle_id' in merged.columns:
                merged['vehicle_id'] = merged['vehicle_id'].astype(str)
                lap_windows['vehicle_id'] = lap_windows['vehicle_id'].astype(str)
            # lap_number numeric
            if 'lap_number' in merged.columns:
                merged['lap_number'] = pd.to_numeric(merged['lap_number'], errors='coerce')
            if 'lap_number' in lap_windows.columns:
                lap_windows['lap_number'] = pd.to_numeric(lap_windows['lap_number'], errors='coerce')

            # prioritized merges
            if 'vehicle_number' in merged.columns and 'vehicle_number' in lap_windows.columns and 'lap_number' in merged.columns:
                merged = pd.merge(merged, lap_windows, on=['vehicle_number', 'lap_number'], how='left')
            elif 'vehicle_id' in merged.columns and 'vehicle_id' in lap_windows.columns and 'lap_number' in merged.columns:
                merged = pd.merge(merged, lap_windows, on=['vehicle_id', 'lap_number'], how='left')
            else:
                # fallback safe conversion and merge
                if 'vehicle_id' in lap_windows.columns and 'vehicle_id' not in merged.columns:
                    merged['vehicle_id'] = merged.get('vehicle_id', merged.get('vehicle_number', merged.index)).astype(str)
                if 'vehicle_id' in merged.columns:
                    merged['vehicle_id'] = merged['vehicle_id'].astype(str)
                    lap_windows['vehicle_id'] = lap_windows['vehicle_id'].astype(str)
                    merged['lap_number'] = pd.to_numeric(merged.get('lap_number', pd.Series([np.nan]*len(merged))), errors='coerce')
                    lap_windows['lap_number'] = pd.to_numeric(lap_windows.get('lap_number', pd.Series([np.nan]*len(lap_windows))), errors='coerce')
                    merged = pd.merge(merged, lap_windows, on=['vehicle_id', 'lap_number'], how='left')
                else:
                    merged = pd.merge(merged, lap_windows, on=['lap_number'], how='left')

        telemetry_agg = pd.DataFrame()
        if tele_csv_path and os.path.exists(tele_csv_path) and not lap_windows.empty:
            print(f"  - Aggregating telemetry (chunked) from: {tele_csv_path}")
            telemetry_agg = aggregate_telemetry_chunked(tele_csv_path, lap_windows, tele_vars=TELE_VARS, chunksize=TELE_CHUNKSIZE)
            if not telemetry_agg.empty:
                telemetry_agg['lap_number'] = pd.to_numeric(telemetry_agg['lap_number'], errors='coerce')
                telemetry_agg['vehicle_id'] = telemetry_agg['vehicle_id'].astype(str)
                if 'vehicle_id' in merged.columns:
                    merged['vehicle_id'] = merged['vehicle_id'].astype(str)
                    merged = pd.merge(merged, telemetry_agg, on=['vehicle_id', 'lap_number'], how='left')
                else:
                    merged = pd.merge(merged, telemetry_agg, on=['lap_number'], how='left')

        weather_entries = cats.get('weather', [])
        if weather_entries:
            wpath, wdf = weather_entries[0]
            wdf = ensure_header_row(wdf)
            wdf = sanitize_columns(wdf)
            try:
                merged['weather_air_temp_mean'] = float(wdf['air_temp'].dropna().astype(float).mean()) if 'air_temp' in wdf.columns else np.nan
            except:
                merged['weather_air_temp_mean'] = np.nan
            try:
                merged['weather_humidity_mean'] = float(wdf['humidity'].dropna().astype(float).mean()) if 'humidity' in wdf.columns else np.nan
            except:
                merged['weather_humidity_mean'] = np.nan

        merged['session'] = session
        master_laps.append(merged)

        group_col = None
        for cand in ['vehicle_id', 'vehicle_number', 'number', 'driver_number']:
            if cand in merged.columns:
                group_col = cand
                break
        if group_col:
            gb = merged.groupby(group_col, dropna=True)
            for veh, g in gb:
                best = g['lap_time_ms'].min() if 'lap_time_ms' in g.columns else np.nan
                avg = g['lap_time_ms'].mean() if 'lap_time_ms' in g.columns else np.nan
                std = g['lap_time_ms'].std() if 'lap_time_ms' in g.columns else np.nan
                summary_profiles.append({
                    'session': session,
                    'vehicle_id': str(veh) if group_col == 'vehicle_id' else None,
                    'vehicle_number': int(veh) if group_col == 'vehicle_number' and not pd.isna(veh) else (int(g['vehicle_number'].dropna().iloc[0]) if 'vehicle_number' in g.columns and not g['vehicle_number'].dropna().empty else None),
                    'avg_lap_time_ms': float(avg) if not pd.isna(avg) else np.nan,
                    'best_lap_time_ms': float(best) if not pd.isna(best) else np.nan,
                    'consistency_ms': float(std) if not pd.isna(std) else np.nan,
                    'n_laps': int(g.shape[0])
                })
        else:
            print(f"  [WARN] session {session} has no vehicle id/number for grouping; skipping summary for this session")

        b10_entries = cats.get('best_10_laps', [])
        if b10_entries:
            b10p, b10df = b10_entries[0]
            b10df = ensure_header_row(b10df)
            b10df = fix_smashed_headers(b10df)
            b10df = sanitize_columns(b10df)
            lapcols = [c for c in b10df.columns if c.startswith('bestlap') or 'bestlap' in c or c.startswith('best_')]
            for _, brow in b10df.iterrows():
                vnum = brow.get('vehicle_number')
                vid = brow.get('vehicle_id') if 'vehicle_id' in brow else None
                bests = []
                for c in lapcols:
                    val = brow.get(c)
                    if pd.notna(val):
                        ms = time_to_millis(val)
                        if not pd.isna(ms):
                            bests.append(ms)
                optimal_rows.append({
                    'session': session,
                    'vehicle_id': str(vid) if vid is not None else None,
                    'vehicle_number': int(vnum) if not pd.isna(vnum) else None,
                    'best10_avg_ms': float(np.nanmean(bests)) if bests else np.nan,
                    'best10_count': len(bests)
                })

        session_summaries.append({'session': session, 'rows': int(merged.shape[0]), 'vehicles': int(merged['vehicle_number'].nunique()) if 'vehicle_number' in merged.columns else int(merged['vehicle_id'].nunique() if 'vehicle_id' in merged.columns else 0)})

    driver_lap_data = pd.concat(master_laps, ignore_index=True, sort=False) if master_laps else pd.DataFrame()
    driver_summary = pd.DataFrame(summary_profiles) if summary_profiles else pd.DataFrame()
    optimal_model = pd.DataFrame(optimal_rows) if optimal_rows else pd.DataFrame()

    os.makedirs(PROCESSED_COMBINED, exist_ok=True)
    driver_lap_data.to_csv(os.path.join(PROCESSED_COMBINED, 'driver_lap_data.csv'), index=False)
    driver_summary.to_csv(os.path.join(PROCESSED_COMBINED, 'driver_summary_profile.csv'), index=False)
    optimal_model.to_csv(os.path.join(PROCESSED_COMBINED, 'optimal_model_data.csv'), index=False)
    with open(os.path.join(METADATA_DIR, 'session_summary.json'), 'w', encoding='utf8') as fh:
        json.dump(session_summaries, fh, indent=2, default=str)

    print(f"\nWrote outputs to {PROCESSED_COMBINED}")
    return driver_lap_data, driver_summary, optimal_model

# ---------- Main ----------
def main():
    print("Scanning raw directory...")
    idx = scan_and_index(RAW_DIR)
    print("Processing files and extracting tables...")
    sessions, metadata = process_session_files(idx)
    print("Building master tables (with chunked telemetry aggregation)...")
    dl, ds, om = build_master_tables(sessions, metadata)
    print("ETL complete.")
    print("Rows in driver_lap_data:", len(dl))
    print("Driver summary count:", len(ds))
    print("Optimal model rows:", len(om))
    metadata.update({'driver_lap_rows': len(dl), 'driver_summary_rows': len(ds), 'optimal_rows': len(om)})
    with open(LOG_FILE, 'w', encoding='utf8') as fh:
        json.dump(metadata, fh, indent=2, default=str)

if __name__ == '__main__':
    main()

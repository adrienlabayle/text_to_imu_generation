from __future__ import annotations
import re
from typing import Optional, Literal, List, Tuple, Dict
import numpy as np
import pandas as pd
from pathlib import Path
import json


# Optional: requires scipy in requirements.txt
try:
    from scipy.signal import decimate
except Exception:
    decimate = None

def resample_df(
    df: pd.DataFrame, target_cols: List[str], factor: int = 2
) -> pd.DataFrame:
    """
    FIR low-pass + decimation downsample (e.g., 100 Hz → 50 Hz with factor=2).
    Assumes df is (roughly) uniformly sampled and target_cols are numeric.

    Keeps non-target columns by simple stride (iloc[::factor]) which is fine
    when labels/timestamps align with the decimated signal.
    """
    if decimate is None:
        raise ImportError("scipy is required: pip install scipy")

    # Downsample timestamp/labels/etc. by striding
    base = df.iloc[::factor].reset_index(drop=True)

    # Replace sensor columns with filtered+decimated versions
    for col in target_cols:
        base[col] = decimate(
            df[col].to_numpy(), q=factor, ftype="fir", zero_phase=True
        )
    return base

def zscore_normalize(arr: np.ndarray) -> np.ndarray:
    """Z-score normalization across columns (features)."""
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std = np.where(std == 0, 1.0, std)
    return (arr - mean) / std

def convert_unit(
    arr: np.ndarray, kind: Optional[Literal["acc", "gyro"]] = None
) -> np.ndarray:
    """
    Convert IMU units:
    - 'acc': g → m/s² (× 9.80665)
    - 'gyro': deg/s → rad/s
    """
    if kind == "acc":
        return arr * 9.80665
    if kind == "gyro":
        return arr * (np.pi / 180.0)
    return arr

def _canon(s):
    """Normalize any string to lowercase snake_case alphanumerics. THE canonical normalizer."""
    if pd.isna(s): return ""
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


# ===== Shared pipeline boilerplate (Step 0 / 3 / 4) =====

def load_contracts(root: Path) -> dict:
    """
    Load schema + activity mapping from Unification/schemas/.
    Returns dict with keys: SCHEMA, ACT_MAP_FULL, UNKNOWN_ID, TARGET_HZ,
                             RAW2ID, ID2NAME, CLEANED, MERGED.
    """
    schema_path  = root / "Unification" / "schemas" / "continuous_stream_schema.json"
    act_map_path = root / "Unification" / "schemas" / "activity_mapping.json"

    schema  = json.loads(schema_path.read_text())
    act_map = json.loads(act_map_path.read_text())

    unknown_id = int(act_map.get("unknown_activity_id", 9000))
    target_hz  = int(schema.get("rate_hz", 50))
    raw2id  = {_canon(k): int(v) for k, v in act_map.get("mapping", {}).items()}
    id2name = {int(x["id"]): x["name"] for x in act_map["label_set"]}

    return {
        "SCHEMA":       schema,
        "ACT_MAP_FULL": act_map,
        "UNKNOWN_ID":   unknown_id,
        "TARGET_HZ":    target_hz,
        "RAW2ID":       raw2id,
        "ID2NAME":      id2name,
        "CLEANED":      root / "data" / "cleaned_premerge",
        "MERGED":       root / "data" / "merged_dataset",
    }


def to_continuous_stream(
    df_native: pd.DataFrame,
    schema: dict,
    raw2id: dict,
    id2name: dict,
    unknown_id: int,
) -> pd.DataFrame:
    """
    Generic Step 3: map native labels → global, enforce dtypes, reorder to schema.
    Expects df_native to have: dataset, subject_id, session_id, timestamp_ns,
    acc_x/y/z, optionally gyro_x/y/z, dataset_activity_id, dataset_activity_label.
    """
    if df_native.empty:
        return pd.DataFrame(columns=[c["name"] for c in schema["columns"]])

    raw_key = df_native["dataset_activity_label"].astype("string").map(_canon)
    gid     = raw_key.map(raw2id).fillna(unknown_id).astype("int16")
    glabel  = gid.map(lambda x: id2name.get(int(x), "other")).astype("string")

    has_gyro = "gyro_x" in df_native.columns

    out = pd.DataFrame({
        "dataset":                df_native["dataset"].astype("string"),
        "subject_id":             df_native["subject_id"].astype("string"),
        "session_id":             df_native["session_id"].astype("string"),
        "timestamp_ns":           df_native["timestamp_ns"].astype("int64"),
        "acc_x":                  df_native["acc_x"].astype("float32"),
        "acc_y":                  df_native["acc_y"].astype("float32"),
        "acc_z":                  df_native["acc_z"].astype("float32"),
        "gyro_x":                 df_native["gyro_x"].astype("float32") if has_gyro else np.float32(np.nan),
        "gyro_y":                 df_native["gyro_y"].astype("float32") if has_gyro else np.float32(np.nan),
        "gyro_z":                 df_native["gyro_z"].astype("float32") if has_gyro else np.float32(np.nan),
        "global_activity_id":     gid,
        "global_activity_label":  glabel,
        "dataset_activity_id":    df_native["dataset_activity_id"].astype("int16"),
        "dataset_activity_label": df_native["dataset_activity_label"].astype("string"),
    })

    order = [c["name"] for c in schema["columns"]]
    return out[order]


def est_hz_ns(ts_ns: pd.Series) -> float:
    """Estimate sampling rate from nanosecond timestamps."""
    arr = ts_ns.to_numpy()
    if arr.size < 3:
        return np.nan
    dt = np.diff(arr) / 1e9
    dt = dt[(dt > 0) & np.isfinite(dt)]
    return float(np.median(1.0 / dt)) if dt.size else np.nan


def run_qa_checks(df: pd.DataFrame, schema: dict, unknown_id: int) -> None:
    """Standard Step 4 QA: monotonicity, Hz, not-null, mapping coverage, label dist."""
    groups = ["subject_id", "session_id"]

    print("Subjects:", df["subject_id"].nunique(),
          "| Sessions:", df["session_id"].nunique())

    # Monotonic timestamps per group
    viol = 0
    for _, g in df.groupby(groups, sort=False):
        ts = g["timestamp_ns"].to_numpy()
        if ts.size and not np.all(np.diff(ts) >= 0):
            viol += 1
    print("Monotonic violations (groups):", viol)

    # Hz estimation
    hz = df.groupby(groups)["timestamp_ns"].apply(est_hz_ns)
    print(f"Median Hz: {np.nanmedian(hz.values):.2f} (target={schema['rate_hz']})")

    # Required-not-null
    req = schema["expectations"]["required_not_null"]
    pct = df[req].notnull().all(axis=1).mean() * 100
    print(f"Rows meeting required-not-null: {pct:.2f}%")

    # Global mapping coverage
    cov = (df["global_activity_id"] != unknown_id).mean() * 100
    print(f"Global mapping coverage: {cov:.1f}% (unknown={unknown_id})")

    print("\nTop-15 dataset_activity_label:")
    print(df["dataset_activity_label"].value_counts().head(15))
    print("\nTop-15 global labels:")
    print(df["global_activity_label"].value_counts().head(15))

def check_sample_integrity(
    df: pd.DataFrame,
    schema: dict,
    min_samples: int = 128,
) -> None:
    """
    Step 4b: verify that every contiguous run of a single
    (subject_id, session_id, dataset_activity_label) has:
      1. Monotonically non-decreasing timestamps
      2. Consistent step size (~20 ms for 50 Hz)
      3. At least `min_samples` consecutive rows

    Prints a summary and flags violations.
    """
    target_hz = int(schema.get("rate_hz", 50))
    expected_dt_ns = int(1e9 // target_hz)          # 20_000_000
    tol_ns = expected_dt_ns // 2                     # 10 ms tolerance

    groups = ["subject_id", "session_id"]
    total_runs = 0
    short_runs = 0
    mono_violations = 0
    dt_violations = 0
    short_details: list[tuple] = []                  # (subj, sess, label, run_len)

    for (sid, sess), g in df.groupby(groups, sort=False):
        labels = g["dataset_activity_label"].to_numpy(dtype=str)
        ts = g["timestamp_ns"].to_numpy(dtype=np.int64)

        # detect contiguous run boundaries (label changes)
        change = np.concatenate(([True], labels[1:] != labels[:-1]))
        run_ids = np.cumsum(change)

        for rid in range(1, run_ids[-1] + 1):
            mask = run_ids == rid
            run_ts = ts[mask]
            run_label = labels[mask][0]
            run_len = int(mask.sum())
            total_runs += 1

            # 1. monotonic
            diffs = np.diff(run_ts)
            if diffs.size and not np.all(diffs >= 0):
                mono_violations += 1

            # 2. consistent dt (only check positive diffs)
            pos = diffs[diffs > 0]
            if pos.size:
                med_dt = int(np.median(pos))
                if abs(med_dt - expected_dt_ns) > tol_ns:
                    dt_violations += 1

            # 3. min length
            if run_len < min_samples:
                short_runs += 1
                if len(short_details) < 20:          # cap detail output
                    short_details.append((sid, sess, run_label, run_len))

    print(f"\n=== Sample Integrity Check (min_samples={min_samples}) ===")
    print(f"Total contiguous runs: {total_runs:,}")
    print(f"Monotonic violations:  {mono_violations}")
    print(f"dt-step violations:    {dt_violations}  (expected {expected_dt_ns/1e6:.0f} ms ± {tol_ns/1e6:.0f} ms)")
    print(f"Short runs (<{min_samples} rows): {short_runs}  ({100*short_runs/max(total_runs,1):.1f}%)")
    if short_details:
        print(f"\nFirst {len(short_details)} short runs:")
        for sid, sess, lbl, n in short_details:
            print(f"  {sid} | {sess} | {lbl:30s} | {n} rows")


def upsample_df_rate(df: pd.DataFrame, tcol: str, num_cols, src_hz: float, dst_hz: int) -> pd.DataFrame:
    """
    Resample to dst_hz using seconds, phase-locked to session start, and
    robust interpolation:
      - If >=2 points: linear interp with edge holding (np.interp default behavior).
      - If ==1 point: hold that single value across the grid.
      - If ==0 points: NaN (should not occur for acc_* here).
    """
    if df.empty:
        out = pd.DataFrame({tcol: np.array([], dtype=np.float64)})
        for c in num_cols: out[c] = np.nan
        return out

    # time axis in seconds (float64)
    t_src_full = pd.to_numeric(df[tcol], errors="coerce").to_numpy(dtype=np.float64)
    m_t = np.isfinite(t_src_full)
    if m_t.sum() == 0:
        out = pd.DataFrame({tcol: np.array([], dtype=np.float64)})
        for c in num_cols: out[c] = np.nan
        return out

    t_src = t_src_full[m_t]
    STEP_S = 1.0 / float(dst_hz)

    # phase-locked grid: round start to nearest 0.02s tick, floor end
    t0_round = np.round(t_src.min() / STEP_S) * STEP_S
    t1       = t_src.max()
    n_ticks  = int(np.floor((t1 - t0_round) / STEP_S)) + 1
    if n_ticks < 1:
        n_ticks = 1
    ticks = np.arange(n_ticks, dtype=np.int64)
    t_new = t0_round + ticks * STEP_S

    out = pd.DataFrame({tcol: t_new})

    # per-column robust interpolation
    for c in num_cols:
        v_full = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
        m = m_t & np.isfinite(v_full)
        if m.sum() >= 2:
            # np.interp already holds edges; no NaNs
            out[c] = np.interp(t_new, t_src_full[m], v_full[m]).astype(np.float32)
        elif m.sum() == 1:
            # single finite point: hold constant
            val = float(v_full[m][0])
            out[c] = np.full(t_new.shape, val, dtype=np.float32)
        else:
            out[c] = np.nan  # truly missing channel
    return out


def read_opp_column_names(col_path: Path) -> List[str]:
    """
    'Column: 1 MILLISEC; ...' -> ['1 MILLISEC', '2 <NAME>', ...]
    Keep the numeric prefix; we'll strip it during canonicalization.
    """
    names = []
    for ln in col_path.read_text().splitlines():
        ln = ln.strip()
        if not ln or "Column:" not in ln:
            continue
        lhs = ln.split(";")[0].strip()
        names.append(lhs.replace("Column: ", ""))
    return names

_axis_fix = re.compile(r"(acc|gyro|magnetic)([xyz])$", re.IGNORECASE)

def canonicalize_opp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Strip leading '<idx> ' prefix
    - Remove verbose prefixes
    - Lowercase + underscores
    - Ensure 'accx' -> 'acc_x' etc
    """
    df = df.copy()
    df.columns = [re.sub(r"^\d+\s+", "", c) for c in df.columns]
    df.columns = [
        c.replace("InertialMeasurementUnit ", "")
         .replace("Accelerometer ", "")
         .replace("_ ", " ")
        for c in df.columns
    ]
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df.columns = [_axis_fix.sub(r"\1_\2", c) for c in df.columns]
    return df

def parse_opp_subject_session(stem: str) -> Tuple[str, str]:
    """
    Accepts: S1-ADL1, s2_adl3, S3-DRILL, S1-Drill, 1-ADL1, etc.
    Returns ('S<id>', session_upper)
    """
    m = re.match(r"^[sS]?(\d+)[-_](ADL\d+|DRILL)$", stem, flags=re.IGNORECASE)
    if not m:
        # last-chance soft parse: split on - or _ and pick tokens
        parts = re.split(r"[-_]", stem)
        if len(parts) >= 2 and parts[0].isdigit():
            subj, sess = parts[0], parts[1]
        elif len(parts) >= 2 and parts[0].lower().startswith("s") and parts[0][1:].isdigit():
            subj, sess = parts[0][1:], parts[1]
        else:
            raise ValueError(f"Unexpected Opportunity++ filename format: {stem}")
        return f"S{subj}", sess.upper()
    subj, sess = m.groups()
    return f"S{subj}", sess.upper()


def nearest_label_join_1d(
    src_ts_ns: np.ndarray,
    src_label_df: pd.DataFrame,
    target_ts_ns: np.ndarray,
    half_frame_ns: int,
) -> pd.DataFrame:
    """
    Align labels from (src_ts_ns, src_label_df) to target_ts_ns by nearest neighbor
    within a half-frame tolerance. If outside tolerance, forward-fill.
    """
    # Precondition
    order = np.argsort(src_ts_ns)
    src_ts_ns = src_ts_ns[order]
    src = src_label_df.iloc[order].reset_index(drop=True)

    # nearest neighbor
    idx = np.searchsorted(src_ts_ns, target_ts_ns, side="left")
    idx = np.clip(idx, 0, len(src_ts_ns) - 1)

    left_idx = np.maximum(idx - 1, 0)
    right_idx = idx

    # choose nearer of left/right
    left_dist = np.abs(target_ts_ns - src_ts_ns[left_idx])
    right_dist = np.abs(target_ts_ns - src_ts_ns[right_idx])
    choose_left = left_dist <= right_dist
    chosen = np.where(choose_left, left_idx, right_idx)

    # tolerance: where distance > half frame, we’ll ffill (later)
    dist = np.minimum(left_dist, right_dist)
    out = src.iloc[chosen].reset_index(drop=True).copy()
    out.loc[dist > half_frame_ns, :] = np.nan

    # forward-fill NaNs produced by tolerance gaps
    out = out.ffill().bfill()
    return out.reset_index(drop=True)



# ------ SAMoSA-specific helpers ------

def _collect_samosa_files(raw_dir: Path) -> List[Path]:
    files = sorted(raw_dir.glob("*.pkl")) or sorted(raw_dir.rglob("*.pkl"))
    print(f"[SAMoSA] Found {len(files)} pickle files under {raw_dir}")
    return files


def _estimate_hz_from_index(n_rows: int, assumed_hz: float = 50.0) -> float:
    # With synthetic equally spaced timestamps, report the assumed rate.
    return float(assumed_hz) if n_rows >= 3 else np.nan


def _apply_axis_map(vec: np.ndarray,
                    ch_names: list[str],
                    mapping: Dict[str, str]) -> np.ndarray:
    """
    Generic axis-mapping helper.

    mapping: {out_name: src_name or '-src_name'}
    ch_names: list of canonical names for vec columns, e.g. ["acc_x","acc_y","acc_z"]
    """
    name2idx = {n: i for i, n in enumerate(ch_names)}
    out = np.zeros_like(vec)
    for out_name, expr in mapping.items():
        sign, src = (-1.0, expr[1:]) if expr.startswith("-") else (1.0, expr)
        out_idx = name2idx[out_name]
        src_idx = name2idx[src]
        out[:, out_idx] = sign * vec[:, src_idx]
    return out


# pmap-specific helpers

def _split_on_gaps_seconds(df_in: pd.DataFrame, ts_col: str, cutoff_s: float) -> list[pd.DataFrame]:
    if df_in.empty:
        return []
    ts = df_in[ts_col].to_numpy(dtype=np.float64)
    dt = np.diff(ts)
    cut = np.where(dt > float(cutoff_s))[0] + 1
    bounds = np.concatenate(([0], cut, [len(df_in)]))
    out = []
    for k in range(len(bounds) - 1):
        a = int(bounds[k])
        b = int(bounds[k + 1])
        seg = df_in.iloc[a:b].copy()
        if len(seg) > 0:
            out.append(seg)
    return out


def _resample_segment_to_grid_50hz(
    seg: pd.DataFrame,
    ts_col: str,
    sensor_cols: list[str],
    label_cols: list[str],
    target_hz: float = 50.0,
) -> pd.DataFrame:
    dt = 1.0 / float(target_hz)
    t0 = float(seg[ts_col].iloc[0])
    t1 = float(seg[ts_col].iloc[-1])
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return pd.DataFrame()

    grid = np.arange(t0, t1 + 1e-12, dt, dtype=np.float64)
    if grid.size < 2:
        return pd.DataFrame()

    out = pd.DataFrame({ts_col: grid})

    x = seg[ts_col].to_numpy(dtype=np.float64)
    for c in sensor_cols:
        y = seg[c].to_numpy(dtype=np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2:
            out[c] = np.nan
        else:
            out[c] = np.interp(grid, x[m], y[m]).astype(np.float32)

    seg_lab = seg[[ts_col] + label_cols].copy()
    seg_lab = seg_lab.sort_values(ts_col)
    out = pd.merge_asof(out, seg_lab, on=ts_col, direction="nearest", tolerance=dt * 0.51)

    out = out.dropna(subset=sensor_cols, how="any")
    out = out.dropna(subset=label_cols, how="any")

    return out.reset_index(drop=True)

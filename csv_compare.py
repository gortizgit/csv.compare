#!/usr/bin/env python3
"""
csv_compare.py — Compare two CSV files for schema & data equality (with subset check).
- Verifies same columns (names & order) and dtypes (after normalization).
- Normalizes whitespace and NaNs; optionally ignores case.
- If sizes differ, checks whether the smaller dataset is a subset of the larger.
- Optionally compare by a key; if not provided, auto-detects a likely key.
Outputs a report folder with summary.md and CSVs for differences.

Note:
- Comparison is strict (no numeric tolerance, no equality by rounding).
- Differences are *displayed* with numeric values rounded to --display-decimals (default 2).
"""

import argparse
import os
import sys
import pandas as pd
from datetime import datetime


def load_csv(path, delimiter, na_values, encoding, keep_default_na):
    try:
        df = pd.read_csv(
            path,
            delimiter=delimiter,
            na_values=na_values,
            keep_default_na=keep_default_na,
            encoding=encoding,
            dtype=str,
        )
        return df
    except Exception as e:
        print(f"[ERROR] Failed to read '{path}': {e}", file=sys.stderr)
        sys.exit(2)


def normalize_df(df, trim=True, ignore_case=False):
    # Ensure all columns are strings for consistent comparison
    for col in df.columns:
        df[col] = df[col].astype(str)
        if trim:
            df[col] = df[col].str.strip()
        # Normalize common missing indicators
        df[col] = df[col].replace(
            {"None": pd.NA, "nan": pd.NA, "NaN": pd.NA, "": pd.NA}
        )
        if ignore_case:
            df[col] = df[col].str.lower()
    return df


def detect_key(dfA, dfB, max_candidates=3):
    """Find columns that look like a primary key in BOTH dataframes:
       - non-null, unique within each DF
       - same column name exists in both
       Try single columns, then pairs (lightweight heuristic).
    """
    common_cols = [c for c in dfA.columns if c in dfB.columns]
    # single-column candidates
    singles = []
    for c in common_cols:
        sA = dfA[c].dropna()
        sB = dfB[c].dropna()
        if len(sA) == len(dfA) and len(sB) == len(dfB) and sA.is_unique and sB.is_unique:
            singles.append([c])
    if singles:
        return singles[:max_candidates]
    # try simple 2-col combos (first 8 columns to keep it light)
    subset = common_cols[:8]
    pairs = []
    for i in range(len(subset)):
        for j in range(i + 1, len(subset)):
            cols = [subset[i], subset[j]]
            tA = dfA[cols].dropna().apply(tuple, axis=1)
            tB = dfB[cols].dropna().apply(tuple, axis=1)
            if len(tA) == len(dfA) and len(tB) == len(dfB) and tA.is_unique and tB.is_unique:
                pairs.append(cols)
    if pairs:
        return pairs[:max_candidates]
    return []


def fmt_display(x, decimals):
    """Format numeric-like strings to fixed decimals for OUTPUT ONLY."""
    if pd.isna(x):
        return ""
    try:
        num = float(x)
        return f"{num:.{decimals}f}"
    except Exception:
        return str(x)


# ---------- NUEVO: helpers para clasificar diferencias ----------

def is_date_like(val):
    """Heurística simple para detectar valores tipo fecha."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return False
    try:
        pd.to_datetime(s, errors="raise")
        return True
    except Exception:
        return False


def classify_diff_type(a, b, diff_value=None, decimals=2):
    """
    Clasifica el tipo de diferencia:
    - Numeric (Rounding)
    - Numeric (Value Change)
    - Date Mismatch
    - Text Difference
    - Other
    """
    # Intentar numérico
    a_num = pd.to_numeric(a, errors="coerce")
    b_num = pd.to_numeric(b, errors="coerce")

    if pd.notna(a_num) and pd.notna(b_num):
        if diff_value is None:
            diff_value = b_num - a_num
        # umbral pequeño basado en los decimales de display
        threshold = 0.5 * (10 ** (-decimals))
        if abs(diff_value) <= threshold:
            return "Numeric (Rounding)"
        return "Numeric (Value Change)"

    # Intentar fecha
    if is_date_like(a) and is_date_like(b):
        try:
            da = pd.to_datetime(a)
            db = pd.to_datetime(b)
            if da.date() != db.date():
                return "Date Mismatch"
        except Exception:
            pass
        return "Other"

    # Texto
    a_str, b_str = str(a), str(b)
    if a_str.strip().lower() != b_str.strip().lower():
        return "Text Difference"

    return "Other"


# ---------------------------------------------------------------

def compare(args):
    out_dir = args.out or f"csv_compare_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.md")

    dfA = load_csv(
        args.left,
        args.delimiter,
        args.na_values,
        args.encoding,
        not args.no_default_na,
    )
    dfB = load_csv(
        args.right,
        args.delimiter,
        args.na_values,
        args.encoding,
        not args.no_default_na,
    )

    # Keep original columns for schema checks
    colsA_orig = list(dfA.columns)
    colsB_orig = list(dfB.columns)

    # Normalize for content comparison
    dfA = normalize_df(dfA.copy(), trim=not args.no_trim, ignore_case=args.ignore_case)
    dfB = normalize_df(dfB.copy(), trim=not args.no_trim, ignore_case=args.ignore_case)

    # Align columns by name (order-sensitive schema check, but we also track set equality)
    set_only_in_A = [c for c in colsA_orig if c not in colsB_orig]
    set_only_in_B = [c for c in colsB_orig if c not in colsA_orig]

    same_order = colsA_orig == colsB_orig
    same_set = (set(colsA_orig) == set(colsB_orig))

    # Prepare for data comparison: intersecting columns only
    common_cols = [c for c in colsA_orig if c in colsB_orig]

    # Optional key handling
    key_cols = None
    autodetected_keys = []
    if args.key:
        key_cols = [k.strip() for k in args.key.split(",")]
        for k in key_cols:
            if k not in common_cols:
                print(
                    f"[WARN] Provided key '{k}' not found in both CSVs; it will be ignored for joining."
                )
        key_cols = [k for k in key_cols if k in common_cols]
        if not key_cols:
            print(
                "[WARN] No valid key columns found in both files. Falling back to full-row comparison."
            )
            key_cols = None
    else:
        autodetected_keys = detect_key(dfA[common_cols], dfB[common_cols])
        if autodetected_keys:
            key_cols = autodetected_keys[0]

    # For stable comparison, reduce to common columns
    A = dfA[common_cols].copy()
    B = dfB[common_cols].copy()

    # Write schema info
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# CSV Comparison Report\n\n")
        f.write(f"- Left (A): `{os.path.basename(args.left)}`\n")
        f.write(f"- Right (B): `{os.path.basename(args.right)}`\n")
        f.write(f"- Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write("## Schema\n")
        f.write(f"- Same column order: **{same_order}**\n")
        f.write(f"- Same column set (ignoring order): **{same_set}**\n")
        if set_only_in_A:
            f.write(f"- Columns only in A ({len(set_only_in_A)}): {set_only_in_A}\n")
        if set_only_in_B:
            f.write(f"- Columns only in B ({len(set_only_in_B)}): {set_only_in_B}\n")
        if autodetected_keys:
            f.write(f"- Auto-detected key candidates: {autodetected_keys}\n")
        if args.key:
            f.write(
                f"- User-provided key: {key_cols if key_cols else 'None valid in both files'}\n"
            )
        f.write("\n")

    # Data comparison
    if key_cols is None:
        # Full-row (set) comparison across common columns
        A_tuples = A.apply(
            lambda r: tuple(r.fillna(pd.NA).astype(object).tolist()), axis=1
        )
        B_tuples = B.apply(
            lambda r: tuple(r.fillna(pd.NA).astype(object).tolist()), axis=1
        )
        setA = set(A_tuples.tolist())
        setB = set(B_tuples.tolist())

        only_in_A = setA - setB
        only_in_B = setB - setA

        if only_in_A:
            pd.DataFrame(list(only_in_A), columns=common_cols).to_csv(
                os.path.join(out_dir, "rows_only_in_A.csv"), index=False
            )
        if only_in_B:
            pd.DataFrame(list(only_in_B), columns=common_cols).to_csv(
                os.path.join(out_dir, "rows_only_in_B.csv"), index=False
            )

        if len(A) <= len(B):
            subset_note = (
                f"A subset of B? {'YES' if len(only_in_A)==0 else 'NO'} "
                f"(rows in A not in B: {len(only_in_A)})"
            )
        else:
            subset_note = (
                f"B subset of A? {'YES' if len(only_in_B)==0 else 'NO'} "
                f"(rows in B not in A: {len(only_in_B)})"
            )

        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("## Data (Full-row comparison across common columns)\n")
            f.write(f"- Rows in A: {len(A)}\n")
            f.write(f"- Rows in B: {len(B)}\n")
            f.write(f"- Rows only in A: {len(only_in_A)}\n")
            f.write(f"- Rows only in B: {len(only_in_B)}\n")
            f.write(f"- {subset_note}\n")
            if only_in_A:
                f.write(f"- See `rows_only_in_A.csv` for examples.\n")
            if only_in_B:
                f.write(f"- See `rows_only_in_B.csv` for examples.\n")

    else:
        # Keyed comparison (strict), with formatted output
        A_keyed = A.set_index(key_cols, drop=False)
        B_keyed = B.set_index(key_cols, drop=False)

        # Keys presence
        keys_only_in_A = set(A_keyed.index) - set(B_keyed.index)
        keys_only_in_B = set(B_keyed.index) - set(A_keyed.index)

        if keys_only_in_A:
            pd.DataFrame(list(keys_only_in_A), columns=key_cols).to_csv(
                os.path.join(out_dir, "keys_only_in_A.csv"), index=False
            )
        if keys_only_in_B:
            pd.DataFrame(list(keys_only_in_B), columns=key_cols).to_csv(
                os.path.join(out_dir, "keys_only_in_B.csv"), index=False
            )

        # Compare values where keys exist in both
        common_keys = list(set(A_keyed.index) & set(B_keyed.index))
        diffs = []           # CSV original
        diffs_enriched = []  # CSV extendido

        for k in common_keys:
            rowA = A_keyed.loc[k]
            rowB = B_keyed.loc[k]
            # If key not unique in either DF, skip deep comparison for that key
            if isinstance(rowA, pd.DataFrame) or isinstance(rowB, pd.DataFrame):
                continue
            for col in common_cols:
                va = rowA[col]
                vb = rowB[col]
                # Equal if both NA
                if pd.isna(va) and pd.isna(vb):
                    continue
                # If one NA and the other not -> difference
                if pd.isna(va) or pd.isna(vb):
                    is_diff = True
                else:
                    # Strict compare (no tolerance)
                    is_diff = (str(va) != str(vb))

                if not is_diff:
                    continue

                # ----- Registro base (como antes) -----
                diff_rec = {c: rowA[c] for c in key_cols}
                diff_rec.update(
                    {
                        "column": col,
                        "A": fmt_display(va, args.display_decimals),
                        "B": fmt_display(vb, args.display_decimals),
                    }
                )
                diffs.append(diff_rec)

                # ----- Registro enriquecido (nuevo) -----
                a_num = pd.to_numeric(va, errors="coerce")
                b_num = pd.to_numeric(vb, errors="coerce")
                if pd.notna(a_num) and pd.notna(b_num):
                    diff_value = b_num - a_num
                    diff_value_display = fmt_display(diff_value, args.display_decimals)
                else:
                    diff_value = None
                    diff_value_display = ""

                diff_type = classify_diff_type(
                    va, vb, diff_value=diff_value, decimals=args.display_decimals
                )

                diff_rec_enriched = diff_rec.copy()
                diff_rec_enriched.update(
                    {
                        "Difference": diff_value_display,
                        "Diff_Type": diff_type,
                    }
                )
                diffs_enriched.append(diff_rec_enriched)

        # Escribimos el CSV original (sin romper nada)
        if diffs:
            df_base = pd.DataFrame(diffs)
            df_base.to_csv(
                os.path.join(out_dir, "value_differences.csv"), index=False
            )

            # Nuevo CSV detallado (como la tabla value_differences_with_diff_type)
            df_detail = pd.DataFrame(diffs_enriched)
            df_detail.to_csv(
                os.path.join(out_dir, "value_differences_detailed.csv"), index=False
            )

        if len(A_keyed) <= len(B_keyed):
            subset_note = (
                f"A keys subset of B? {'YES' if len(keys_only_in_A)==0 else 'NO'} "
                f"(keys only in A: {len(keys_only_in_A)})"
            )
        else:
            subset_note = (
                f"B keys subset of A? {'YES' if len(keys_only_in_B)==0 else 'NO'} "
                f"(keys only in B: {len(keys_only_in_B)})"
            )

        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("## Data (Keyed comparison)\n")
            f.write(f"- Rows in A: {len(A_keyed)}\n")
            f.write(f"- Rows in B: {len(B_keyed)}\n")
            f.write(f"- Keys only in A: {len(keys_only_in_A)}\n")
            f.write(f"- Keys only in B: {len(keys_only_in_B)}\n")
            if keys_only_in_A:
                f.write(f"- See `keys_only_in_A.csv`.\n")
            if keys_only_in_B:
                f.write(f"- See `keys_only_in_B.csv`.\n")
            f.write(f"- Value differences on common keys (rows): {len(diffs)}\n")
            if diffs:
                f.write(
                    f"- See `value_differences.csv` for per-column mismatches "
                    f"(values formatted to {args.display_decimals} decimals where numeric).\n"
                )
                f.write(
                    "- See `value_differences_detailed.csv` for enriched differences "
                    "(including numeric difference and type).\n"
                )
            f.write(f"- {subset_note}\n")

    print(f"[OK] Report generated at: {out_dir}")
    print(f"[OK] Summary: {summary_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Compare two CSVs for schema & data equality (and subset)."
    )
    ap.add_argument("--left", "-l", required=True, help="Path to left CSV (A)")
    ap.add_argument("--right", "-r", required=True, help="Path to right CSV (B)")
    ap.add_argument(
        "--key",
        "-k",
        help="Comma-separated key column(s). If omitted, auto-detects.",
    )
    ap.add_argument(
        "--delimiter", "-d", default=",", help="CSV delimiter (default ,)"
    )
    ap.add_argument(
        "--na-values",
        nargs="*",
        default=["", "NULL", "null", "None", "NaN"],
        help="Additional NA tokens",
    )
    ap.add_argument(
        "--no-default-na",
        action="store_true",
        help="Do NOT use pandas default NA values",
    )
    ap.add_argument(
        "--ignore-case", action="store_true", help="Case-insensitive comparison"
    )
    ap.add_argument(
        "--no-trim", action="store_true", help="Do NOT trim whitespace"
    )
    ap.add_argument(
        "--encoding", default="utf-8", help="File encoding (default utf-8)"
    )
    ap.add_argument(
        "--out", "-o", help="Output folder for the report"
    )
    # Display formatting (does not affect equality)
    ap.add_argument(
        "--display-decimals",
        type=int,
        default=2,
        help="Format numeric values in outputs to N decimals (display only, default 2)",
    )
    args = ap.parse_args()
    compare(args)


if __name__ == "__main__":
    main()

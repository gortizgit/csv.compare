#!/usr/bin/env python3
"""
csv_compare.py — Compare two CSV files for schema & data equality (with subset check).
- Verifies same columns (names & order) and dtypes (after normalization).
- Normalizes whitespace and NaNs; optionally ignores case.
- If sizes differ, checks whether the smaller dataset is a subset of the larger.
- Optionally compare by a key; if not provided, auto-detects a likely key.
Outputs a report folder with summary.md and CSVs for differences.
"""
import argparse
import os
import sys
import pandas as pd
from datetime import datetime

def load_csv(path, delimiter, na_values, encoding, keep_default_na):
    try:
        df = pd.read_csv(path, delimiter=delimiter, na_values=na_values,
                         keep_default_na=keep_default_na, encoding=encoding, dtype=str)
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
        df[col] = df[col].replace({"None": pd.NA, "nan": pd.NA, "NaN": pd.NA, "": pd.NA})
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
        for j in range(i+1, len(subset)):
            cols = [subset[i], subset[j]]
            tA = dfA[cols].dropna().apply(tuple, axis=1)
            tB = dfB[cols].dropna().apply(tuple, axis=1)
            if len(tA) == len(dfA) and len(tB) == len(dfB) and tA.is_unique and tB.is_unique:
                pairs.append(cols)
    if pairs:
        return pairs[:max_candidates]
    return []

def compare(args):
    out_dir = args.out or f"csv_compare_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.md")

    dfA = load_csv(args.left, args.delimiter, args.na_values, args.encoding, not args.no_default_na)
    dfB = load_csv(args.right, args.delimiter, args.na_values, args.encoding, not args.no_default_na)

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
                print(f"[WARN] Provided key '{k}' not found in both CSVs; it will be ignored for joining.")
        key_cols = [k for k in key_cols if k in common_cols]
        if not key_cols:
            print("[WARN] No valid key columns found in both files. Falling back to full-row comparison.")
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
            f.write(f"- User-provided key: {key_cols if key_cols else 'None valid in both files'}\n")
        f.write("\n")

    # Data comparison
    # If no key, compare by full-row tuples across common columns
    if key_cols is None:
        # Build tuple rows for hashing
        A_tuples = A.apply(lambda r: tuple(r.fillna(pd.NA).astype(object).tolist()), axis=1)
        B_tuples = B.apply(lambda r: tuple(r.fillna(pd.NA).astype(object).tolist()), axis=1)

        setA = set(A_tuples.tolist())
        setB = set(B_tuples.tolist())

        only_in_A = setA - setB
        only_in_B = setB - setA

        # Save diffs
        onlyA_path = os.path.join(out_dir, "rows_only_in_A.csv")
        onlyB_path = os.path.join(out_dir, "rows_only_in_B.csv")
        if only_in_A:
            pd.DataFrame(list(only_in_A), columns=common_cols).to_csv(onlyA_path, index=False)
        if only_in_B:
            pd.DataFrame(list(only_in_B), columns=common_cols).to_csv(onlyB_path, index=False)

        subset_note = ""
        if len(A) <= len(B):
            missing_from_B = len(only_in_A)
            subset_note = f"A subset of B? {'YES' if missing_from_B == 0 else 'NO'} (rows in A not in B: {missing_from_B})"
        else:
            missing_from_A = len(only_in_B)
            subset_note = f"B subset of A? {'YES' if len(missing_from_A) == 0 else 'NO'} (rows in B not in A: {missing_from_A})"

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
        # Keyed comparison
        A_keyed = A.set_index(key_cols, drop=False)
        B_keyed = B.set_index(key_cols, drop=False)

        # Keys presence
        keys_only_in_A = set(A_keyed.index) - set(B_keyed.index)
        keys_only_in_B = set(B_keyed.index) - set(A_keyed.index)

        # Save key presence diffs
        only_keys_A_path = os.path.join(out_dir, "keys_only_in_A.csv")
        only_keys_B_path = os.path.join(out_dir, "keys_only_in_B.csv")
        if keys_only_in_A:
            pd.DataFrame(list(keys_only_in_A), columns=key_cols).to_csv(only_keys_A_path, index=False)
        if keys_only_in_B:
            pd.DataFrame(list(keys_only_in_B), columns=key_cols).to_csv(only_keys_B_path, index=False)

        # Compare values where keys exist in both
        common_keys = list(set(A_keyed.index) & set(B_keyed.index))
        diffs = []
        for k in common_keys:
            rowA = A_keyed.loc[k]
            rowB = B_keyed.loc[k]
            # If key not unique in either DF, skip deep comparison for that key
            if isinstance(rowA, pd.DataFrame) or isinstance(rowB, pd.DataFrame):
                continue
            for col in common_cols:
                va = rowA[col]
                vb = rowB[col]
                # 1) Ambos NA => iguales
                if pd.isna(va) and pd.isna(vb):
                    continue

                iguales = False
                # 2) Si uno es NA y el otro no => diferentes (deja iguales=False)
                if pd.isna(va) != pd.isna(vb):
                    iguales = False
                else:
                    # 3) Intento numérico con redondeo/tolerancia si se solicitaron
                    try:
                        a_num = float(va)
                        b_num = float(vb)
                        # Redondeo exacto
                        if hasattr(args, "round_decimals") and args.round_decimals is not None:
                            if round(a_num, args.round_decimals) == round(b_num, args.round_decimals):
                                iguales = True
                        # Tolerancia numérica
                        if not iguales and hasattr(args, "numeric_tol") and args.numeric_tol is not None:
                            if abs(a_num - b_num) <= args.numeric_tol:
                                iguales = True
                    except Exception:
                        # No eran numéricos: sigue comparación string
                        pass

                    # 4) Comparación como texto si aún no son iguales
                    if not iguales and va == vb:
                        iguales = True

                if not iguales:
                    diff_rec = {c: rowA[c] for c in key_cols}
                    diff_rec.update({"column": col, "A": va, "B": vb})
                    diffs.append(diff_rec)

        diffs_path = os.path.join(out_dir, "value_differences.csv")
        if diffs:
            pd.DataFrame(diffs).to_csv(diffs_path, index=False)

        subset_note = ""
        if len(A_keyed) <= len(B_keyed):
            subset_note = f"A keys subset of B? {'YES' if len(keys_only_in_A)==0 else 'NO'} (keys only in A: {len(keys_only_in_A)})"
        else:
            subset_note = f"B keys subset of A? {'YES' if len(keys_only_in_B)==0 else 'NO'} (keys only in B: {len(keys_only_in_B)})"

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
            f.write(f"- Value differences on common keys: {len(diffs)}\n")
            if diffs:
                f.write(f"- See `value_differences.csv` for per-column mismatches.\n")
            f.write(f"- {subset_note}\n")

    print(f"[OK] Report generated at: {out_dir}")
    print(f"[OK] Summary: {summary_path}")

def main():
    ap = argparse.ArgumentParser(description="Compare two CSVs for schema & data equality (and subset).")
    ap.add_argument("--left", "-l", required=True, help="Path to left CSV (A)")
    ap.add_argument("--right", "-r", required=True, help="Path to right CSV (B)")
    ap.add_argument("--key", "-k", help="Comma-separated key column(s). If omitted, auto-detects.")
    ap.add_argument("--delimiter", "-d", default=",", help="CSV delimiter (default ,)")
    ap.add_argument("--na-values", nargs="*", default=["", "NULL", "null", "None", "NaN"], help="Additional NA tokens")
    ap.add_argument("--no-default-na", action="store_true", help="Do NOT use pandas default NA values")
    ap.add_argument("--ignore-case", action="store_true", help="Case-insensitive comparison")
    ap.add_argument("--no-trim", action="store_true", help="Do NOT trim whitespace")
    ap.add_argument("--encoding", default="utf-8", help="File encoding (default utf-8)")
    ap.add_argument("--out", "-o", help="Output folder for the report")
    # ---- new flags for numeric comparison ----
    ap.add_argument("--round-decimals", type=int, default=None,
                    help="Round numeric values to N decimals before equality check (e.g., 2)")
    ap.add_argument("--numeric-tol", type=float, default=None,
                    help="Numeric tolerance: abs(A-B) <= tol is considered equal (e.g., 0.005)")
    args = ap.parse_args()
    compare(args)

if __name__ == "__main__":
    main()

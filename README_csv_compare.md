# CSV Compare (Schema + Data + Subset)

This small utility checks that two CSV files are **equivalent in design and data**. If one file is larger, it also verifies that the smaller one is a **subset** of the larger.

## What it does
- **Schema check**
  - Same column order?
  - Same column set (ignoring order)?
  - Columns only in A / only in B
- **Normalization** (by default)
  - Trims whitespace
  - Normalizes empty/NULL/NaN to a consistent NA
  - Optional case-insensitive comparison
- **Data comparison**
  - If you **provide a key** (`--key id` or `--key col1,col2`), it compares values per column on matching keys
  - If you **don’t provide a key**, it compares **full rows** across the common columns
- **Subset check**
  - If row counts differ, it tells you whether the **smaller** is fully contained in the **larger**

## Quickstart (local)
1) Make sure you have Python 3.9+ and `pandas` installed:
```bash
pip install pandas
```
2) Run the tool:
```bash
python csv_compare.py --left path/to/A.csv --right path/to/B.csv

upython .\csv_compare.py -l .\Informatica.csv -r .\DOMO.csv --key Membership_Number

```
3) Optional flags:
```bash
# Delimiter, case-insensitive, custom NA tokens, and a composite key
python csv_compare.py -l A.csv -r B.csv -d ';' --ignore-case --na-values "" NA N/A --key id,store_id
```
4) Results
- A folder `csv_compare_report_YYYYMMDD_HHMMSS/` with:
  - `summary.md` → human-readable summary
  - `rows_only_in_A.csv` / `rows_only_in_B.csv` (when no key)
  - `keys_only_in_A.csv` / `keys_only_in_B.csv` (when key)
  - `value_differences.csv` (per-column mismatches on common keys)

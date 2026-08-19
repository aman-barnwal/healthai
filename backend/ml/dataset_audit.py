from pathlib import Path
import pandas as pd

DATASET_DIR = Path("dataset")

print("=" * 100)
print("HEALTHCAREAI DATASET AUDIT")
print("=" * 100)

results = []

for file_path in sorted(DATASET_DIR.rglob("*")):

    if not file_path.is_file():
        continue

    if file_path.suffix.lower() != ".csv":
        continue

    print("\n" + "-" * 100)
    print(f"DATASET: {file_path}")

    try:
        df = pd.read_csv(file_path)

        rows, cols = df.shape
        missing = int(df.isnull().sum().sum())
        duplicates = int(df.duplicated().sum())

        print(f"Shape: {rows:,} rows × {cols} columns")
        print(f"Missing values: {missing:,}")
        print(f"Duplicate rows: {duplicates:,}")

        print("\nColumns:")
        print(df.columns.tolist())

        print("\nData types:")
        print(df.dtypes.astype(str).value_counts().to_dict())

        print("\nPossible target:")
        print(df.columns[-1])

        results.append({
            "dataset": str(file_path),
            "rows": rows,
            "columns": cols,
            "missing_values": missing,
            "duplicate_rows": duplicates,
            "possible_target": df.columns[-1],
            "status": "OK"
        })

    except Exception as e:

        print("STATUS: FAILED")
        print("ERROR:", e)

        results.append({
            "dataset": str(file_path),
            "rows": None,
            "columns": None,
            "missing_values": None,
            "duplicate_rows": None,
            "possible_target": None,
            "status": f"FAILED: {str(e)[:100]}"
        })


print("\n" + "=" * 100)
print("FINAL DATASET SUMMARY")
print("=" * 100)

results_df = pd.DataFrame(results)

if not results_df.empty:
    print(results_df.to_string(index=False))

    output = DATASET_DIR / "dataset_audit_report.csv"
    results_df.to_csv(output, index=False)

    print(f"\nReport saved to: {output}")

print("\n" + "=" * 100)
print("AUDIT COMPLETE")
print("=" * 100)

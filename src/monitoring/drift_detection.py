import sqlite3
import pandas as pd
from pathlib import Path
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

DB_PATH = Path("predictions.db")
REPORTS_PATH = Path("reports")
REPORTS_PATH.mkdir(exist_ok=True)

# ── Load training data (reference) ────────────────────────────
print("Loading training data...")
ratings = pd.read_csv(
    "data/ratings.dat", sep="\t",
    names=["userId", "movieId", "rating", "timestamp"]
)
reference_df = ratings[["rating"]].rename(columns={"rating": "top_score"})

# ── Load live predictions (current) ───────────────────────────
print("Loading live predictions...")
conn = sqlite3.connect(DB_PATH)
current_df = pd.read_sql("SELECT top_score FROM predictions", conn)
conn.close()

if len(current_df) < 5:
    print("Not enough predictions yet — need at least 5. Skipping.")
    exit(0)

print(f"Reference: {len(reference_df)} rows | Current: {len(current_df)} rows")

# ── Run Evidently drift report ─────────────────────────────────
definition = DataDefinition(numerical_columns=["top_score"])
reference = Dataset.from_pandas(reference_df, data_definition=definition)
current   = Dataset.from_pandas(current_df,   data_definition=definition)

report = Report(metrics=[DataDriftPreset()])
result = report.run(reference_data=reference, current_data=current)

# ── Save HTML report ───────────────────────────────────────────
report_path = REPORTS_PATH / "drift_report.html"
result.save_html(str(report_path))
print(f"Drift report saved → {report_path}")

# ── Check drift — inspect structure ───────────────────────────
result_dict = result.dict()
first_metric = result_dict["metrics"][0]
print(f"Metric keys: {first_metric.keys()}")

# Try to find drift result anywhere in the dict
import json
result_str = json.dumps(result_dict, default=str)
drift_detected = "dataset_drift" in result_str and '"dataset_drift": true' in result_str

if drift_detected:
    print("DRIFT DETECTED — model may need retraining!")
    exit(1)
else:
    print("No drift detected — model is stable.")
    exit(0)

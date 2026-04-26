import json
from datasets import load_dataset

# Load dataset
ds = load_dataset("SWE-Lego/SWE-Lego-Synthetic-Data", split="resolved")

# Select required columns
processed_ds = ds.select_columns(["instance_id", "messages"])

# Convert to list format
data_list = processed_ds.to_list()

# Save as JSON file
filename = "swe_lego_synthetic_data_resolved_trajectories.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(data_list, f, ensure_ascii=False, indent=4)
print(f"Saved {len(data_list)} records to {filename}")



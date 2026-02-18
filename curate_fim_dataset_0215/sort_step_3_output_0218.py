import json

INPUT = "/data/yubo/datasets/process_data_output_0215/step_3_selected_fim_functions_0215_functions.json"
OUTPUT = "/data/yubo/datasets/process_data_output_0215/step_3_selected_fim_functions_0215_functions_sorted_0218.json"

with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

# Check for missing fim_score
for i, item in enumerate(data):
    if "fim_score" not in item:
        print(f"[WARNING] Item {i} missing fim_score: sample_id={item.get('sample_id')}, "
              f"repo_id={item.get('repo_id')}, func_name={item.get('func_name')}, "
              f"notes={item.get('notes')}")

def sort_key(item):
    notes = item.get("notes", "")
    is_swe = 0 if "[SWE-BENCH-REPO]" in notes else 1
    fim_score = -(item.get("fim_score", 0))
    return (is_swe, fim_score)

data.sort(key=sort_key)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Sorted {len(data)} items -> {OUTPUT}")
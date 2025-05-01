import json
import glob
import re
import os

# Pattern to match all relevant files
input_pattern = "run_20250501_185100_sciqva_llamaVo1_*.json"
output_file = "unified_llamaVo1.json"

def extract_index(filename):
    match = re.search(r"_(\d+)\.json$", filename)
    return int(match.group(1)) if match else -1

# Sort files based on the numeric suffix
files = sorted(glob.glob(input_pattern), key=extract_index)

if os.path.exists(output_file):
    with open(output_file, "r") as f:
        try:
            combined_data = json.load(f)
            if not isinstance(combined_data, list):
                print(f"Warning: {output_file} does not contain a JSON array.")
                combined_data = []
        except json.JSONDecodeError:
            print(f"Warning: {output_file} is not valid JSON.")
            combined_data = []
else:
    combined_data = []

new_records = 0
for file_path in files:
    with open(file_path, "r") as f:
        data = json.load(f)
        if isinstance(data, list):
            combined_data.extend(data)
            new_records += len(data)
        else:
            print(f"Warning: {file_path} does not contain a JSON array.")

# Write to unified file
with open(output_file, "w") as out_file:
    json.dump(combined_data, out_file, indent=2)

print(f"Appended {new_records} new unique records from {len(files)} files into: {output_file} ({len(combined_data)} total records)")

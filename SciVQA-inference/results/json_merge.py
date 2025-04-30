import json
import glob
import re

# Pattern to match all relevant files
input_pattern = "run_20250430_054931_sciqva_llamaVo1_*.json"
output_file = "unified_llamaVo1.json"

def extract_index(filename):
    match = re.search(r"_(\d+)\.json$", filename)
    return int(match.group(1)) if match else -1

# Sort files based on the numeric suffix
files = sorted(glob.glob(input_pattern), key=extract_index)

all_data = []

for file_path in files:
    with open(file_path, "r") as f:
        data = json.load(f)
        if isinstance(data, list):
            all_data.extend(data)
        else:
            print(f"Warning: {file_path} does not contain a JSON array.")

# Write to unified file
with open(output_file, "w") as out_file:
    json.dump(all_data, out_file, indent=2)

print(f"Merged {len(files)} files into: {output_file} ({len(all_data)} total records)")


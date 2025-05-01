import json
from collections import Counter

with open("results/unified_llamaVo1.json", "r") as f:
    unified_data = json.load(f)

with open("../SciVQA/test_without_answers_2025-04-14_15-30.json", "r") as f:
    test_data = json.load(f)

unified_ids = [item.get("instance_id") for item in unified_data if "instance_id" in item]
test_ids = [item.get("instance_id") for item in test_data if "instance_id" in item]

missing_ids = sorted(set(test_ids) - set(unified_ids))

id_counts = Counter(unified_ids)
duplicate_ids = sorted([id_ for id_, count in id_counts.items() if count > 1])

print(f"Missing instance_id values from unified_llamaVo1.json (present in test.json): {len(missing_ids)}")
print(missing_ids)

print(f"\nDuplicate instance_id values in unified_llamaVo1.json: {len(duplicate_ids)}")
print(duplicate_ids)

unique_data = []
seen_ids = set()
for item in unified_data:
    instance_id = item.get("instance_id")
    if instance_id and instance_id not in seen_ids:
        unique_data.append(item)
        seen_ids.add(instance_id)

with open("results/unified_llamaVo1_cleaned.json", "w") as f:
    json.dump(unique_data, f, indent=2)

print(f"\nDuplicates removed. Cleaned data saved to: results/unified_llamaVo1_cleaned.json")

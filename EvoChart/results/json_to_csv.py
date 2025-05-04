import os
import json
import pandas as pd
import re

# # List of JSON files
# json_files = [
#     "unified_evochart_0_445.json",
#     "unified_evochart_450_775.json",
#     "unified_evochart_779_1120.json",
#     "unified_evochart_1122_2480.json",
#     "unified_evochart_2484_3400.json",
#     "unified_evochart_3404_4190.json",
#     "unified_evochart_4191_4199.json"
# ]

# all_data = []

# # Process each file
# for file in json_files:
#     # Extract starting index from filename
#     match = re.search(r"_(\d+)_(\d+)", file)
#     if not match:
#         continue
#     start_idx = int(match.group(1))

#     with open(file, 'r') as f:
#         data = json.load(f)
#         for i, entry in enumerate(data):
#             entry["idx"] = start_idx + i
#             all_data.append(entry)

# # Convert to DataFrame and save as CSV
# merged_df = pd.DataFrame(all_data)
# merged_df[['idx', 'instance_id', 'answer_pred']].to_csv(
#     "merged_with_idx.csv", index=False)
# print(len(merged_df))

import pandas as pd

# Load both datasets
df1 = pd.read_csv("final_predictions_clean_4200.csv")
print(f"✅ Final shape: {df1.shape}")


# df1 = pd.read_csv("merged_with_idx.csv")
# df2 = pd.read_csv("../predictions.csv")

# # Normalize and cast instance_id to string
# df1['instance_id'] = df1['instance_id'].astype(str).str.strip()
# df2['instance_id'] = df2['instance_id'].astype(str).str.strip()

# # Drop any duplicates in both files (keep last for safety)
# df1 = df1.drop_duplicates(subset='instance_id', keep='last')
# df2 = df2.drop_duplicates(subset='instance_id', keep='last')

# # Merge and resolve answer_pred
# merged = pd.merge(
#     df2,
#     df1[['instance_id', 'answer_pred']],
#     on='instance_id',
#     how='outer',
#     suffixes=('_pred', '_df1')
# )

# # Prefer answer_pred from df1
# merged['answer_pred'] = merged['answer_pred_df1'].combine_first(
#     merged['answer_pred_pred'])

# # Final deduplication and trim
# final_df = merged[['instance_id', 'answer_pred']
#                   ].drop_duplicates(subset='instance_id', keep='last')

# # Optional: check if instance_ids are integers from 0 to 4199
# print(len(df2))
# print(len(final_df))
# expected_ids = set(df2['instance_id'])
# actual_ids = set(final_df['instance_id'])


# missing = expected_ids - actual_ids
# extra = actual_ids - expected_ids

# if missing:
#     print(f"⚠️ Missing instance_ids: {len(missing)}")
# if extra:
#     print(f"⚠️ Extra instance_ids: {len(extra)}")

# # Final filtering if needed
# final_df = final_df.drop_duplicates(subset='instance_id', keep='last')

# # Save to file
# final_df.to_csv("final_predictions_clean_4200.csv", index=False)
# print(f"✅ Final shape: {final_df.shape}")

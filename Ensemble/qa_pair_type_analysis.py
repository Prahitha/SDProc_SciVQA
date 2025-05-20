import pandas as pd
import os


test_without_answers = pd.read_json(
    'test_without_answers_2025-04-14_15-30.json')

print(test_without_answers['qa_pair_type'].value_counts())

bespoke = pd.read_csv('bespoke.csv')
qwen_32 = pd.read_csv('Annoted/500/predictions.csv')

# Ensure both columns are string and lowercase for comparison
bespoke['answer_pred_str'] = bespoke['answer_pred'].astype(str).str.lower()
qwen_32['answer_pred_str'] = qwen_32['answer_pred'].astype(str).str.lower()

# Merge on id and compare answer_pred_str
merged = bespoke.merge(qwen_32, left_on='instance_id',
                       right_on='instance_id', suffixes=('_bespoke', '_qwen'))

qa_pair_type_merged = merged.merge(test_without_answers, left_on='instance_id',
                                   right_on='instance_id', how='left')

print(qa_pair_type_merged['qa_pair_type'].value_counts())

# Create answer_pred_merged column, default to answer_pred_qwen
qa_pair_type_merged['answer_pred'] = qa_pair_type_merged['answer_pred_qwen']

# For closed-ended infinite answer set visual, if answers differ, set to 'yes'
mask = (
    (qa_pair_type_merged['qa_pair_type'] == 'closed-ended finite answer set binary visual') &
    (qa_pair_type_merged['answer_pred_str_bespoke']
     != qa_pair_type_merged['answer_pred_str_qwen'])
)
qa_pair_type_merged.loc[mask, 'answer_pred'] = 'no'

mask = (
    (qa_pair_type_merged['qa_pair_type'] == 'closed-ended finite answer set binary non-visual') &
    (qa_pair_type_merged['answer_pred_str_bespoke']
     != qa_pair_type_merged['answer_pred_str_qwen'])
)
qa_pair_type_merged.loc[mask, 'answer_pred'] = 'no'

# Save instance_id and answer_pred_merged to CSV
output_path = '/Users/harshita/Documents/Paper/SDProc_SciVQA/Ensemble/Annoted/500_visual_yes/binary_no.csv'

# Check if there are null values in the answer_pred column
if qa_pair_type_merged['answer_pred'].isnull().any():
    print("There are null values in the answer_pred column.")
    print(qa_pair_type_merged[qa_pair_type_merged['answer_pred'].isnull()])
else:
    print("There are no null values in the answer_pred column.")

# qa_pair_type_merged[['instance_id', 'answer_pred']].to_csv(
#     output_path, index=False)

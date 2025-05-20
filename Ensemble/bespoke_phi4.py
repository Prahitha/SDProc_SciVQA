import pandas as pd


annotated = pd.read_csv(
    '/Users/harshita/Documents/Paper/SDProc_SciVQA/Ensemble/predictions - merged_wandb_export_updated copy.csv')
annotated = annotated.rename(
    columns={'id': 'instance_id', 'final_answer': 'answer_pred'})
annotated = annotated[['instance_id', 'answer_pred']]

qwen = pd.read_csv(
    '/Users/harshita/Documents/Paper/SDProc_SciVQA/Ensemble/predictions.csv')


annotated.to_csv('annotated.csv', index=False)


# qwen_32['answer_pred'] = qwen_32['answer_pred_qwen']

# if qwen_32['answer_pred'].isnull().any():
#     # only replace null values
#     qwen_32.loc[qwen_32['answer_pred'].isnull(
#     ), 'answer_pred'] = qwen_32.loc[qwen_32['answer_pred'].isnull(), 'answer_pred_bespoke']
# else:
#     print("No null values to replace")

# qwen_32 = qwen_32[['instance_id', 'answer_pred']]

# qwen_32.to_csv('qwen_32_bespoke.csv', index=False)

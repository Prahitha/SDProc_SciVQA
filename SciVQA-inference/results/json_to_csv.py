import pandas as pd


df = pd.read_json('./unified_llamaVo1_cleaned.json')
df['instance_id'] = df['instance_id'].astype(str)
df['answer_pred'] = df['answer_pred'].astype(str)
df[['instance_id', 'answer_pred']].to_csv('predictions.csv', index=False)

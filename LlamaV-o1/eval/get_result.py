from openai import OpenAI
import base64
from PIL import Image
import io
import os
from pathlib import Path
import json
import ast
import pandas as pd

openai_api_key = "<add openai API key>"
client = OpenAI(
    api_key=openai_api_key  
)
system_prompt = """

You are a reasoning evaluator designed to assess the alignment, coherence, and quality of reasoning steps in text responses. Your task is to evaluate reasoning steps between the *ground truth* and the *LLM response* using the following metrics:

1. **Faithfulness-Step (1-10):**  
   - Definition: Measures how well the reasoning steps in the LLM response align with the source reasoning steps.
   - Scoring Guidelines:  
     - 9-10: All or almost all steps match or closely reflect the ground truth reasoning.  
     - 7-8: Most steps are aligned, with minor deviations.  
     - 5-6: Some steps align, but several are missing or significantly altered.  
     - 3-4: Few steps align correctly; most are off or missing.  
     - 1-2: The majority of steps are not aligned with the source.

2. **Faithfulness-Token (1-10):**  
   - Definition: Extends Faithfulness-Step to a token-level granularity, checking if the content within each reasoning step is true to the source.
   - Scoring Guidelines:  
     - 9-10: Token-level details mirror the ground truth closely.  
     - 7-8: Minor token-level deviations but largely faithful.  
     - 5-6: Noticeable inaccuracies in token-level details.  
     - 3-4: Several token-level discrepancies.  
     - 1-2: Most token-level details are incorrect or fabricated.

3. **Informativeness-Step (Info-Step) (1-10):**  
   - Definition: Measures how well the reasoning steps extract all relevant information from the source.
   - Scoring Guidelines:  
     - 9-10: Almost all critical information steps are present and accurate.  
     - 7-8: Most important points are included, with minor omissions.  
     - 5-6: Some key information is missing or underdeveloped.  
     - 3-4: Limited inclusion of critical content.  
     - 1-2: Very poor extraction of relevant information.

4. **Repetition-Token (1-10):**  
   - Definition: Identifies repeated or unnecessarily paraphrased reasoning steps within the hypothesis.
   - Scoring Guidelines:  
     - 9-10: No or minimal unnecessary repetition.  
     - 7-8: Minor repetition that doesn’t impede clarity.  
     - 5-6: Noticeable repetition that doesn’t add value.  
     - 3-4: Frequent repetition that disrupts coherence.  
     - 1-2: Excessive repetition reducing the quality of reasoning.

5. **Hallucination (1-10):**  
   - Definition: Detect irrelevant or invented reasoning steps not aligned with the source.
   - Scoring Guidelines:  
     - 9-10: No hallucinations; all reasoning is grounded in the source.  
     - 7-8: One or two minor hallucinations.  
     - 5-6: Several steps contain invented or irrelevant details.  
     - 3-4: Many hallucinations, but some grounding remains.  
     - 1-2: Mostly hallucinated reasoning.

6. **Redundancy (1-10):**  
   - Definition: Identify redundant reasoning steps that do not add value.
   - Scoring Guidelines:  
     - 9-10: No unnecessary steps; very concise.  
     - 7-8: Minor redundancy.  
     - 5-6: Some steps clearly unnecessary.  
     - 3-4: Many redundant steps.  
     - 1-2: Excessive redundancy that hampers clarity.

7. **Semantic Coverage-Step (1-10):**  
   - Definition: How well the hypothesis covers the essential semantic elements from the source reasoning steps.
   - Scoring Guidelines:  
     - 9-10: Almost complete semantic coverage of all important elements.  
     - 7-8: Good coverage but some minor elements are missing.  
     - 5-6: Partial coverage with noticeable gaps.  
     - 3-4: Significant semantic gaps.  
     - 1-2: Very poor coverage of essential meaning.

8. **Reasoning Alignment (1-10):**  
   - Definition: Overall alignment between the hypothesis and the reference reasoning chain.
   - Scoring Guidelines:  
     - 9-10: Very closely aligned, minimal divergence.  
     - 7-8: Mostly aligned, with some minor issues.  
     - 5-6: Some alignment, but also several misalignments.  
     - 3-4: Poor alignment, though occasional matches.  
     - 1-2: Fundamentally misaligned reasoning.

9. **Commonsense (1-10):**  
   - Definition: Check for missing commonsense reasoning required to solve the problem.
   - Scoring Guidelines:  
     - 9-10: Adequate commonsense reasoning present.  
     - 7-8: Minor commonsense gaps but mostly adequate.  
     - 5-6: Noticeable commonsense gaps.  
     - 3-4: Many commonsense steps missing.  
     - 1-2: Almost entirely lacking necessary commonsense.

10. **Missing Step (1-10):**  
    - Definition: Identify if any necessary reasoning steps are missing.
    - Scoring Guidelines:  
      - 9-10: No critical steps missing.  
      - 7-8: Minor missing steps that don’t significantly affect the conclusion.  
      - 5-6: Some important steps absent, affecting the outcome.  
      - 3-4: Several crucial missing steps.  
      - 1-2: Major gaps; the reasoning chain is incomplete.

**Additional Instructions for Consistency:**

- Always follow the above scoring guidelines strictly.  
- Before scoring, re-read both the ground truth and the LLM response carefully.  
- Compare the reasoning steps directly to determine where they align or diverge.
- Use the provided scoring benchmarks (anchor examples, if any) as a reference to maintain consistency across evaluations.
- Avoid subjective interpretation and adhere to the given thresholds.
- Once scores for all metrics are determined, compute the Overall Score as the average of all metric scores.
- Provide the final output as a Python dictionary with the structure only dont add a anything extra , beacuase your out will be used in code pipeline. So single change in you output will crash whole system. :

# Example output : {'Faithfulness-Step': 8.0, 'Faithfulness-Token': 7.5, 'Informativeness-Step': 8.5, 'Repetition-Token': 9.0, 'Hallucination': 9.5, 'Redundancy': 8.0, 'Semantic Coverage-Step': 8.5, 'Reasoning Alignment': 8.0, 'Commonsense': 9.0, 'Missing Step': 8.5 , 'Overall Score': 8.65}

# Do not give output in following format :

```python
{
  'Faithfulness-Step': 1.0,
  'Faithfulness-Token': 1.0,
  'Informativeness-Step': 1.0,
  'Repetition-Token': 9.0,
  'Hallucination': 1.0,
  'Redundancy': 9.0,
  'Semantic Coverage-Step': 1.0,
  'Reasoning Alignment': 1.0,
  'Commonsense': 1.0,
  'Missing Step': 1.0,
  'Overall Score': 2.6
}
```

 """

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "EvaluationScores",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "Faithfulness-Step": {"type": "number"},
                "Faithfulness-Token": {"type": "number"},
                "Informativeness-Step": {"type": "number"},
                "Repetition-Token": {"type": "number"},
                "Hallucination": {"type": "number"},
                "Redundancy": {"type": "number"},
                "Semantic Coverage-Step": {"type": "number"},
                "Reasoning Alignment": {"type": "number"},
                "Commonsense": {"type": "number"},
                "Missing Step": {"type": "number"},
                "Overall Score": {"type": "number"}
            },
            "required": [
                "Faithfulness-Step",
                "Faithfulness-Token",
                "Informativeness-Step",
                "Repetition-Token",
                "Hallucination",
                "Redundancy",
                "Semantic Coverage-Step",
                "Reasoning Alignment",
                "Commonsense",
                "Missing Step",
                "Overall Score"
            ],
            "additionalProperties": False
        }
    }
}
 

def evaluate_steps(question , ground_truth, llm_response):
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question + "\n" + f"Ground Truth : {ground_truth}" + "\n" + f"LLM Response : {llm_response}" },
            ],
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",  
        messages=messages,
        response_format=response_format,
        max_tokens=500,
        temperature = 0.0,
    )
    return response.choices[0].message.content

system_prompt_2 = """You are a helpful Assistant. Provide helpful response to the user's question."""

def compare_results(question, ground_truth, llm_response):
    # 
    messages = [
        {"role": "system", "content": system_prompt_2},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"""   
Evaluate the following answer based on Accuracy:
Question: {question}
Ground Truth: {ground_truth}
Model Prediction: {llm_response}
Match the meaning of the ground truth with the model prediction and if it matches give a 1. Otherwise 0.
Strictly return only the numeric score, without any additional commentary"""},
            ],
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",  
        messages=messages,
        max_tokens=10,
        temperature=0.0
    )

    return response.choices[0].message.content

import json
import ast
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

result_file = "path/to/result_file"
print("Using : {}".format(result_file))
print("="*50)

with open(result_file) as flp:
    files = json.load(flp)

data = []


def process_file(fp):
    question = fp["question"]
    ground_truth = fp["answer"]
    llm_response = fp["llm_response"]

    res = evaluate_steps(question, ground_truth, llm_response)
    res = ast.literal_eval(res)

    res["idx"] = fp["idx"]
    res["question"] = fp["question"]
    res["ground truth"] = fp["answer"]
    res["llm response"] = fp["llm_response"]
    res["ground truth final answer"] = ground_truth
    res["llm response final answer"] = llm_response.split("\n")[-1]
    if "is_correct" in fp:
        res["Final Answer Score"] = fp["is_correct"]
    else:
        if isinstance(ground_truth, int) and len(choices) > 0:
            ground_truth = choices[ground_truth]
        res["Final Answer Score"] = compare_results(question, ground_truth, llm_response)
    return res

with ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_file, files), total=len(files), desc="Processing"))

df = pd.DataFrame(results)
output_path = "out/{}_outfile.csv".format(result_file)
df.to_csv(output_path, index=False)


# df = pd.read_csv(output_path)

int_list = []
for item in df["Final Answer Score"].tolist():
    try:
        int_list.append(float(item))
    except Exception as e:
        print("Skipping for", str(e))
        continue
correct_answers = sum(int_list)
total_answers = len(int_list)
percentage_correct = (correct_answers / total_answers) * 100

print(f"Percentage of correct final answers: {percentage_correct:.2f}%")

max_score = 10  
percentage_correct = (df["Overall Score"].sum() / (len(df) * max_score)) * 100
print(f"Percentage of correct steps (normalized): {percentage_correct:.2f}%")

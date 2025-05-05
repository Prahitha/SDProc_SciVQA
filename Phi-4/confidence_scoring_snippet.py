from torch.nn.functional import softmax
import torch

def generate_inner(question, image, caption, qa_pair_type, answer_options):
    kwargs = {
        'max_new_tokens': max_new_tokens,
        "top_p": 0.9,
        "pad_token_id": 128004,
        "bos_token_id": 128000,
        "do_sample": True,
        "eos_token_id": [
            128001,
            128008,
            128009
        ],
        "temperature": 0.4,
        "num_beams": num_beams,
        "use_cache": True,
    }

    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image'},
                {'type': 'text', 'text': question + summary_prompt}
            ],
        }
    ]

    def infer(messages: dict):
        input_text = processor.apply_chat_template(
            messages, add_generation_prompt=True)
        inputs = processor(image, input_text, return_tensors='pt').to(model.device)
        
        # Generate output with logits for confidence calculation
        output = model.generate(**inputs, **kwargs, output_scores=True, return_dict_in_generate=True)
        
        # Extract logits for confidence score calculation
        logits = output.scores[-1]  # Get logits of the final token prediction
        probs = softmax(logits, dim=-1)  # Apply softmax to convert to probabilities
        confidence_score = torch.max(probs).item()  # Get confidence as the max probability
        generated_output = processor.decode(output.sequences[0][inputs['input_ids'].shape[1]:]).replace('<|eot_id|>', '').replace("<|end_of_text|>", "")
        
        return generated_output, confidence_score

    out, confidence_score = infer(messages)
    
    caption_prompt = f"Provide a detailed description of the image titled {{{caption}}}, particularly emphasizing the aspects related to the question."
    messages.extend(tmp(out, caption_prompt))

    out, confidence_score = infer(messages)  # Update with new confidence score after caption generation

    reasoning_prompt = (
        "Provide a chain-of-thought, logical explanation of the problem. "
        "In case of multiple plots, carefully compare the y-axis and x-axis scales across plots and note any differences. "
        "Identify which lines correspond to which entities and distinguish them from shaded regions, which represent confidence intervals. "
        "Pay attention to overlapping lines or narrow peaks. Provide a step-by-step reasoning to reach the answer."
    )
    messages.extend(tmp(out, reasoning_prompt))

    reasoning, confidence_score = infer(messages)  # Update with confidence score for reasoning

    conclusion_prompt = "Return a short, exact answer, no more than a few words. Do not explain or describe. The answer does not have to be a full sentence."
    messages.extend(tmp(out, conclusion_prompt))

    out, confidence_score = infer(messages)  # Final confidence score update for conclusion

    if "closed-ended" in qa_pair_type and "finite answer set" in qa_pair_type:
        if "non-binary" in qa_pair_type and answer_options:
            final_prompt = (
                f"Based on the reasoning above, match it to one or more of the provided answer options: {answer_options}. "
                "Return only the corresponding letter(s) of the correct answer(s). "
                "Do not explain your choice, do not rephrase the answer, and do not repeat the option text. "
                "Only output the letter(s) corresponding to the correct choice. "
                "If multiple letters are correct, separate them by commas without spaces (for example: B,C). "
                "If all options are correct, return A,B,C,D. "
                "Do not add anything else."
            )
        elif "binary" in qa_pair_type:
            final_prompt = "Return either 'Yes' or 'No'. Do not add anything else - not even punctuation marks."
        else:
            final_prompt = "Give the exact correct answer, with no extra explanation."
    else:
        final_prompt = "Give a direct, concise answer to the question only."

    messages.extend(tmp(out, final_prompt))
    out, confidence_score = infer(messages)

    print(f"Question: {question}\nAnswer: {out}\nConfidence Score: {confidence_score}")
    
    return out, reasoning, confidence_score

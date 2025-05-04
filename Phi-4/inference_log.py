import torch
import os
import json
import argparse
import time
import logging
import sys
import traceback
import pandas as pd
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("phi4_inference.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Phi-4 Multimodal Batch Inference")
parser.add_argument("--samples", type=int, default=10, help="No of samples from the selected data type")
parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search")
parser.add_argument("--output_path", type=str, default="results_phi4.json", help="Path to output results file")
parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
args = parser.parse_args()

# Update log level from args
numeric_level = getattr(logging, args.log_level.upper(), None)
if isinstance(numeric_level, int):
    logger.setLevel(numeric_level)

max_new_tokens = 512

def generate_inner(question, image, caption, qa_pair_type, answer_options):
    try:
        logger.info(f"Processing question: {question[:50]}...")
        
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cuda"
        logger.info(f"Using device: {device}")
        model_path = "microsoft/Phi-4-multimodal-instruct"
        
        logger.info("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            num_crops=1
        )
        
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype='auto',
            _attn_implementation="flash_attention_2",
        ).to(device)
        
        kwargs = {
            'max_new_tokens': max_new_tokens,
            "top_p": 0.9,
            "bos_token_id": 128000,
            "do_sample": True,
            "pad_token_id": processor.tokenizer.eos_token_id,
            "eos_token_id": processor.tokenizer.eos_token_id,
            "temperature": 0.4,
            "num_beams": args.num_beams,
            "use_cache": True,
        }

        prompt_text = question + "\nSummarize how you will approach the problem and explain the steps you will take to reach the answer."
        
        messages = [
            {
                'role': 'user',
                'content': f"<|image_1|>\n{{{prompt_text}}}"
            }
        ]

        def infer(messages: list) -> str:
            try:
                logger.debug(f"Inferring with message length: {len(messages)}")
                full_prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                #inputs = processor(text=full_prompt, images=[image], return_tensors='pt')
                #inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                #input_length = inputs['input_ids'].shape[1]
                
                #logger.debug(f"Starting generation with input length: {input_length}")
                #output = model.generate(**inputs, **kwargs, num_logits_to_keep=1)
                #response = processor.batch_decode(output[:, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                #logger.debug(f"Generated response of length: {len(response)}")
               # return response

                inputs = processor(text=full_prompt, images=image, return_tensors='pt').to("cuda")
                generate_ids = model.generate(
                                **inputs,
                                **kwargs
                               # generation_config=generation_config,
                                )
                generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
                response = processor.batch_decode(
                        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0]
                return response
            except Exception as e:
                logger.error(f"Error during inference: {str(e)}")
                logger.error(traceback.format_exc())
                raise

        def tmp(inp, out):
            return [
                {
                    'role': 'assistant',
                    'content': inp
                },
                {
                    'role': 'user',
                    'content': f"<|image_1|>\n{out}"
                }
            ]
        
        logger.info("Starting initial inference...")
        out = infer(messages)
        
        logger.info("Processing caption prompt...")
        caption_prompt = f"Provide a detailed description of the image titled {{{caption}}}, particularly emphasizing the aspects related to the question."
        messages.extend(tmp(out, caption_prompt))

        out = infer(messages)
        
        logger.info("Processing reasoning prompt...")
        reasoning_prompt = (
            "Provide a chain-of-thought, logical explanation of the problem. "
            "In case of multiple plots, carefully compare the y-axis and x-axis scales across plots and note any differences. "
            "Identify which lines correspond to which entities and distinguish them from shaded regions, which represent confidence intervals. "
            "Pay attention to overlapping lines or narrow peaks. Provide a step-by-step reasoning to reach the answer."
        )
        messages.extend(tmp(out, reasoning_prompt))

        reasoning = infer(messages)
        
        logger.info("Processing conclusion prompt...")
        conclusion_prompt = "Return a short, exact answer, no more than a few words. Do not explain or describe. The answer does not have to be a full sentence."
        messages.extend(tmp(reasoning, conclusion_prompt))

        out = infer(messages)
        
        logger.info("Processing final prompt...")
        if "closed-ended" in qa_pair_type and "finite answer set" in qa_pair_type:
            if "non-binary" in qa_pair_type and answer_options:
                final_prompt = (
                    f"Based on the reasoning above, match it to one or more of the provided answer options: {{{answer_options}}}. "
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
        
        out = infer(messages)
        kwargs['max_new_tokens'] = 50

        logger.info(f"Final answer: {out}")
        return out, reasoning
    except Exception as e:
        logger.error(f"Error in generate_inner: {str(e)}")
        logger.error(traceback.format_exc())
        return f"ERROR: {str(e)}", f"Reasoning failed due to error: {str(e)}"

def should_override_with_unanswerable(reasoning: str) -> bool:
    reasoning_lower = reasoning.lower()
    keywords = [
        "cannot be determined",
        "cannot determine",
        "not possible to determine",
        "insufficient information",
        "not enough information",
        "unanswerable",
        "data is missing",
        "lack of information",
        "figure is not provided"
    ]
    return any(keyword in reasoning_lower for keyword in keywords)

def reasoning_steps_answer(question, image, caption, qa_pair_type, answer_options):
    predicted_answer, reasoning = generate_inner(question, image, caption, qa_pair_type, answer_options)
    return predicted_answer, reasoning


logger.info("="*50)
logger.info("Starting Phi-4 Multimodal Batch Inference")
logger.info(f"Samples: {args.samples}, Beams: {args.num_beams}")


def get_old_results(output_path, idx):
    try:
        with open(output_path, "r") as json_file:
            old_data = json.load(json_file)
        for d in old_data:
            if d['instance_id'] == idx:
                return d
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load old results: {str(e)}")
        return None
    return None


# Load dataset
try:
    logger.info("Loading dataset...")
    with open("../SciVQA/validation_2025-03-27_18-34-44.json", 'r') as f:
        ds = json.load(f)

    ds = ds[:args.samples]
    images_dir = "../images_validation"
    logger.info(f"Loaded {len(ds)} samples from dataset")
except Exception as e:
    logger.critical(f"Failed to load dataset: {str(e)}")
    logger.critical(traceback.format_exc())
    sys.exit(1)

all_data = []
error_count = 0
success_count = 0

output_path = f"results_phi4_beams{args.num_beams}.json"

for data in tqdm(ds, total=len(ds)):
    try:
        start_time = time.time()

        # Extract data fields
        image_file = data['image_file']
        question = data['question']
        caption = data['caption']
        idx = data['instance_id']
        qa_pair_type = data['qa_pair_type']
        answer_options = data['answer_options']

        logger.info(f"Processing instance {idx}")
        logger.info(f"Question: {question}")

        # Check if we already have results for this instance
        existing_result = get_old_results(output_path, idx)
        if existing_result:
            logger.info(f"Found existing result for instance {idx}, skipping")
            all_data.append(existing_result)
            continue

        # Load image and generate answer
        image_path = os.path.join(images_dir, image_file)
        if not os.path.exists(image_path):
            error_msg = f"Image file not found: {image_path}"
            logger.error(error_msg)
            all_data.append({
                "instance_id": idx,
                "question": question,
                "answer_pred": f"ERROR: {error_msg}",
                "reasoning": f"ERROR: {error_msg}",
                "error": error_msg,
                "processing_time": 0
            })
            error_count += 1
            continue
            
        try:
            image = Image.open(image_path)
        except Exception as e:
            error_msg = f"Failed to open image {image_path}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            all_data.append({
                "instance_id": idx,
                "question": question,
                "answer_pred": f"ERROR: {error_msg}",
                "reasoning": f"ERROR: {error_msg}",
                "error": error_msg,
                "processing_time": 0
            })
            error_count += 1
            continue
            
        try:
            logger.info(f"Starting inference for instance {idx}")
            result, reasoning = generate_inner(
                question, image, caption, qa_pair_type, answer_options)

            # Check if the question should be considered unanswerable
            if should_override_with_unanswerable(reasoning):
                logger.info(f"Instance {idx} considered unanswerable based on reasoning")
                result = "It is not possible to answer this question based only on the provided data."

            elapsed_time = time.time() - start_time
            logger.info(f"Processed instance {idx} in {elapsed_time:.2f} seconds")

            all_data.append({
                "instance_id": idx,
                "question": question,
                "answer_pred": result,
                "reasoning": reasoning,
                "processing_time": elapsed_time
            })
            success_count += 1

        except Exception as e:
            error_msg = f"Inference failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            all_data.append({
                "instance_id": idx,
                "question": question,
                "answer_pred": f"ERROR: {error_msg}",
                "reasoning": f"ERROR: {error_msg}",
                "error": error_msg,
                "processing_time": time.time() - start_time
            })
            error_count += 1

        # Save intermediate results
        try:
            with open(output_path, "w") as json_file:
                json.dump(all_data, json_file, indent=4)
            logger.info(f"Saved intermediate results to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {str(e)}")
            logger.error(traceback.format_exc())

    except Exception as e:
        logger.error(f"Error processing sample {data.get('instance_id', 'unknown')}: {str(e)}")
        logger.error(traceback.format_exc())
        error_count += 1
        continue

# Final save of results
try:
    with open(output_path, "w") as json_file:
        json.dump(all_data, json_file, indent=4)
    logger.info(f"Finished processing {len(all_data)} samples. Results saved to {output_path}")
    logger.info(f"Successful: {success_count}, Errors: {error_count}")
except Exception as e:
    logger.critical(f"Failed to save final results: {str(e)}")
    logger.critical(traceback.format_exc())

logger.info("Script completed")

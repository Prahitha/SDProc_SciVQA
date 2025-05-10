import json
import os
from pathlib import Path
from typing import List, Dict, Any
import yaml
from wandb_config import init_wandb
import pandas as pd
import time
import wandb
from vllm_inference import VLLMInference, VLLMConfig
from tqdm import tqdm
from dataset import SciVQADataset
from datetime import datetime


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_run_directory(config: dict) -> Path:
    """Create a directory for the current run with timestamp and optional custom name."""
    # Get base directory from config or use default
    base_dir = Path(config['output'].get('base_dir', 'results'))
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create run name with timestamp and optional custom name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config['output'].get('run_name')
    if run_name is None:
        run_name = f'{config["vllm"]["model"]}_{config["dataset"]["split"]}_{timestamp}'
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def save_results(results: List[Dict[str, Any]], config: dict, split: str, run_dir: Path, wandb_run=None):
    """Save results to file and log to wandb.

    Args:
        results: List of result dictionaries
        config: Configuration dictionary
        split: Dataset split name
        run_dir: Directory to save results
        wandb_run: Optional wandb run instance
    """
    # Create predictions directory in run directory
    predictions_dir = run_dir / 'predictions'
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Save to file with split name
    output_path = predictions_dir / f"{split}_predictions.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Log to wandb if initialized
    if wandb_run is not None:
        try:
            wandb_run.log({
                f"{split}_results": wandb.Table(
                    dataframe=pd.DataFrame(results)
                )
            })
        except Exception as e:
            print(f"Warning: Failed to log to wandb: {e}")


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


def main():
    start_time = time.time()

    # Load configuration
    config = load_config('config.yaml')

    # Create run directory
    run_dir = create_run_directory(config)
    print(f"Created run directory: {run_dir}")

    # Initialize wandb for all splits
    wandb_run = init_wandb("inference", config)
    if wandb_run is None:
        print("Warning: Running without wandb logging")

    try:
        # Initialize VLLM inference
        vllm_config = VLLMConfig(
            model_name=config['vllm']['model'],
            max_tokens=config['vllm']['max_tokens'],
            temperature=config['vllm']['temperature'],
            vllm_url=config['vllm']['url']
        )
        vllm = VLLMInference(config=vllm_config)

        # Load dataset
        dataset = SciVQADataset.from_huggingface(
            split=config['dataset']['split'],
            data_dir=config['dataset']['data_dir']
        )
        print(f"Loaded {len(dataset)} examples")

        # Process examples
        start_idx = config['dataset']['start_idx']
        end_idx = config['dataset']['end_idx'] if config['dataset']['end_idx'] is not None else len(
            dataset)
        batch_size = config['vllm']['batch_size']
        results = []

        print(
            f"\nProcessing examples from {start_idx} to {end_idx} with batch size {batch_size}")
        n_batches = (end_idx - start_idx + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
            batch_start = start_idx + batch_idx * batch_size
            batch_end = min(batch_start + batch_size, end_idx)
            batch = dataset.get_batch(batch_start, batch_size)

            if len(batch) == 0:
                print(
                    f"WARNING: Batch {batch_idx+1}/{n_batches} contains no valid items. Skipping.")
                continue

            print(
                f"\nProcessing batch {batch_idx+1}/{n_batches} with {len(batch)} examples")

            # Create prompts for batch
            image_paths = [example['image_path'] for example in batch]

            # Run inference
            outputs = vllm.cot_batch_infer(batch, image_paths)
            import re

            def remove_all_tags(text):
                pattern = r'<\|?[a-zA-Z0-9_\s]*\|?>'
                return re.sub(pattern, '', text)

            batch_results = []
            for example, output in zip(batch, outputs):
                if output and 'question_analysis' in output and 'qa_type_analysis' in output and 'final_answer' in output:
                    # Get initial analysis
                    question_analysis = output['question_analysis']['choices'][0]['message']['content'].strip(
                    )
                    # Get final answer
                    qa_type_analysis = output['qa_type_analysis']['choices'][0]['message']['content'].strip(
                    )
                    if example['choices'] is None:
                        final_answer = output['final_answer']['choices'][0]['message']['content'].strip(
                        )
                    else:
                        final_answer = qa_type_analysis

                    final_answer = remove_all_tags(final_answer)
                    if should_override_with_unanswerable(final_answer):
                        final_answer = "It is not possible to answer this question based only on the provided data."

                    # Create result dictionary based on split
                    result = {
                        'id': example.get('id', ''),
                        'question': example['question'],
                        'question_analysis': question_analysis,
                        'qa_pair_type_analysis': qa_type_analysis,
                        'final_answer': final_answer,
                        'qa_pair_type': example['qa_pair_type'],
                        'fig_type': example['figure_type']
                    }

                    # Add ground truth and correctness only for train/validation
                    if config['dataset']['split'] != 'test':
                        ground_truth = example['answer'].strip().lower()
                        generated_clean = final_answer.strip().lower()
                        is_correct = generated_clean == ground_truth
                        result.update({
                            'answer': example['answer'],
                            'correct': is_correct
                        })

                        # Print example details for train/validation
                        print(f"\nExample {len(results)}:")
                        print(f"Question: {example['question']}")
                        print(f"Question Analysis: {question_analysis}")
                        print(f"QA Type Analysis: {qa_type_analysis}")
                        print(f"Final Answer: {final_answer}")
                        print(f"Ground Truth: {example['answer']}")
                        print(f"Correct: {'✅' if is_correct else '❌'}")
                    else:
                        # Print only question and responses for test
                        print(f"\nExample {len(results)}:")
                        print(f"Question: {example['question']}")
                        print(f"Question Analysis: {question_analysis}")
                        print(f"QA Type Analysis: {qa_type_analysis}")
                        print(f"Final Answer: {final_answer}")

                    batch_results.append(result)
                    results.append(result)

                else:
                    print(
                        f"Failed to get response for example {example.get('id', 'unknown')}")

            # Log progress
            progress = (batch_end - start_idx) / (end_idx - start_idx) * 100
            print(
                f"\nProgress: {progress:.1f}% ({batch_end - start_idx}/{end_idx - start_idx} examples)")

            # Log batch results to wandb
            if wandb_run and batch_results:
                try:
                    # Log individual examples
                    for result in batch_results:
                        wandb_run.log({
                            "example": wandb.Table(
                                dataframe=pd.DataFrame([result])
                            )
                        })

                    # Log batch metrics
                    wandb_run.log({
                        "progress": progress,
                        "examples_processed": batch_end - start_idx
                    })

                    # Log additional metrics for train/validation
                    if config['dataset']['split'] != 'test':
                        batch_accuracy = sum(r['correct']
                                             for r in batch_results) / len(batch_results)
                        wandb_run.log({
                            "batch_accuracy": batch_accuracy,
                            "correct": sum(r['correct'] for r in batch_results)
                        })
                except Exception as e:
                    print(
                        f"Warning: Failed to log batch results to wandb: {e}")

            # Save intermediate results
            if len(results) % 5 == 0:
                save_results(results, config,
                             config['dataset']['split'], run_dir, wandb_run)
                print(
                    f"\nIntermediate results saved to {run_dir}/predictions/{config['dataset']['split']}_predictions.json")

        # Save final results
        save_results(results, config,
                     config['dataset']['split'], run_dir, wandb_run)

        # Calculate and log final metrics
        print(f"\nFinal Results:")
        print(f"Total examples processed: {len(results)}")

        if wandb_run:
            try:
                wandb_run.log({
                    "total_examples": len(results),
                    "completion_time": time.time() - start_time
                })

                # Log additional metrics for train/validation
                if config['dataset']['split'] != 'test':
                    total_accuracy = sum(
                        r['correct'] for r in results) / len(results) if results else 0
                    print(f"Overall accuracy: {total_accuracy:.2%}")
                    wandb_run.log({
                        "final_accuracy": total_accuracy
                    })
            except Exception as e:
                print(f"Warning: Failed to log final metrics to wandb: {e}")

    finally:
        # Close wandb
        if wandb_run:
            try:
                wandb_run.finish()
                print("\nWandb run completed and logged")
            except Exception as e:
                print(f"Warning: Failed to finish wandb run: {e}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
import json
import pandas as pd
from typing import List, Dict, Any
from evaluate import load
from rouge_score import rouge_scorer
from wandb_config import init_wandb


def load_results(file_path: str) -> pd.DataFrame:
    """Load results from JSON or CSV file.

    Args:
        file_path: Path to results file

    Returns:
        DataFrame containing results
    """
    file_path = Path(file_path)
    if file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    elif file_path.suffix == '.csv':
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def calculate_rouge(predictions: List[str], references: List[str], r_type: str = "") -> Dict[str, float]:
    """Calculate ROUGE scores.

    Args:
        predictions: List of predicted answers
        references: List of reference answers
        r_type: ROUGE type (rouge1, rouge2, rougeL)

    Returns:
        Dictionary containing precision, recall, and F1 scores
    """
    precision = []
    recall = []
    f1 = []
    scorer = rouge_scorer.RougeScorer([r_type], use_stemmer=True)

    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)
        precision.append(score[r_type].precision)
        recall.append(score[r_type].recall)
        f1.append(score[r_type].fmeasure)

    return {
        'precision': sum(precision) / len(precision),
        'recall': sum(recall) / len(recall),
        'f1': sum(f1) / len(f1)
    }


def calculate_bertscore(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate BERTScore.

    Args:
        predictions: List of predicted answers
        references: List of reference answers

    Returns:
        Dictionary containing precision, recall, and F1 scores
    """
    bertscore = load("bertscore")
    results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en"
    )

    return {
        'precision': sum(results["precision"]) / len(results["precision"]),
        'recall': sum(results["recall"]) / len(results["recall"]),
        'f1': sum(results["f1"]) / len(results["f1"])
    }


def evaluate_results(results_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Evaluate results using ROUGE and BERTScore.

    Args:
        results_df: DataFrame containing results with 'answer' and 'response' columns

    Returns:
        Dictionary containing all evaluation metrics
    """
    # Convert to string and get lists
    references = results_df["answer"].astype(str).tolist()
    predictions = results_df["response"].astype(str).tolist()

    # Calculate metrics
    metrics = {
        'rouge1': calculate_rouge(predictions, references, "rouge1"),
        'rougeL': calculate_rouge(predictions, references, "rougeL"),
        'bertscore': calculate_bertscore(predictions, references)
    }

    return metrics


def save_metrics(metrics: Dict[str, Dict[str, float]], output_path: str):
    """Save evaluation metrics to a file and log to wandb.

    Args:
        metrics: Dictionary containing evaluation metrics
        output_path: Path to save metrics
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to file
    with open(output_path, 'w') as f:
        for metric_name, scores in metrics.items():
            f.write(f"\n{metric_name}:\n")
            for score_name, value in scores.items():
                f.write(f"{score_name}: {value:.4f}\n")
                print(f"{metric_name}.{score_name}: {value:.4f}")

    # Log to wandb
    for metric_name, scores in metrics.items():
        for score_name, value in scores.items():
            wandb.log({
                f"evaluation/{metric_name}/{score_name}": value
            })


def main():
    parser = argparse.ArgumentParser(description='Evaluate inference results')
    parser.add_argument('--results-file', type=str, required=True,
                        help='Path to results file (JSON or CSV)')
    parser.add_argument('--output-file', type=str, default='evaluation_metrics.txt',
                        help='Path to save evaluation metrics')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val', 'test'],
                        help='Dataset split being evaluated')

    args = parser.parse_args()

    # Initialize wandb
    wandb = init_wandb("evaluation", {
        "split": args.split,
        "results_file": args.results_file
    })

    try:
        # Load results
        print(f"Loading results from {args.results_file}...")
        results_df = load_results(args.results_file)

        # Log examples to wandb
        wandb.log({
            f"{args.split}_examples": wandb.Table(
                dataframe=results_df
            )
        })

        # Evaluate results
        print("Calculating metrics...")
        metrics = evaluate_results(results_df)

        # Save metrics
        print(f"Saving metrics to {args.output_file}...")
        save_metrics(metrics, args.output_file)

    finally:
        # Close wandb
        wandb.finish()


if __name__ == "__main__":
    main()

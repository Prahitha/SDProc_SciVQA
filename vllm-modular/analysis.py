import argparse
from pathlib import Path
import json
import pandas as pd
from typing import List, Dict, Any, Tuple
from evaluate import load
from rouge_score import rouge_scorer
from prompts import QAPairType, FigureType
from evaluate import calculate_rouge, calculate_bertscore
from wandb_config import init_wandb
import wandb


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


def get_metrics_by_category(
    results_df: pd.DataFrame,
    category_column: str
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Calculate metrics for each category.

    Args:
        results_df: DataFrame containing results
        category_column: Column name to group by (qa_pair_type or fig_type)

    Returns:
        Dictionary containing metrics for each category
    """
    metrics_by_category = {}

    for category in results_df[category_column].unique():
        # Filter results for this category
        category_df = results_df[results_df[category_column] == category]

        # Get predictions and references
        references = category_df["answer"].astype(str).tolist()
        predictions = category_df["response"].astype(str).tolist()

        # Calculate metrics
        metrics = {
            'rouge1': calculate_rouge(predictions, references, "rouge1"),
            'rougeL': calculate_rouge(predictions, references, "rougeL"),
            'bertscore': calculate_bertscore(predictions, references)
        }

        metrics_by_category[category] = metrics

    return metrics_by_category


def analyze_results(results_df: pd.DataFrame, split: str) -> Tuple[Dict, Dict]:
    """Analyze results by QA pair type and figure type.

    Args:
        results_df: DataFrame containing results
        split: Dataset split (train/val/test)

    Returns:
        Tuple of (metrics_by_qa_type, metrics_by_fig_type)
    """
    # Ensure required columns exist
    required_columns = ['answer', 'response',
                        'qa_pair_type', 'fig_type', 'question']
    missing_columns = [
        col for col in required_columns if col not in results_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Calculate metrics by QA pair type
    metrics_by_qa_type = get_metrics_by_category(results_df, 'qa_pair_type')

    # Calculate metrics by figure type
    metrics_by_fig_type = get_metrics_by_category(results_df, 'fig_type')

    # Log examples to wandb
    wandb.log({
        f"{split}_examples": wandb.Table(
            dataframe=results_df
        )
    })

    # Log metrics to wandb
    for qa_type, metrics in metrics_by_qa_type.items():
        for metric_name, scores in metrics.items():
            for score_name, value in scores.items():
                wandb.log({
                    f"{split}/qa_type/{qa_type}/{metric_name}/{score_name}": value
                })

    for fig_type, metrics in metrics_by_fig_type.items():
        for metric_name, scores in metrics.items():
            for score_name, value in scores.items():
                wandb.log({
                    f"{split}/fig_type/{fig_type}/{metric_name}/{score_name}": value
                })

    return metrics_by_qa_type, metrics_by_fig_type


def save_analysis(
    metrics_by_qa_type: Dict[str, Dict[str, Dict[str, float]]],
    metrics_by_fig_type: Dict[str, Dict[str, Dict[str, float]]],
    output_path: str
):
    """Save analysis results to a file.

    Args:
        metrics_by_qa_type: Metrics grouped by QA pair type
        metrics_by_fig_type: Metrics grouped by figure type
        output_path: Path to save analysis
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        # Write QA pair type analysis
        f.write("Analysis by QA Pair Type:\n")
        f.write("=" * 50 + "\n")
        for qa_type, metrics in metrics_by_qa_type.items():
            f.write(f"\n{qa_type}:\n")
            f.write("-" * 30 + "\n")
            for metric_name, scores in metrics.items():
                f.write(f"\n{metric_name}:\n")
                for score_name, value in scores.items():
                    f.write(f"{score_name}: {value:.4f}\n")
                    print(f"{qa_type}.{metric_name}.{score_name}: {value:.4f}")

        # Write figure type analysis
        f.write("\n\nAnalysis by Figure Type:\n")
        f.write("=" * 50 + "\n")
        for fig_type, metrics in metrics_by_fig_type.items():
            f.write(f"\n{fig_type}:\n")
            f.write("-" * 30 + "\n")
            for metric_name, scores in metrics.items():
                f.write(f"\n{metric_name}:\n")
                for score_name, value in scores.items():
                    f.write(f"{score_name}: {value:.4f}\n")
                    print(f"{fig_type}.{metric_name}.{score_name}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze inference results by QA pair type and figure type')
    parser.add_argument('--results-file', type=str, required=True,
                        help='Path to results file (JSON or CSV)')
    parser.add_argument('--output-file', type=str, default='analysis_results.txt',
                        help='Path to save analysis results')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val', 'test'],
                        help='Dataset split being analyzed')

    args = parser.parse_args()

    # Initialize wandb
    wandb = init_wandb("analysis", {
        "split": args.split,
        "results_file": args.results_file
    })

    try:
        # Load results
        print(f"Loading results from {args.results_file}...")
        results_df = load_results(args.results_file)

        # Analyze results
        print("Analyzing results...")
        metrics_by_qa_type, metrics_by_fig_type = analyze_results(
            results_df, args.split)

        # Save analysis
        print(f"Saving analysis to {args.output_file}...")
        save_analysis(metrics_by_qa_type,
                      metrics_by_fig_type, args.output_file)

    finally:
        # Close wandb
        wandb.finish()


if __name__ == "__main__":
    main()

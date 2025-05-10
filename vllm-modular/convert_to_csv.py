import json
import pandas as pd
from pathlib import Path
import argparse


def convert_predictions_to_csv(json_path: str, output_path: str):
    """
    Convert JSON predictions to CSV format with instance_id and answer_pred columns.

    Args:
        json_path (str): Path to the JSON predictions file
        output_path (str): Path to save the CSV file
    """
    # Read JSON file
    with open(json_path, 'r') as f:
        predictions = json.load(f)

    # Convert to DataFrame
    df = pd.DataFrame(predictions)

    # Rename columns
    df = df.rename(columns={
        'id': 'instance_id',
        'response': 'answer_pred'
    })

    # Select only required columns
    df = df[['instance_id', 'answer_pred']]

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Converted predictions saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert JSON predictions to CSV format')
    parser.add_argument('--json_path', type=str, required=True,
                        help='Path to JSON predictions file')
    parser.add_argument('--output_path', type=str,
                        default='predictions.csv', help='Path to save CSV file')

    args = parser.parse_args()

    convert_predictions_to_csv(args.json_path, args.output_path)


if __name__ == "__main__":
    main()

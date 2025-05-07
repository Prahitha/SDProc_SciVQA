import json
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from pydantic import BaseModel, model_validator, HttpUrl, Field
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset


class SciVQASample(BaseModel):
    """Base model for a single SciVQA sample with validation."""
    instance_id: str = Field(default_factory=lambda: f"id-{id(object())}")
    image_file: str
    figure_id: str = ""
    caption: str = ""
    figure_type: str = ""
    compound: bool = False
    figs_numb: str = ""
    qa_pair_type: str = ""
    question: str
    answer: Optional[str] = None
    answer_options: dict = {}
    venue: str = ""
    categories: str = ""
    source_dataset: str = ""
    paper_id: str = ""
    pdf_url: HttpUrl = "https://example.com"
    split: Optional[str] = None  # Track which split this sample belongs to

    @model_validator(mode="before")
    def merge_answer_options(cls, values):
        """Merge answer options from list format to dictionary format.

        Example input:
        [
            {"A": "2"},
            {"B": "1"},
            {"C": "3"},
            {"D": "All of the above"}
        ]

        Example output:
        {
            "A": "2",
            "B": "1",
            "C": "3",
            "D": "All of the above"
        }
        """
        options = values.get('answer_options')
        if isinstance(options, list):
            merged = {}
            for item in options:
                if isinstance(item, dict):
                    for key, value in item.items():
                        if value is not None:
                            merged[key] = value
            values['answer_options'] = merged
        return values

    def to_dict(self) -> Dict[str, Any]:
        """Convert the sample to a dictionary format suitable for model input."""
        return {
            'id': self.instance_id,
            'image_path': self.image_file,
            'question': self.question,
            'caption': self.caption,
            'answer': self.answer if self.answer else "",
            'choices': self.answer_options,
            'qa_pair_type': self.qa_pair_type,
            'figure_type': self.figure_type,
            'split': self.split,
            'compound': self.compound,
            'figs_numb': self.figs_numb
        }


class SciVQADataset:
    """Dataset manager for SciVQA data with loading and batching capabilities."""

    def __init__(self, data_dir: Union[str, Path] = "./scivqa_data", split: str = "validation"):
        print(
            f"Initializing dataset with data_dir: {data_dir} and split: {split}")
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_folder = self.data_dir / f"images_{split}"
        self.samples: List[SciVQASample] = []

    @classmethod
    def from_huggingface(cls, split: Optional[str] = None, data_size: Optional[int] = None,
                         data_dir: Union[str, Path] = "./scivqa_data") -> 'SciVQADataset':
        """Create a dataset instance by loading from HuggingFace.

        Args:
            split: Optional split name. If None, loads all splits.
            data_size: Optional size limit for the dataset.
            data_dir: Directory to store the dataset.
        """
        print(f"Loading dataset from HuggingFace...")
        dataset = cls(data_dir=data_dir, split=split)

        try:
            # Load the dataset using HuggingFace
            raw_dataset = load_dataset("katebor/SciVQA", split=split)
            print(
                f"Raw dataset loaded with {len(raw_dataset)} examples for split '{split}'")

            if data_size is not None:
                raw_dataset = raw_dataset.select(
                    range(min(data_size, len(raw_dataset))))
            dataset._process_split(raw_dataset, split)

        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Detailed traceback:")
            import traceback
            traceback.print_exc()

        return dataset

    def _process_split(self, raw_data, split_name: str):
        """Process a single split of the dataset."""
        validated_data = []
        for i, item in enumerate(tqdm(raw_data, desc=f"Validating {split_name} data")):
            try:
                item_dict = dict(item)
                item_dict['split'] = split_name
                sample = SciVQASample(**item_dict)
                validated_data.append(sample)
            except Exception as e:
                print(f"Error validating item {i} in split {split_name}: {e}")

        self.samples.extend(validated_data)
        print(
            f"Successfully loaded and validated {len(validated_data)} items for split '{split_name}'")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample by index with resolved image path."""
        try:
            sample = self.samples[idx]
            result = sample.to_dict()

            # Resolve image path
            image_file_name = sample.image_file

            # Try split-specific folder first, then fall back to general image folder
            image_path = self.image_folder / image_file_name

            if image_path is None:
                print(
                    f"WARNING: Image file not found for {image_file_name}. Tried paths: {image_path}")

            result['image_path'] = str(image_path) if image_path else None
            return result

        except Exception as e:
            print(f"Error getting item {idx}: {str(e)}")
            return {
                'id': str(idx),
                'image_path': None,
                'question': '',
                'caption': '',
                'answer': '',
                'choices': [],
                'qa_pair_type': '',
                'figure_type': '',
                'split': None
            }

    def get_batch(self, start_idx: int, batch_size: int) -> List[Dict[str, Any]]:
        """Get a batch of samples starting from start_idx with size batch_size."""
        end_idx = min(start_idx + batch_size, len(self))
        batch = []
        for i in range(start_idx, end_idx):
            item = self[i]
            print(item)
            if item['question'] and item['image_path']:
                batch.append(item)
            else:
                print(f"Skipping invalid item at index {i}")
        return batch

    @classmethod
    def load_samples(cls, filepath: Union[str, Path], data_dir: Union[str, Path] = "./scivqa_data",
                     split: Optional[str] = None) -> 'SciVQADataset':
        """Load samples from a JSON file."""
        dataset = cls(data_dir=data_dir, split=split)
        with open(filepath, 'r') as f:
            data = json.load(f)
        dataset.samples = [SciVQASample(**item) for item in data]
        return dataset

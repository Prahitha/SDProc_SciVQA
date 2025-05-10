import os
import base64
import requests
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
from dataclasses import dataclass
from prompts import COTPromptCreator, PromptCreator


@dataclass
class VLLMConfig:
    """Configuration for VLLM inference."""
    model_name: str = "microsoft/Phi-4-multimodal-instruct"
    max_tokens: int = 150
    temperature: float = 0.0
    vllm_url: str = "http://localhost:8000/v1/chat/completions"


class VLLMInference:
    """Class to handle VLLM inference for Phi-4 multimodal model."""

    def __init__(self, config: Optional[VLLMConfig] = None):
        """Initialize the VLLM inference handler.

        Args:
            config: Optional configuration for VLLM inference. If None, uses default config.
        """
        self.config = config or VLLMConfig()
        self.headers = {"Content-Type": "application/json"}
        self.cot_creator = COTPromptCreator()
        self.prompt_creator = PromptCreator()

    def _encode_image(self, image_path: Union[str, Path]) -> Optional[str]:
        """Encode image to base64 string.

        Args:
            image_path: Path to the image file.

        Returns:
            Base64 encoded string of the image if successful, None otherwise.
        """
        try:
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found at {image_path}")
                return None

            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None

    def _create_payload(self, prompt: str, image_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Create the request payload for VLLM API.

        Args:
            prompt: The text prompt for the model.
            image_path: Optional path to the image file.

        Returns:
            Dictionary containing the request payload.
        """
        if image_path:
            encoded_image = self._encode_image(image_path)
            if encoded_image:
                return {
                    #  "model": self.config.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{encoded_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature
                }

        # Fallback to text-only query
        print("Warning: No valid image data. Proceeding with text-only query.")
        return {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{prompt} (Note: Image unavailable)"
                        }
                    ]
                }
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }

    def infer(self, prompt: str, image_path: Optional[Union[str, Path]] = None) -> Optional[Dict[str, Any]]:
        """Perform inference using VLLM API.

        Args:
            prompt: The text prompt for the model.
            image_path: Optional path to the image file.

        Returns:
            Dictionary containing the model's response if successful, None otherwise.
        """
        try:
            payload = self._create_payload(prompt, image_path)
            response = requests.post(
                self.config.vllm_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error in API request: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error in inference: {e}")
            return None

    def batch_infer(self, prompts: list[str], image_paths: Optional[list[Union[str, Path]]] = None) -> list[Optional[Dict[str, Any]]]:
        """Perform batch inference using VLLM API.

        Args:
            prompts: List of text prompts.
            image_paths: Optional list of image paths. If provided, must match length of prompts.

        Returns:
            List of model responses, with None for failed inferences.
        """
        if image_paths and len(prompts) != len(image_paths):
            raise ValueError(
                "Number of prompts must match number of image paths")

        results = []
        for i, prompt in enumerate(prompts):
            image_path = image_paths[i] if image_paths else None
            result = self.infer(prompt, image_path)
            results.append(result)
        return results

    def cot_infer(self, example: Dict[str, Any], image_path: Optional[Union[str, Path]] = None) -> Optional[Dict[str, Any]]:
        """Perform Chain of Thought inference with two-step prompting.

        Args:
            example: Dictionary containing the example data (caption, question, etc.)
            image_path: Optional path to the image file.

        Returns:
            Dictionary containing the model's final response if successful, None otherwise.
        """
        try:
            # Create the two-step prompts
            question_analysis_prompt, qa_type_analysis_prompt, answer_prompt = self.cot_creator.create_prompt(
                example)

            # Step 1: Initial Analysis
            question_analysis_response = self.infer(
                question_analysis_prompt, image_path)
            if not question_analysis_response:
                print("Failed to get question analysis response")
                return None

            # Step 2: Detailed Analysis
            # Combine initial response with detailed prompt
            combined_prompt = f"{question_analysis_response}\n\n{qa_type_analysis_prompt}"
            qa_type_analysis_response = self.infer(combined_prompt, image_path)

            # Step 3: Final Answer
            combined_prompt = f"{qa_type_analysis_response}\n\n{answer_prompt}"
            final_response = self.infer(combined_prompt, image_path)

            if not final_response:
                print("Failed to get final answer response")
                return None

            return {
                'question_analysis': question_analysis_response,
                'qa_type_analysis': qa_type_analysis_response,
                'final_answer': final_response
            }

        except Exception as e:
            print(f"Error in COT inference: {e}")
            return None

    def cot_batch_infer(self, examples: List[Dict[str, Any]], image_paths: Optional[List[Union[str, Path]]] = None) -> List[Optional[Dict[str, Any]]]:
        """Perform batch Chain of Thought inference with two-step prompting.

        Args:
            examples: List of dictionaries containing example data
            image_paths: Optional list of image paths. If provided, must match length of examples.

        Returns:
            List of dictionaries containing model responses, with None for failed inferences.
        """
        if image_paths and len(examples) != len(image_paths):
            raise ValueError(
                "Number of examples must match number of image paths")

        results = []
        for i, example in enumerate(examples):
            image_path = image_paths[i] if image_paths else None
            result = self.cot_infer(example, image_path)
            results.append(result)
        return results

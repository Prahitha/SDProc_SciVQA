from typing import Dict, Any, List, Optional
from enum import Enum


class QAPairType(str, Enum):
    """Enum for different types of QA pairs."""
    # Answer Possibility
    CLOSED_ENDED = "closed-ended"
    UNANSWERABLE = "unanswerable"

    # Answer Set Type
    INFINITE_ANSWER_SET = "infinite_answer_set"
    FINITE_ANSWER_SET = "finite_answer_set"

    # Finite Answer Set Subtypes
    BINARY = "binary"
    NON_BINARY = "non-binary"

    # Visual Aspects
    VISUAL = "visual"
    NON_VISUAL = "non-visual"


class FigureType(str, Enum):
    """Enum for different types of figures."""
    COMPOUND = "compound"  # Multiple subfigures
    NON_COMPOUND = "non-compound"  # Single figure

    # Specific figure types
    LINE_CHART = "line chart"
    BAR_CHART = "bar chart"
    BOX_PLOT = "box plot"
    CONFUSION_MATRIX = "confusion matrix"
    PIE_CHART = "pie chart"


class PromptCreator:
    """Class to create prompts for different QA pair types."""

    def __init__(self):
        """Initialize the prompt creator."""
        self.base_instruction = "Answer the question based on the information in the image, caption and the context provided. Do not hallucinate or infer information from general knowledge. Provide only the direct answer without any explanation or reasoning."
        self.base_instruction = "Answer the question with only the exact value or fact, without any sentences, explanations, or additional text."
        self.base_instruction = "Answer the question with only the raw value (number, word, or phrase) without any units, explanations, or complete sentences."
        self.base_instruction = "Answer the question with only the raw numerical value or single word/phrase, omitting all units, context words, and explanatory text, remove <|end|> tag in the end answer."

    def _format_choices(self, choices: Dict[str, str]) -> List[str]:
        """Format choices into a readable format."""
        return [f"Option {k}: {v}" for k, v in choices.items()]

    def _get_figure_type_instruction(self, figure_type: str) -> str:
        """Get specific instructions based on figure type."""
        figure_type = figure_type.lower()

        if figure_type == FigureType.LINE_CHART:
            return (
                "Focus on the following aspects of the line chart:\n"
                "- Colors of different lines and their meanings\n"
                "- X and Y axis labels and their units\n"
                "- Scale and range of values\n"
                "- Trends and patterns in the lines\n"
            )
        elif figure_type in FigureType.BAR_CHART:
            return (
                "Focus on the following aspects of the bar chart:\n"
                "- Colors of different bars and their meanings\n"
                "- X and Y axis labels and their units\n"
                "- Scale and range of values\n"
                "- Height and position of bars\n"
            )
        elif figure_type in FigureType.BOX_PLOT:
            return (
                "Focus on the following aspects of the box plot:\n"
                "- Median line position\n"
                "- Box boundaries (Q1 and Q3)\n"
                "- Whisker extent\n"
                "- Outliers if present\n"
            )
        elif figure_type in FigureType.CONFUSION_MATRIX:
            return (
                "Focus on the following aspects of the confusion matrix:\n"
                "- Row and column labels\n"
                "- Numerical values in each cell\n"
                "- Color intensity if present\n"
                "- Overall distribution of values\n"
            )
        elif figure_type in FigureType.PIE_CHART:
            return (
                "Focus on the following aspects of the pie chart:\n"
                "- Segments and their labels\n"
                "- Percentage or proportion values\n"
                "- Colors of different segments\n"
                "- Size of each segment relative to others\n"
            )
        return (
            "Focus on the following aspects of the figure:\n"
            "- Colors and the labels present in the figure\n"
            "- Any other relevant information present in the figure\n"
        )
    
    def _get_binary_instruction(self, question: str, qa_pair_type: str = "") -> str:
        """Get appropriate binary instruction based on question type."""
        question_lower = question.lower()
    
    # Determine if it's True/False or Yes/No
        if any(phrase in question_lower for phrase in ['is it true', 'is it false', 'is this true', 'is this false']):
            answer_format = "Answer with either 'True' or 'False'"
            evidence_type = "visual" if "visual" in qa_pair_type else "textual"
        else:
            answer_format = "Answer with either 'Yes' or 'No'"
            evidence_type = "visual" if "visual" in qa_pair_type else "textual"
        
    # Return the complete instruction
        return f"This is a binary question. {answer_format} based on the {evidence_type} evidence. Respond affirmatively if the statement is supported by the evidence."


    def _get_compound_navigation(self, example: Dict[str, Any]) -> str:
        """Get navigation instructions for compound figures."""
        fig_numb = example.get('fig_numb', 1)

        return (
            "This is a compound figure containing multiple subfigures. "
            f"Navigate to {fig_numb} graph in the compound figure to answer the question."
        )

    def _create_base_prompt(self, example: Dict[str, Any]) -> List[str]:
        """Create the base prompt parts."""
        prompt_parts = []

        # Add caption if available
        if example.get('caption'):
            prompt_parts.append(
                "\nThe caption of the figure is mentioned as,\n"
                f"{example['caption']} \n"
            )

        # Handle compound figures
        if example.get('compound'):
            prompt_parts.append(self._get_compound_navigation(example))

        # Add figure type information and specific instructions
        figure_type = example.get('figure_type', '').lower()
        prompt_parts.append(
            f"This is a {figure_type.replace('_', ' ')}.")
        type_instruction = self._get_figure_type_instruction(
            figure_type)
        if type_instruction:
            prompt_parts.append(type_instruction)

        # Add question
        prompt_parts.append(
            f"{example['question']}"
        )

        # Add choices if available
        if example.get('choices'):
            prompt_parts.extend(self._format_choices(example['choices']))

        return prompt_parts

    def _get1_binary_instruction(self, question: str) -> str:
        """Get appropriate binary instruction based on question type."""
        question_lower = question.lower()

        # Check for True/False type questions
        if any(phrase in question_lower for phrase in ['is it true', 'is it false', 'is this true', 'is this false']):
            return "This is a true/false question. Answer with either 'True' or 'False'"

        # Check for Yes/No type questions
        if any(phrase in question_lower for phrase in ['is there', 'are there', 'does', 'do', 'is', 'are', 'can', 'could', 'would', 'should']):
            return "This is a yes/no question. Answer with either 'Yes' or 'No'"

        # Default to Yes/No if question starts with a verb
        if question_lower.split()[0] in ['is', 'are', 'does', 'do', 'can', 'could', 'would', 'should']:
            return "This is a yes/no question. Answer with either 'Yes' or 'No'"

        # Default to True/False for other binary questions
        return "This is a true/false question. Answer with either 'True' or 'False'."

    def _create_answer_instruction(self, example: Dict[str, Any]) -> str:
        """Create the answer instruction based on QA pair type."""
        qa_pair_type = example.get('qa_pair_type', '').lower()
# Handle non-binary multiple choice
        if "non-binary" in qa_pair_type and example.get('choices'):
            return (
                #"Choose one or more correct answers from the provided options. "
                #"Return only the corresponding letter(s) of the correct answer(s). "
                #"If multiple letters are correct, separate them by commas without spaces (e.g., B,C)."
                "This is a multiple-choice question. First, analyze the evidence to determine the correct answer(s). "
                "If multiple letters are correct, separate them by commas without spaces (e.g., B,C)."
                "Respond with ONLY the letter(s) corresponding to the correct option(s). "
                "Be precise in your selection - include only letters that are definitively correct based on the evidence shown. "
                "Verify each option individually against the figure before including it in your answer."
                )

        # Handle binary questions
        elif "binary" in qa_pair_type:
            return self._get_binary_instruction(example['question'], qa_pair_type)

        # Handle infinite answer set
        if "infinite_answer_set" in qa_pair_type:
            return "Answer precisely with numerical values or facts. Consider chart scale for math questions. Provide a concise answer with no extra explanation."

        # Handle visual questions
        if "non-visual" in qa_pair_type:
            return "Answer based on the information provided in the text. Provide a concise answer with no extra explanation."
        elif "visual" in qa_pair_type:
            return "Focus on the visual aspects (shape, size, position, height, direction, or color) in your answer. Provide a concise answer with no extra explanation."
        
        if "unanswerable" in qa_pair_type:
            return "The question is not answerable based on the information provided."
        

        # Default instruction
        return "Provide a direct answer based on the information available. Provide a concise answer with no extra explanation."

    def create_prompt(self, example: Dict[str, Any]) -> str:
        """Create a prompt based on the example's QA pair type and figure type."""
        prompt_parts = self._create_base_prompt(example)
        instruction = self._create_answer_instruction(example)
        prompt_parts.append(f"{self.base_instruction} {instruction}")

        return "\n".join(prompt_parts)

    def create_batch_prompts(self, examples: List[Dict[str, Any]]) -> List[str]:
        """Create prompts for a batch of examples."""
        return [self.create_prompt(example) for example in examples]

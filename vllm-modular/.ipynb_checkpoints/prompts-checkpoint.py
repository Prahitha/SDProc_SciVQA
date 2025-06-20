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
        self.base_instruction = "Answer the question with only the raw numerical value or single word/phrase, omitting all units, context words, and explanatory text"

    def _format_choices(self, choices: Dict[str, str]) -> List[str]:
        """Format choices into a readable format."""
        return [f"{k}: {v}" for k, v in choices.items()]

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
            prompt_parts.append("Return only the corresponding letter(s) of the correct answer(s). "
                "Only output the letter(s) corresponding to the correct choice. "
                "If multiple letters are correct, separate them by commas without spaces (for example: B,C).")
            

        return prompt_parts

    def create_prompt(self, example: Dict[str, Any]) -> str:
        """Create a prompt based on the example's QA pair type and figure type."""
        prompt_parts = []
        # prompt_parts.append(f"{self.base_instruction}")
        # prompt_parts.extend(self._create_base_prompt(example))

        prompt_parts.append(f"Answer the question with only the raw numerical value or single word/phrase, omitting all units, context words, and explanatory text. The caption of the figure is mentioned as, {example["caption"]}. The question for the figure is, \n{example["question"]}.")
        

        return "\n".join(prompt_parts)

    def create_batch_prompts(self, examples: List[Dict[str, Any]]) -> List[str]:
        """Create prompts for a batch of examples."""
        return [self.create_prompt(example) for example in examples]


class COTPromptCreator:
    """Class to create Chain of Thought prompts with streamlined reasoning."""

    def __init__(self):
        """Initialize the COT prompt creator."""
        self.base_instruction = (
            "Answer the question with only the raw numerical value or single word/phrase, omitting all units, context words, and explanatory text. Approximations in the scale are allowed.")

    def _format_choices(self, choices: Dict[str, str]) -> List[str]:
        """Format choices into a readable format."""
        return [f"{k}: {v}" for k, v in choices.items()]

    def _get_binary_instruction(self, question: str, qa_pair_type: str = "") -> str:
        """Get appropriate binary instruction based on question type."""

        if qa_pair_type == QAPairType.BINARY:
            question_lower = question.lower()
            answer_format = "True or False" if any(phrase in question_lower for phrase in [
                'is it true', 'is it false', 'is this true', 'is this false']) else "\"Yes\" or \"No\""
            evidence_type = "visual" if "visual" in qa_pair_type else "textual"

            return (
                f"This is a binary question requiring a {answer_format} answer based on {evidence_type} evidence.\n"
                "1. Identify the key elements in the question\n"
                "2. Examine the evidence\n"
                f"3. Provide your {answer_format} answer\n"
            )
        else:
            return ""

    def _get_compound_navigation(self, example: Dict[str, Any]) -> str:
        """Get navigation instructions for compound figures."""
        fig_numb = example.get('fig_numb', 1)
        return (
            f"Navigate to the {fig_numb} graph in the compound figure:\n"
            "1. Locate the correct subfigure\n"
            "2. Proceed with analysis\n"
        )

    def create_initial_analysis_prompt(self, example: Dict[str, Any]) -> str:
        """Create an initial analysis prompt focusing on caption, question, and image analysis."""
        prompt_parts = []

        # Step 1: Initial Analysis
        prompt_parts.append("STEP 1: INITIAL ANALYSIS")
        prompt_parts.append(
            "Given the figure, caption, and question, analyze and answer step by step. "
            "Regularly perform self-questioning, self-verification, self-correction to check your ongoing reasoning, using connectives such as "
            "\"Wait a moment\", \"Wait, does it seem right?\" etc."
        )

        # Combined Analysis
        analysis_parts = []

        # Figure Type Analysis
        # figure_type = example.get('figure_type', '').lower()
        # analysis_parts.append(
        #     f"Figure Type: {figure_type.replace('_', ' ')}\n"
        #     # "1. What type of visualization is this?\n"
        #     # "2. What are the key elements?\n"
        # )

        # Caption Analysis
        if example.get('caption'):
            analysis_parts.append(
                "\nCaption: \n"
                f"{example['caption']}\n"
                # "1. What is the main topic?\n"
                # "2. What key information is provided?\n"
            )

        # Question Analysis
        analysis_parts.append(
            "Question:\n"
            f"{example['question']}\n"
            # "1. What information is needed?\n"
            # "2. Where can we find it in the figure?\n"
        )

        # Integration Analysis
        # analysis_parts.append(
        #     "Integration Analysis:\n"
        #     "1. How do caption and question relate?\n"
        #     "2. Which parts of the figure are relevant?\n"
        # )

        analysis_parts.append(
            "Analyse the key visual elements (lines, shapes, colors) that address the question and analyze the relationships between elements. Then, extract the specific numerical/positional information from the figure and caption to answer the question."
        )
        prompt_parts.append("\n".join(analysis_parts))

        # Step 2: Compound Figure Navigation (if applicable)
        if example.get('compound'):
           # prompt_parts.append("\nSTEP 2: COMPOUND FIGURE NAVIGATION")
            prompt_parts.append(self._get_compound_navigation(example))

        return "\n".join(prompt_parts)

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

    def create_prompt(self, example: Dict[str, Any]) -> List[str]:
        """Create a complete Chain of Thought prompt based on the example."""
        # First create the initial analysis prompt
        initial_analysis = self.create_initial_analysis_prompt(example)

        instruction = ["\nSTEP 2: COT INFERENCE", self.base_instruction]
        figure_type = example.get('figure_type', '').lower()
        instruction.append(
            f"This is a {figure_type.replace('_', ' ')}.")
        type_instruction = self._get_figure_type_instruction(
            figure_type)
        if type_instruction:
            instruction.append(type_instruction)

        instruction.append(
            example['question'] + "\n" + self._get_binary_instruction(
                example['question'], example['qa_pair_type'])
        )

        # Add choices if available
        if example.get('choices') and len(example['choices']) > 0:
            instruction.append(
                f"Based on the reasoning above, match it to one or more of the provided answer options: {self._format_choices(example['choices'])}. "
                "Return only the corresponding letter(s) of the correct answer(s). "
                "Do not explain your choice, do not rephrase the answer, and do not repeat the option text. "
                "Only output the letter(s) corresponding to the correct choice. "
                "If multiple letters are correct, separate them by commas without spaces (for example: B,C). "
                "If all options are correct, return A,B,C,D. "
                "Do not add anything else."
            )

        instruction = "\n".join(instruction)

        # Combine both parts with the base instruction
        return [initial_analysis, instruction]

    def create_batch_prompts(self, examples: List[Dict[str, Any]]) -> List[str]:
        """Create Chain of Thought prompts for a batch of examples."""
        return [self.create_prompt(example) for example in examples]

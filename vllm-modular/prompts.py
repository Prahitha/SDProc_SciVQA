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


class COTPromptCreator:
    """Class to create Chain of Thought prompts with streamlined reasoning."""

    def __init__(self):
        """Initialize the COT prompt creator."""
        self.base_instruction = (
            "Answer the question with only the raw numerical value or single word/phrase, omitting all units, context words, and explanatory text."
        )

    def _get_figure_type_instruction(self, figure_type: str) -> str:
        """Get specific instructions based on figure type."""
        figure_type = figure_type.lower()

        if figure_type in FigureType.LINE_CHART:
            return (
                "Analyze this line chart:\n"
                "1. Identify the lines and their meanings\n"
                "2. Note the axes labels and units\n"
                "3. Analyze the trends and patterns\n"
            )
        elif figure_type in FigureType.BAR_CHART:
            return (
                "Analyze this bar chart:\n"
                "1. Identify the bars and their meanings\n"
                "2. Note the axes labels and units\n"
                "3. Compare the bar heights\n"
            )
        elif figure_type in FigureType.BOX_PLOT:
            return (
                "Analyze this box plot:\n"
                "1. Note the median line position\n"
                "2. Identify the box boundaries (Q1 and Q3)\n"
                "3. Check for outliers\n"
            )
        elif figure_type in FigureType.CONFUSION_MATRIX:
            return (
                "Analyze this confusion matrix:\n"
                "1. Note the row and column labels\n"
                "2. Examine the numerical values\n"
                "3. Identify the distribution pattern\n"
            )
        elif figure_type in FigureType.PIE_CHART:
            return (
                "Analyze this pie chart:\n"
                "1. Identify the segments and their labels\n"
                "2. Note the percentage values\n"
                "3. Compare segment sizes\n"
            )
        return (
            "Analyze this figure:\n"
            "1. Identify the key elements\n"
            "2. Note the relevant information\n"
            "3. Consider how it relates to the question\n"
        )

    def _get_binary_instruction(self, question: str, qa_pair_type: str = "") -> str:
        """Get appropriate binary instruction based on question type."""
        question_lower = question.lower()
        answer_format = "True or False" if any(phrase in question_lower for phrase in [
                                               'is it true', 'is it false', 'is this true', 'is this false']) else "Yes or No"
        evidence_type = "visual" if "visual" in qa_pair_type else "textual"

        return (
            f"This is a binary question requiring a {answer_format} answer based on {evidence_type} evidence.\n"
            "1. Identify the key elements in the question\n"
            "2. Examine the evidence\n"
            "3. Provide your {answer_format} answer\n"
        )

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
            "Given the figure, caption, and question, analyze and answer step by step.")

        # Combined Analysis
        analysis_parts = []

        # Figure Type Analysis
        figure_type = example.get('figure_type', '').lower()
        analysis_parts.append(
            f"Figure Type: {figure_type.replace('_', ' ')}\n"
            "1. Identify the visualization type and its key elements\n"
            "2. Note the main components and their relationships\n"
        )

        # Caption Analysis
        if example.get('caption'):
            analysis_parts.append(
                "Caption Analysis:\n"
                f"{example['caption']}\n"
                "1. Summarize the main topic and key information\n"
                "2. Note any specific terms or units mentioned\n"
            )

        # Question Analysis
        analysis_parts.append(
            "Question Analysis:\n"
            f"{example['question']}\n"
            "1. Identify the required information and its location in the figure\n"
            "2. Determine the type of answer needed\n"
        )

        # Integration Analysis
        analysis_parts.append(
            "Integration Analysis:\n"
            "1. Connect the caption context with the question requirements\n"
            "2. Identify the relevant figure elements for answering\n"
        )
        prompt_parts.append("\n".join(analysis_parts))

        # Step 2: Compound Figure Navigation (if applicable)
        if example.get('compound'):
            prompt_parts.append("\nSTEP 2: COMPOUND FIGURE NAVIGATION")
            prompt_parts.append(self._get_compound_navigation(example))

        return "\n".join(prompt_parts)

    def create_prompt(self, example: Dict[str, Any]) -> List[str]:
        """Create a complete Chain of Thought prompt based on the example."""
        # First create the initial analysis prompt
        initial_analysis = self.create_initial_analysis_prompt(example)

        # Then create the answer instruction
        instruction = self.base_instruction + "\n\n" + \
            self._create_answer_instruction(example)
        # Combine both parts with the base instruction
        return [initial_analysis, instruction]

    def _create_answer_instruction(self, example: Dict[str, Any]) -> str:
        """Create the answer instruction based on QA pair type."""
        instruction_parts = []

        # Handle compound figures if not already handled in initial analysis
        if example.get('compound'):
            instruction_parts.append(
                f"Navigate to the {example.get('fig_numb', 1)} graph in the compound figure and analyze it.\n"
            )

        qa_pair_type = example.get('qa_pair_type', '').lower()

        # Handle non-binary multiple choice
        if "non-binary" in qa_pair_type and example.get('choices'):
            instruction_parts.append(
                "This is a multiple-choice question:\n"
                "1. Examine each option\n"
                "2. Match with the evidence\n"
                "3. Provide the letter(s) of correct option(s)\n"
                "Remember: If multiple letters are correct, separate them by commas without spaces (e.g., B,C)"
            )

            # Add choices analysis
            instruction_parts.append("\nOptions:")
            for k, v in example['choices'].items():
                instruction_parts.append(f"Option {k}: {v}")

        # Handle binary questions
        elif "binary" in qa_pair_type:
            instruction_parts.append(self._get_binary_instruction(
                example['question'], qa_pair_type))

        # Handle infinite answer set
        if "infinite_answer_set" in qa_pair_type:
            instruction_parts.append(
                "This question requires a precise numerical answer:\n"
                "1. Identify the required value\n"
                "2. Locate it in the figure\n"
                "3. Provide the exact numerical value\n"
            )

        # Handle visual questions
        if "non-visual" in qa_pair_type:
            instruction_parts.append(
                "This question is based on textual information:\n"
                "1. Identify the relevant text\n"
                "2. Provide a concise answer\n"
            )
        elif "visual" in qa_pair_type:
            instruction_parts.append(
                "This question requires visual analysis:\n"
                "1. Identify the relevant visual elements\n"
                "2. Provide a concise answer based on visual evidence, approximations in the scale are allowed\n"
            )

        if "unanswerable" in qa_pair_type:
            instruction_parts.append(
                "This question appears to be unanswerable:\n"
                "1. Identify what information is needed\n"
                "2. Check if it's available\n"
                "3. Confirm if the question can be answered\n"
            )

        return "\n".join(instruction_parts)

    def create_batch_prompts(self, examples: List[Dict[str, Any]]) -> List[str]:
        """Create Chain of Thought prompts for a batch of examples."""
        return [self.create_prompt(example) for example in examples]

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
    """Class to create Chain of Thought prompts with self-questioning and verification."""

    def __init__(self):
        """Initialize the COT prompt creator."""
        self.base_instruction = (
            "Please think step by step, and regularly perform self-questioning, "
            "self-verification, self-correction to check your ongoing reasoning, "
            "using connectives such as 'Wait a moment', 'Wait, does it seem right?', etc.\n\n"
            "After your reasoning, provide your final answer in a new line starting with 'Final Answer:'"
        )

    def _get_figure_type_instruction(self, figure_type: str) -> str:
        """Get specific instructions based on figure type."""
        figure_type = figure_type.lower()

        if figure_type in FigureType.LINE_CHART:
            return (
                "Let's analyze this line chart step by step:\n"
                "1. First, let's identify the colors of different lines and their meanings\n"
                "2. Next, examine the X and Y axis labels and their units\n"
                "3. Consider the scale and range of values\n"
                "4. Finally, analyze the trends and patterns in the lines\n"
            )
        elif figure_type in FigureType.BAR_CHART:
            return (
                "Let's analyze this bar chart step by step:\n"
                "1. First, identify the colors of different bars and their meanings\n"
                "2. Next, examine the X and Y axis labels and their units\n"
                "3. Consider the scale and range of values\n"
                "4. Finally, analyze the height and position of bars\n"
            )
        elif figure_type in FigureType.BOX_PLOT:
            return (
                "Let's analyze this box plot step by step:\n"
                "1. First, locate the median line position\n"
                "2. Next, identify the box boundaries (Q1 and Q3)\n"
                "3. Consider the whisker extent\n"
                "4. Finally, check for any outliers\n"
            )
        elif figure_type in FigureType.CONFUSION_MATRIX:
            return (
                "Let's analyze this confusion matrix step by step:\n"
                "1. First, identify the row and column labels\n"
                "2. Next, examine the numerical values in each cell\n"
                "3. Consider the color intensity if present\n"
                "4. Finally, analyze the overall distribution of values\n"
            )
        elif figure_type in FigureType.PIE_CHART:
            return (
                "Let's analyze this pie chart step by step:\n"
                "1. First, identify the segments and their labels\n"
                "2. Next, examine the percentage or proportion values\n"
                "3. Consider the colors of different segments\n"
                "4. Finally, analyze the size of each segment relative to others\n"
            )
        return (
            "Let's analyze this figure step by step:\n"
            "1. First, identify the colors and labels present\n"
            "2. Next, examine any other relevant information\n"
            "3. Finally, consider how these elements relate to the question\n"
        )

    def _get_binary_instruction(self, question: str, qa_pair_type: str = "") -> str:
        """Get appropriate binary instruction based on question type."""
        question_lower = question.lower()

        # Determine if it's True/False or Yes/No
        if any(phrase in question_lower for phrase in ['is it true', 'is it false', 'is this true', 'is this false']):
            answer_format = "True or False"
            evidence_type = "visual" if "visual" in qa_pair_type else "textual"
        else:
            answer_format = "Yes or No"
            evidence_type = "visual" if "visual" in qa_pair_type else "textual"

        return (
            f"This is a binary question requiring a {answer_format} answer based on {evidence_type} evidence.\n"
            "Let's analyze this step by step:\n"
            "1. First, identify the key elements in the question\n"
            "2. Next, examine the evidence carefully\n"
            "3. Consider if the evidence supports or contradicts the statement\n"
            "4. Verify your reasoning by asking: 'Does this make sense?'\n"
            "5. Finally, provide your {answer_format} answer with confidence\n"
        )

    def _get_compound_navigation(self, example: Dict[str, Any]) -> str:
        """Get navigation instructions for compound figures."""
        fig_numb = example.get('fig_numb', 1)
        return (
            "This is a compound figure containing multiple subfigures.\n"
            f"Let's navigate to the {fig_numb} graph in the compound figure:\n"
            "1. First, locate the correct subfigure\n"
            "2. Next, verify we're looking at the right one\n"
            "3. Finally, proceed with our analysis\n"
        )

    def _create_base_prompt(self, example: Dict[str, Any]) -> List[str]:
        """Create the base prompt parts with step-by-step analysis."""
        prompt_parts = []

        # Add caption if available
        if example.get('caption'):
            prompt_parts.append(
                "Let's start by analyzing the caption:\n"
                f"{example['caption']}\n"
                "1. What key information does the caption provide?\n"
                "2. How does this relate to our question?\n"
            )

        # Handle compound figures
        if example.get('compound'):
            prompt_parts.append(self._get_compound_navigation(example))

        # Add figure type information and specific instructions
        figure_type = example.get('figure_type', '').lower()
        prompt_parts.append(
            f"This is a {figure_type.replace('_', ' ')}.")
        type_instruction = self._get_figure_type_instruction(figure_type)
        if type_instruction:
            prompt_parts.append(type_instruction)

        # Add question with step-by-step analysis
        prompt_parts.append(
            f"Now, let's analyze the question:\n"
            f"{example['question']}\n"
            "1. What is the question asking?\n"
            "2. What information do we need to answer it?\n"
            "3. How can we find this information in the figure?\n"
        )

        # Add choices if available
        if example.get('choices'):
            prompt_parts.append("Let's examine the available options:")
            for k, v in example['choices'].items():
                prompt_parts.append(f"Option {k}: {v}")
            prompt_parts.append(
                "For each option:\n"
                "1. What evidence supports this option?\n"
                "2. What evidence contradicts this option?\n"
                "3. Is this option consistent with the figure?\n"
            )

        return prompt_parts

    def _create_answer_instruction(self, example: Dict[str, Any]) -> str:
        """Create the answer instruction based on QA pair type, compound navigation, and choices."""
        instruction_parts = []

        # Handle compound figures if not already handled in initial analysis
        if example.get('compound'):
            instruction_parts.append(
                "Let's navigate to the correct subfigure:\n"
                f"1. Locate the {example.get('fig_numb', 1)} graph in the compound figure\n"
                "2. Verify we're looking at the right one\n"
                "3. Focus our analysis on this specific subfigure\n"
            )

        qa_pair_type = example.get('qa_pair_type', '').lower()

        # Handle non-binary multiple choice
        if "non-binary" in qa_pair_type and example.get('choices'):
            instruction_parts.append(
                "This is a multiple-choice question. Let's analyze it step by step:\n"
                "1. First, examine each option carefully\n"
                "2. For each option, ask: 'Is this supported by the evidence?'\n"
                "3. Verify your reasoning: 'Have I considered all possibilities?'\n"
                "4. If multiple options are correct, ensure they're all justified\n"
                "5. Finally, provide the letter(s) of the correct option(s)\n"
                "Remember: If multiple letters are correct, separate them by commas without spaces (e.g., B,C)"
            )

            # Add choices analysis
            instruction_parts.append("\nLet's examine each option:")
            for k, v in example['choices'].items():
                instruction_parts.append(f"Option {k}: {v}")
            instruction_parts.append(
                "For each option:\n"
                "1. What evidence supports this option?\n"
                "2. What evidence contradicts this option?\n"
                "3. Is this option consistent with the figure?\n"
            )

        # Handle binary questions
        elif "binary" in qa_pair_type:
            instruction_parts.append(self._get_binary_instruction(
                example['question'], qa_pair_type))

        # Handle infinite answer set
        if "infinite_answer_set" in qa_pair_type:
            instruction_parts.append(
                "This question requires a precise numerical answer. Let's analyze it step by step:\n"
                "1. First, identify what numerical value we need to find\n"
                "2. Next, locate this information in the figure\n"
                "3. Consider the scale and units carefully\n"
                "4. Verify your calculation: 'Does this make sense?'\n"
                "5. Finally, provide the exact numerical value\n"
            )

        # Handle visual questions
        if "non-visual" in qa_pair_type:
            instruction_parts.append(
                "This question is based on textual information. Let's analyze it step by step:\n"
                "1. First, identify the relevant text information\n"
                "2. Next, consider how it relates to the question\n"
                "3. Verify your understanding: 'Have I interpreted this correctly?'\n"
                "4. Finally, provide a concise answer\n"
            )
        elif "visual" in qa_pair_type:
            instruction_parts.append(
                "This question requires visual analysis. Let's examine it step by step:\n"
                "1. First, identify the relevant visual elements\n"
                "2. Next, analyze their properties (shape, size, position, etc.)\n"
                "3. Verify your observations: 'Am I seeing this correctly?'\n"
                "4. Finally, provide a concise answer based on visual evidence\n"
            )

        if "unanswerable" in qa_pair_type:
            instruction_parts.append(
                "This question appears to be unanswerable. Let's verify this step by step:\n"
                "1. First, identify what information we would need to answer it\n"
                "2. Next, check if this information is available\n"
                "3. Verify our conclusion: 'Have we missed any relevant information?'\n"
                "4. Finally, confirm that the question cannot be answered\n"
            )

        # Add final verification
        instruction_parts.append(
            "\nBefore providing the final answer, let's verify:\n"
            "1. Have we considered all relevant information?\n"
            "2. Is our reasoning consistent with the evidence?\n"
            "3. Have we double-checked our analysis?\n"
            "4. Are we confident in our answer?\n"
        )

        return "\n".join(instruction_parts)

    def create_initial_analysis_prompt(self, example: Dict[str, Any]) -> str:
        """Create an initial analysis prompt focusing on caption, question, and image analysis first."""
        prompt_parts = []

        # Step 1: Comprehensive Initial Analysis
        prompt_parts.append("STEP 1: COMPREHENSIVE INITIAL ANALYSIS")
        prompt_parts.append(
            "Let's analyze the figure, caption, and question together:")

        # Combined Analysis
        analysis_parts = []

        # Figure Type Analysis
        figure_type = example.get('figure_type', '').lower()
        analysis_parts.append(
            f"Figure Type: {figure_type.replace('_', ' ')}\n"
            "1. What type of visualization is this?\n"
            "2. What are the key visual elements we should focus on?\n"
        )

        # Caption Analysis
        if example.get('caption'):
            analysis_parts.append(
                "Caption Analysis:\n"
                f"{example['caption']}\n"
                "1. What is the main topic and purpose of this figure?\n"
                "2. What key information and context does the caption provide?\n"
                "3. Are there any specific terms, units, or concepts we should note?\n"
            )

        # Question Analysis
        analysis_parts.append(
            "Question Analysis:\n"
            f"{example['question']}\n"
            "1. What specific information is the question asking for?\n"
            "2. How does this relate to the figure and caption?\n"
            "3. What visual elements or data points will we need to examine?\n"
        )

        # Integration Analysis
        analysis_parts.append(
            "Integration Analysis:\n"
            "1. How do the caption and question work together?\n"
            "2. Which parts of the figure are most relevant to the question?\n"
            "3. What connections can we make between the caption, question, and visual elements?\n"
        )

        # Add self-verification for initial analysis
        analysis_parts.append(
            "\nLet's verify our initial understanding:\n"
            "1. Have we correctly identified the figure type and its key elements?\n"
            "2. Do we understand how the caption provides context for the question?\n"
            "3. Are we clear about what information we need to find in the figure?\n"
            "4. Have we identified the most relevant parts of the figure to focus on?\n"
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
        instruction = self._create_answer_instruction(example)

        # Combine both parts with the base instruction
        return [initial_analysis, instruction]

    def create_batch_prompts(self, examples: List[Dict[str, Any]]) -> List[str]:
        """Create Chain of Thought prompts for a batch of examples."""
        return [self.create_prompt(example) for example in examples]

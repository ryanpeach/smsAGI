from dataclasses import dataclass
import yaml
from pathlib import Path
from langchain import PromptTemplate

class Prompts:
    DEFAULT_PERSONALITY = Path(__file__).parent.parent.parent / "personalities/default.yaml"

    def __init__(self, personality: Path) -> None:
        self.personality_file = personality
        self.personality = yaml.safe_load(personality.read_text())
        self.default_personality = yaml.safe_load(self.DEFAULT_PERSONALITY.read_text())

    def _get_prompt_template(self, prompt_name: str) -> PromptTemplate:
        personality_prompts = self.personality.get("prompts", {})
        default_prompts = self.default_personality["prompts"]

        prompt = personality_prompts.get(prompt_name, {})
        default_prompt = default_prompts[prompt_name]

        prompt_template = prompt.get("template", default_prompt["template"])
        prompt_input_variables = prompt.get("input_variables", default_prompt["input_variables"])

        return PromptTemplate(template=prompt_template, input_variables=prompt_input_variables)

    def get_task_prioritization_prompt(self) -> PromptTemplate:
        """
        Returns the prompt for task prioritization.
        Returns default prompt variables if any variable is missing.
        """
        return self._get_prompt_template("task_prioritization")

    def get_task_creation_prompt(self) -> PromptTemplate:
        """
        Returns the prompt for task creation.
        Returns default prompt variables if any variable is missing.
        """
        return self._get_prompt_template("task_creation")

    def get_objective(self) -> str:
        """
        Returns the objective of the agent.
        Returns default objective if objective is missing.
        """
        return self.personality.get("objective", self.default_personality["objective"])
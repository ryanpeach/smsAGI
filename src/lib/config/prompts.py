from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import yaml
from langchain import PromptTemplate
from langchain.chat_models.openai import ChatOpenAI

from lib.config import Config


class Prompts(Config):
    def _get_prompt_template(self, prompt_name: str) -> PromptTemplate:
        config_prompts = self.config.get("prompts", {})
        default_prompts = self.default_config["prompts"]

        prompt = config_prompts.get(prompt_name, {})
        default_prompt = default_prompts[prompt_name]

        prompt_template = prompt.get("template", default_prompt["template"])
        prompt_input_variables = prompt.get(
            "input_variables", default_prompt["input_variables"]
        )

        return PromptTemplate(
            template=prompt_template, input_variables=prompt_input_variables
        )

    def _get_prompt_llm(self, prompt_name: str) -> ChatOpenAI:
        config_prompts = self.config.get("prompts", {})
        default_prompts = self.default_config["prompts"]

        prompt = config_prompts.get(prompt_name, {})
        default_prompt = default_prompts[prompt_name]

        llm = prompt.get("llm", {})
        default_llm = default_prompt["llm"]

        llm_model_name = llm.get("model_name", default_llm["model_name"])
        llm_temperature = llm.get("temperature", default_llm["temperature"])

        return ChatOpenAI(model_name=llm_model_name, temperature=llm_temperature)  # type: ignore

    def get_task_prioritization_prompt(self) -> Tuple[PromptTemplate, ChatOpenAI]:
        """
        Returns the prompt for task prioritization.
        Returns default prompt variables if any variable is missing.
        """
        prompt_template = self._get_prompt_template("task_prioritization")
        llm = self._get_prompt_llm("task_prioritization")
        return prompt_template, llm

    def get_task_creation_prompt(self) -> Tuple[PromptTemplate, ChatOpenAI]:
        """
        Returns the prompt for task creation.
        Returns default prompt variables if any variable is missing.
        """
        prompt_template = self._get_prompt_template("task_creation")
        llm = self._get_prompt_llm("task_creation")
        return prompt_template, llm

    def get_goals(self) -> Tuple[PromptTemplate, ChatOpenAI]:
        """
        Returns the objective of the agent.
        Returns default objective if objective is missing.
        """
        prompt_template = self._get_prompt_template("goals")
        llm = self._get_prompt_llm("goals")
        return prompt_template, llm

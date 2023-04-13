from lib.prompts import Prompts
from langchain import BaseLLM, LLMChain

class TaskCreationChain(LLMChain):
    """Chain to generates tasks."""

    @classmethod
    def from_llm(cls, prompts: Prompts, llm: BaseLLM, verbose: bool = True) -> "TaskCreationChain":
        """Get the response parser."""
        return cls(prompt=prompts.get_task_creation_prompt(), llm=llm, verbose=verbose)

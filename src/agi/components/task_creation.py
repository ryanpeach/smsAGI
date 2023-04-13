from lib.prompts import Prompts
from langchain import BaseLLM, LLMChain
import uuid

class TaskCreationChain:
    """Chain to generates tasks."""

    def __init__(self, prompts: Prompts, llm: BaseLLM, verbose: bool = True):
        """Get the response parser."""
        self.chain = LLMChain(prompt=prompts.get_task_creation_prompt(), llm=llm, verbose=verbose)

    def create_next_task(
        self,
        prev_task_result: Dict,
        task_description: str,
        task_list: TaskList,
        goals: Goals
    ) -> None:
        """Get the next task."""
        incomplete_tasks = ", ".join([task["task_name"] for task in task_list])
        response = self.chain.run(
            result=prev_task_result,
            task_description=task_description,
            incomplete_tasks=incomplete_tasks,
            objective=goals.get_prompt(),
        )
        new_tasks = response.split("\n")
        out = [
            Task(task_name=task_name, task_id=uuid.uuid4())
            for task_name in new_tasks
            if task_name.strip()
        ]
        for new_task in out:
            task_list.add_task(new_task)


from typing import List
from lib.prompts import Prompts
from langchain.agent import Task
from langchain import BaseLLM, LLMChain
import uuid

class TaskPrioritizationChain:
    """Chain to prioritize tasks."""

    def __init__(self, prompts: Prompts, llm: BaseLLM, verbose: bool = True):
        self.chain = LLMChain(prompt=prompts.get_task_prioritization_prompt(), llm=llm, verbose=verbose)

    def prioritize_tasks(
        self,
        this_task_id: int,
        task_list: TaskList,
        objective: str,
    ) -> None:
        """Prioritize tasks."""
        task_names = [t["task_name"] for t in task_list]
        next_task_id = int(this_task_id) + 1
        response = self.run(
            task_names=task_names, next_task_id=next_task_id, objective=objective
        )
        new_tasks = response.split("\n")
        prioritized_task_list = []
        for task_string in new_tasks:
            if not task_string.strip():
                continue
            task_parts = task_string.strip().split(".", 1)
            if len(task_parts) == 2:
                task_id = uuid.UUID(task_parts[0].strip())
                task_name = task_parts[1].strip()
                prioritized_task_list.append(Task(task_id=task_id, task_name=task_name))

        task_list.set_task_prioriziation(prioritized_task_list)

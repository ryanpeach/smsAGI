from pathlib import Path
from lib.prompts import Prompts
from langchain.agents import Task
from langchain import BaseLLM, LLMChain
import uuid
from lib.sql import SuperAgent, TaskListItem
from lib.sql.goals import Goals
from sqlalchemy.orm import Session


class TaskPrioritizationAgent:
    """Chain to prioritize tasks."""

    def __init__(self, agent: SuperAgent, session: Session, llm: BaseLLM, config: Path):
        self.goals = Goals(agent=agent, session=session)
        self.chain = LLMChain(
            prompt=Prompts.get_task_prioritization_prompt(), llm=llm, verbose=True
        )

    def run(
        self,
        task_list_item: TaskListItem,
    ) -> None:
        """Prioritize tasks."""
        task_names = [t["task_name"] for t in self.task_list.get_tasks()]
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

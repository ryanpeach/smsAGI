from pathlib import Path
from typing import Dict
from lib.prompts import Prompts
from lib.agents.task_prioritization_agent import TaskPrioritizationAgent
from lib.tools import Tools
from langchain import BaseLLM, LLMChain
from langchain.agents import initialize_agent, AgentType, Task, Tool
from lib.prompts import Prompts
from lib.sql.goals import Goals
from sqlalchemy.orm import Session
from lib.sql import SuperAgent, TaskListItem
import uuid


class TaskExecutionAgent:
    def __init__(self, agent: SuperAgent, session: Session, llm: BaseLLM, config: Path):
        self.tools = Tools.get_tools()
        self.agent_chain = initialize_agent(
            self.tools, llm, agent=AgentType.REACT_DESCRIPTION, verbose=True
        )
        self.agent = agent
        self.chain = LLMChain(
            prompt=Prompts.get_task_creation_prompt(), llm=llm, verbose=True
        )
        self.goals = Goals(agent=agent, session=session)
        self.task_prioritization_agent = TaskPrioritizationAgent(agent=agent, session=session, llm=llm, config=config)

    def get_tools(self) -> list[Tool]:
        """Gets a list of tools from the config file."""
        tools = [
            self.tools._get_search_tool(),
            self.tools._get_todo_tool(),
            self.tools._get_send_message_tool(),
        ]
        return [tool for tool in tools if tool is not None]

    def _execute_task(
        self,
        task: TaskListItem,
    ) -> str:
        """Execute a task."""
        # context = self.vectorstore.get_top_tasks(query=self.agent.objective, k=k)
        return self.agent_chain.run(
            objective=self.goals.get_prompt(), task=task["task_name"]
        )

    def _create_next_task(
        self,
        prev_task_result: Dict,
        task_list_item: TaskListItem,
    ) -> None:
        """Get the next task."""
        response = self.chain.run(
            result=prev_task_result,
            task_description=task_list_item.name,
            objective=self.goals.get_prompt(),
        )
        new_tasks = response.split("\n")
        out = [
            TaskListItem.create()
            for task_name in new_tasks
            if task_name.strip()
        ]
        for new_task in out:
            task_list.add_task(new_task)

    def run(
            self,
            task_list_item: TaskListItem,
    ) -> TaskListItem:
        """Run a task."""
        result = self._execute_task(task=task_list_item)
        TaskListItem = self._create_next_task(prev_task_result=result, task_list=task_list_item)
        self.task_prioritization_agent.run(task_list_item)
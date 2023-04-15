import asyncio
from pathlib import Path
from typing import Dict, List
from agi.twilio import WAITING_FOR_USER_RESPONSE
from lib.config.prompts import Prompts
from lib.agents.task_prioritization_agent import TaskPrioritizationAgent
from lib.config.tools import Tools
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from lib.config.prompts import Prompts
from lib.sql import Goal
from sqlalchemy.orm import Session
from lib.sql import SuperAgent, TaskListItem


class TaskExecutionCreationPrioritizationAgent:
    def __init__(self, super_agent: SuperAgent, session: Session, config: Path):
        self.session = session
        self.tools = self.get_tools(config=config)
        prompts = Prompts(config=config)
        goals = Goal.get_goals(agent=super_agent, session=session)
        self.objective = Goal.get_prompt(goals=goals, prompts=prompts)
        self.super_agent = super_agent
        self.task_execution_agent = initialize_agent(
            self.tools, llm=self.get_task_execution_llm(config=config), agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        )
        task_creation_prompt, task_creation_llm = prompts.get_task_creation_prompt()
        self.task_creation_chain = LLMChain(
            prompt=task_creation_prompt, llm=task_creation_llm, verbose=True
        )
        self.task_prioritization_agent = TaskPrioritizationAgent(super_agent=super_agent, session=session, config=config)

    @staticmethod
    def get_task_execution_llm(config: Path) -> ChatOpenAI:
        """Gets the LLM from the config file."""
        return Tools(config=config).get_zero_shot_llm()

    @staticmethod
    def get_tools(config: Path) -> list[Tool]:
        """Gets a list of tools from the config file."""
        tools = Tools(config=config)
        tools_list = [
            tools.get_search_tool(),
            tools.get_todo_tool(),
            tools.get_send_message_tool(),
        ]
        return [tool for tool in tools_list if tool is not None]

    async def _execute_task(
        self,
        task: TaskListItem,
    ) -> str:
        """Execute a task."""
        # context = self.vectorstore.get_top_tasks(query=self.agent.objective, k=k)
        return await self.task_execution_agent.arun(
            objective=self.objective, task=task.description
        )

    async def _create_next_task(
        self,
        prev_task_result: str,
        task_list_item: TaskListItem,
    ) -> List[TaskListItem]:
        """Get the next task."""
        response = await self.task_creation_chain.arun(
            result=prev_task_result,
            task_description=task_list_item.description,
            objective=self.objective,
        )
        new_tasks = response.split("\n")
        out = [
            TaskListItem(super_agent=self.super_agent, description=task_description)
            for task_description in new_tasks
            if task_description.strip()
        ]
        return out

    async def arun(
            self,
            task_list_item: TaskListItem,
    ) -> None:
        """Run a task."""
        result = await self._execute_task(task=task_list_item)
        if result != WAITING_FOR_USER_RESPONSE:
            created_tasks = await self._create_next_task(prev_task_result=result, task_list_item=task_list_item)
            tasks = [self.task_prioritization_agent.arun(created_task_list_item) for created_task_list_item in created_tasks]
            # Call the prioritization agent on the new tasks all at once
            await asyncio.gather(*tasks)
        return None

    def run(
            self,
            task_list_item: TaskListItem,
    ) -> None:
        """Run a task."""
        asyncio.run(self.arun(task_list_item=task_list_item))
        return None
from pathlib import Path
from typing import Optional
from langchain.agents import initialize_agent, AgentType, Tool
from langchain import BaseLLM
from lib.sql import SuperAgent, TaskListItem
from lib.sql.goals import Goals
from lib.sql.task_list import TaskList
from lib.agents.task_prioritization_agent import TaskPrioritizationAgent
from lib.tools import Tools
from sqlalchemy.orm import Session


class UserAgent:
    def __init__(self, super_agent: SuperAgent, session: Session, llm: BaseLLM, config: Path):
        self.tools = Tools.get_tools()
        self.task_list = TaskList.from_super_agent(
            super_agent=super_agent, session=session
        )
        self.agent_chain = initialize_agent(
            self.tools, llm, agent=AgentType.REACT_DESCRIPTION, verbose=True
        )
        self.super_agent = super_agent
        self.goals = Goals(super_agent=super_agent, session=session)
        self.task_prioritization_agent = TaskPrioritizationAgent(agent=super_agent, session=session, llm=llm, config=config)

    def get_tools(self) -> list[Tool]:
        """Gets a list of tools from the config file."""
        tools = [
            self.tools._get_todo_tool(),
            self.tools._get_send_message_tool(),
        ]
        return [tool for tool in tools if tool is not None]

    async def arun(self, user_msg: str):
        # Get tasks from the user
        objective = await self.goals.get_prompt()
        task: Optional[TaskListItem] = await self.agent_chain.arun(objective=objective, task=user_msg)
        if task is not None:
            await self.task_prioritization_agent.arun(task_list_item=task)
        return
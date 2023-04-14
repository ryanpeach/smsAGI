from langchain.agents import initialize_agent, AgentType, Tool
from langchain import BaseLLM
from lib.sql import SuperAgent
from lib.sql.goals import Goals
from lib.sql.task_list import TaskList
from lib.tools import Tools


class UserAgent:
    def __init__(self, super_agent: SuperAgent, session: Session, llm: BaseLLM):
        self.tools = Tools.get_tools()
        self.task_list = TaskList.from_super_agent(
            super_agent=super_agent, session=session
        )
        self.agent_chain = initialize_agent(
            self.tools, llm, agent=AgentType.REACT_DESCRIPTION, verbose=True
        )
        self.super_agent = super_agent
        self.goals = Goals(super_agent=super_agent, session=session)

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
        return await self.agent_chain.arun(objective=objective, task=user_msg)

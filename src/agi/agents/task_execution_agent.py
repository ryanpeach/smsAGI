from pathlib import Path
from typing import Dict
from lib.prompts import Prompts
from lib.tools import Tools
from langchain import BaseLLM, LLMChain
from langchain.chat_models import BaseLLM
from langchain.agents import initialize_agent, AgentType, Task, Tool
from lib.prompts import Prompts
from lib.sql.goals import Goals
from lib.sql.task_list import TaskList
from lib.sql import SuperAgent
import uuid


class TaskExecutionAgent:
    def __init__(self, agent: SuperAgent, session: Session, llm: BaseLLM):
        self.tools = Tools.get_tools()
        self.task_list = TaskList(agent=agent, session=session)
        self.agent_chain = initialize_agent(
            self.tools, llm, agent=AgentType.REACT_DESCRIPTION, verbose=True
        )
        self.agent = agent
        self.chain = LLMChain(
            prompt=Prompts.get_task_creation_prompt(), llm=llm, verbose=True
        )
        self.goals = Goals(agent=agent, session=session)

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
        task: Task,
    ) -> str:
        """Execute a task."""
        # context = self.vectorstore.get_top_tasks(query=self.agent.objective, k=k)
        return self.agent_chain.run(
            objective=self.goals.get_prompt(), task=task["task_name"]
        )

    def execute_task(self):
        # Step 1: Pull the first task
        task = self.task_list.pop_task()
        self.print_next_task(task)

        # Step 2: Execute the task
        task_result = self._execute_task(task)
        self.print_task_result(task_result)

        # Step 3: Store the result in Pinecone
        # result_id = f"result_{task['task_id']}"
        # self.vectorstore.add_texts(
        #     texts=[task_result],
        #     metadatas=[{"task": task["task_name"]}],
        #     ids=[result_id],
        # )

    def create_next_task(
        self,
        prev_task_result: Dict,
        task_description: str,
        task_list: TaskList,
        goals: Goals,
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

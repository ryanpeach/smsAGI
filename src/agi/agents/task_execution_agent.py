from pathlib import Path
from lib.prompts import Prompts
from agi.tools import Tools
from langchain import BaseLLM
from langchain.chat_models import BaseLLM
from langchain.agents import initialize_agent, AgentType, Task
from lib.prompts import Prompts
from agi.memory import VectorStoreMemory
from lib.sql.goals import Goals
from lib.sql.task_list import TaskList
from lib.sql import SuperAgent

class TaskExecutionAgent:

    def __init__(self, agent: SuperAgent, tools: Tools, session: Session, llm: BaseLLM, vectorstore: VectorStoreMemory, task_list: TaskList):
        self.tools = tools.get_tools()
        self.task_list = task_list
        self.vectorstore = vectorstore
        self.agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)
        self.agent = agent
        self.chain = LLMChain(prompt=prompts.get_task_creation_prompt(), llm=llm, verbose=verbose)
        self.goals = Goals(agent=agent, session=session)

    def get_tools(self) -> list[Tool]:
        """Gets a list of tools from the personality file."""
        tools = [
            self.tools._get_search_tool(),
            self.tools._get_todo_tool(),
            self.tools._get_send_message_tool(),
        ]
        return [tool for tool in tools if tool is not None]

    def _execute_task(
        self,
        task: Task,
        k: int = 5
    ) -> str:
        """Execute a task."""
        context = self.vectorstore.get_top_tasks(query=self.agent.objective, k=k)
        return self.agent_chain.run(objective=self.goals.get_prompt(), context=context, task=task["task_name"])

    def execute_task(self):
        # Step 1: Pull the first task
        task = self.task_list.pop_task()
        self.print_next_task(task)

        # Step 2: Execute the task
        task_result = self.execute_task(
            self.objective, task
        )

        this_task_id = int(task["task_id"])
        self.print_task_result(task_result)

        # Step 3: Store the result in Pinecone
        result_id = f"result_{task['task_id']}"
        self.vectorstore.add_texts(
            texts=[task_result],
            metadatas=[{"task": task["task_name"]}],
            ids=[result_id],
        )

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

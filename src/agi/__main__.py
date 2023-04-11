# REF: https://github.com/hwchase17/langchain/blob/master/docs/use_cases/agents/baby_agi.ipynb
# REF: https://github.com/hwchase17/langchain/blob/66786b0f0fb91f346c00d860eb8472f940d37af8/docs/use_cases/agents/baby_agi_with_agent.ipynb

import os
from pathlib import Path
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, TypedDict
from agi.tools import Tools
from agi.prompts import Prompts

import faiss
from colorama import Fore, Style, init
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
from langchain.chains.base import Chain
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
import argparse
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor

class Task(TypedDict):
    """Task model."""

    task_id: uuid.UUID
    task_name: str

# Initialize colorama
init(autoreset=True)

# Set up the argument parser
parser = argparse.ArgumentParser(description="Configure the BabyAGI agent.")
parser.add_argument(
    "--personality",
    type=Path,
    default=Path("personalities/default.yaml"),
    help="Path to the personality file for smsAGI.",
)
parser.add_argument(
    "--temperature", type=float, default=0, help="Temperature for the OpenAI model."
)
parser.add_argument(
    "--max_iterations",
    type=int,
    default=3,
    help="Maximum number of iterations for the BabyAGI agent.",
)
parser.add_argument(
    "--embedding_size",
    type=int,
    default=1536,
    help="Size of the embeddings for the FAISS index.",
)
parser.add_argument(
    "--verbose", action="store_true", help="Enable verbose output for BabyAGI."
)
args = parser.parse_args()

# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
embedding_size = args.embedding_size
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
tools_obj = Tools(args.personality)

tools = tools_obj.get_tools()
prompts = Prompts(args.personality)

class TaskCreationChain(LLMChain):
    """Chain to generates tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> "TaskCreationChain":
        """Get the response parser."""
        return cls(prompt=prompts.get_task_creation_prompt(), llm=llm, verbose=verbose)


class TaskPrioritizationChain(LLMChain):
    """Chain to prioritize tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> "TaskPrioritizationChain":
        """Get the response parser."""
        return cls(prompt=prompts.get_task_prioritization_prompt(), llm=llm, verbose=verbose)


prompt = tools_obj.get_zero_shot_prompt()


def get_next_task(
    task_creation_chain: TaskCreationChain,
    prev_task_result: Dict,
    task_description: str,
    task_list: List[Task],
    objective: str,
) -> List[Task]:
    """Get the next task."""
    incomplete_tasks = ", ".join([task["task_name"] for task in task_list])
    response = task_creation_chain.run(
        result=prev_task_result,
        task_description=task_description,
        incomplete_tasks=incomplete_tasks,
        objective=objective,
    )
    new_tasks = response.split("\n")
    return [
        Task(task_name=task_name, task_id=uuid.uuid4())
        for task_name in new_tasks
        if task_name.strip()
    ]


def prioritize_tasks(
    task_prioritization_chain: TaskPrioritizationChain,
    this_task_id: int,
    task_list: List[Task],
    objective: str,
) -> List[Task]:
    """Prioritize tasks."""
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id) + 1
    response = task_prioritization_chain.run(
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
    return prioritized_task_list


def _get_top_tasks(vectorstore, query: str, k: int) -> List[Task]:
    """Get the top k tasks based on the query."""
    results = vectorstore.similarity_search_with_score(query, k=k)
    if not results:
        return []
    sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))
    return [item.metadata["task"] for item in sorted_results]


def execute_task(
    vectorstore, execution_chain: LLMChain, objective: str, task: Task, k: int = 5
) -> str:
    """Execute a task."""
    context = _get_top_tasks(vectorstore, query=objective, k=k)
    return execution_chain.run(objective=objective, context=context, task=task)


class BabyAGI(Chain, BaseModel):
    """Controller model for the BabyAGI agent."""

    task_list: deque = Field(default_factory=deque)
    task_creation_chain: TaskCreationChain = Field(...)
    task_prioritization_chain: TaskPrioritizationChain = Field(...)
    execution_chain: AgentExecutor = Field(...)
    task_id_counter: int = Field(1)
    vectorstore: VectorStore = Field(init=False)
    max_iterations: Optional[int] = None

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    def add_task(self, task: Task) -> None:
        self.task_list.append(task)

    def pop_task(self) -> Task:
        return self.task_list.popleft()

    def get_tasks(self) -> List[Task]:
        return list(self.task_list)

    def set_tasks(self, tasks: List[Task]) -> None:
        self.task_list = deque(tasks)

    def _print_with_color(self, color: Fore, title: str, content: str) -> None:
        print(f"{color}{Style.BRIGHT}{title}{Style.RESET_ALL}")
        print(content)

    def print_task_list(self):
        task_list_str = "\n".join(
            [f"{t['task_id']}: {t['task_name']}" for t in self.task_list]
        )
        self._print_with_color(Fore.MAGENTA, "*****TASK LIST*****", task_list_str)

    def print_next_task(self, task: Task):
        self._print_with_color(
            Fore.GREEN, "*****NEXT TASK*****", f"{task['task_id']}: {task['task_name']}"
        )

    def print_task_result(self, result: str):
        self._print_with_color(Fore.YELLOW, "*****TASK RESULT*****", result)

    def _single_step(self):
        self.print_task_list()

        # Step 1: Pull the first task
        task = self.pop_task()
        self.print_next_task(task)

        # Step 2: Execute the task
        task_result = execute_task(
            self.vectorstore, self.execution_chain, self.objective, task["task_name"]
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

        # Get tasks from the user
        user_messages = self.qaagent.receive_all_user_responses()
        for user_msg in user_messages:
            self.add_task(Task(task_id=uuid.uuid4(), task_name=f"Respond to the user, who said: {user_msg}"))

        # Step 4: Create new tasks and reprioritize task list
        new_tasks = get_next_task(
            self.task_creation_chain,
            task_result,
            task["task_name"],
            list(self.get_tasks()),
            self.objective,
        )

        for new_task in new_tasks:
            self.add_task(new_task)

        self.set_tasks(
            (
                prioritize_tasks(
                    self.task_prioritization_chain,
                    this_task_id,
                    list(self.task_list),
                    self.objective,
                )
            )
        )

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        num_iters = 0
        while True:
            if self.task_list:
                self._single_step()
            num_iters += 1
            if self.max_iterations is not None and num_iters == self.max_iterations:
                print(f"{Fore.RED}{Style.BRIGHT}*****TASK ENDING*****{Style.RESET_ALL}")
                break
        return {}

    @classmethod
    def from_llm(
        cls,
        llm: BaseLLM,
        vectorstore: VectorStore,
        verbose: bool = False,
        **kwargs
    ) -> "BabyAGI":
        """Initialize the BabyAGI Controller."""
        task_creation_chain = TaskCreationChain.from_llm(
            llm, verbose=verbose
        )
        task_prioritization_chain = TaskPrioritizationChain.from_llm(
            llm, verbose=verbose
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
        return cls(
            task_creation_chain=task_creation_chain,
            task_prioritization_chain=task_prioritization_chain,
            execution_chain=agent_executor,
            vectorstore=vectorstore,
            **kwargs
        )


# For debugging this code in isolation
if __name__ == "__main__":

    # Initialize the BabyAGI agent using the parsed arguments
    index = faiss.IndexFlatL2(args.embedding_size)
    embeddings_model = OpenAIEmbeddings()
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
    llm = ChatOpenAI(temperature=args.temperature)
    baby_agi = BabyAGI.from_llm(
        llm,
        vectorstore,
        verbose=args.verbose,
        max_iterations=args.max_iterations,
        objective=prompts.get_objective(),
    )
    baby_agi(inputs={})

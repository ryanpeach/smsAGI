# REF: https://github.com/hwchase17/langchain/blob/master/docs/use_cases/agents/baby_agi.ipynb

import os
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, TypedDict

import faiss
from colorama import Fore, Style, init
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.chains.base import Chain
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field

# Initialize colorama
init(autoreset=True)

# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})


class Task(TypedDict):
    """Task model."""

    task_id: uuid.UUID
    task_name: str


class TaskCreationChain(LLMChain):
    """Chain to generates tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> "TaskCreationChain":
        """Get the response parser."""
        task_creation_template = (
            "You are an task creation AI that uses the result of an execution agent"
            " to create new tasks with the following objective: {objective},"
            " The last completed task has the result: {result}."
            " This result was based on this task description: {task_description}."
            " These are incomplete tasks: {incomplete_tasks}."
            " Based on the result, create new tasks to be completed"
            " by the AI system that do not overlap with incomplete tasks."
            " Return the tasks as an array."
        )
        prompt = PromptTemplate(
            template=task_creation_template,
            input_variables=[
                "result",
                "task_description",
                "incomplete_tasks",
                "objective",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class TaskPrioritizationChain(LLMChain):
    """Chain to prioritize tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> "TaskPrioritizationChain":
        """Get the response parser."""
        task_prioritization_template = (
            "You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing"
            " the following tasks: {task_names}."
            " Consider the ultimate objective of your team: {objective}."
            " Do not remove any tasks. Return the result as a numbered list, like:"
            " #. First task"
            " #. Second task"
            " Start the task list with number {next_task_id}."
        )
        prompt = PromptTemplate(
            template=task_prioritization_template,
            input_variables=["task_names", "next_task_id", "objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class ExecutionChain(LLMChain):
    """Chain to execute tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> "ExecutionChain":
        """Get the response parser."""
        execution_template = (
            "You are an AI who performs one task based on the following objective: {objective}."
            " Take into account these previously completed tasks: {context}."
            " Your task: {task}."
            " Response:"
        )
        prompt = PromptTemplate(
            template=execution_template,
            input_variables=["objective", "context", "task"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


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


class BabyAGI(Chain):
    """Controller model for the BabyAGI agent."""

    task_list: deque[Task] = Field(default_factory=deque)
    task_creation_chain: TaskCreationChain = Field(...)
    task_prioritization_chain: TaskPrioritizationChain = Field(...)
    execution_chain: ExecutionChain = Field(...)
    vectorstore: VectorStore = Field(init=False)
    max_iterations: Optional[int]
    objective: str
    _stop: bool

    def __init__(self, **data: Any) -> None:
        """Initialize the model."""
        super().__init__(**data)
        self.task_list = deque()
        self.task_prioritization_chain = TaskPrioritizationChain.from_llm(
            self.task_prioritization_chain.llm
        )
        self.task_creation_chain = TaskCreationChain.from_llm(
            self.task_creation_chain.llm
        )
        self.execution_chain = ExecutionChain.from_llm(self.execution_chain.llm)
        self.vectorstore = VectorStore()
        self._stop = False

    def add_task(self, task: Task) -> None:
        with self.threading_lock:
            self.task_list.append(task)

    def pop_task(self) -> Task:
        with self.threading_lock:
            return self.task_list.popleft()

    def get_tasks(self) -> List[Task]:
        with self.threading_lock:
            return list(self.task_list)

    def set_tasks(self, tasks: List[Task]) -> None:
        with self.threading_lock:
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

        # Step 4: Create new tasks and reprioritize task list
        new_tasks = get_next_task(
            self.task_creation_chain,
            task_result,
            task["task_name"],
            list(self.get_tasks()),
            self.objective,
        )

        for new_task in new_tasks:
            new_task.update({"task_id": uuid.uuid4()})
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

    def _call(self):
        num_iters = 0
        with self.threading_lock:
            stop = self._stop
        while not stop:
            if self.task_list:
                self._single_step()
            num_iters += 1
            if self.max_iterations is not None and num_iters == self.max_iterations:
                print(f"{Fore.RED}{Style.BRIGHT}*****TASK ENDING*****{Style.RESET_ALL}")
                break
            with self.threading_lock:
                stop = self._stop

    @classmethod
    def from_llm(
        cls, llm: BaseLLM, vectorstore: VectorStore, verbose: bool = False, **kwargs
    ) -> "BabyAGI":
        """Initialize the BabyAGI Controller."""
        task_creation_chain = TaskCreationChain.from_llm(llm, verbose=verbose)
        task_prioritization_chain = TaskPrioritizationChain.from_llm(
            llm, verbose=verbose
        )
        execution_chain = ExecutionChain.from_llm(llm, verbose=verbose)
        return cls(
            task_creation_chain=task_creation_chain,
            task_prioritization_chain=task_prioritization_chain,
            execution_chain=execution_chain,
            vectorstore=vectorstore,
            **kwargs,
        )


# For debugging this code in isolation
if __name__ == "__main__":
    OBJECTIVE = "Write a weather report for SF today"
    llm = OpenAI(temperature=0)
    # Logging of LLMChains
    verbose = False
    # If None, will keep on going forever
    max_iterations: Optional[int] = 3
    baby_agi = BabyAGI.from_llm(
        llm=llm, vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations
    )
    baby_agi({"objective": OBJECTIVE})

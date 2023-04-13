# REF: https://github.com/hwchase17/langchain/blob/master/docs/use_cases/agents/baby_agi.ipynb
# REF: https://github.com/hwchase17/langchain/blob/66786b0f0fb91f346c00d860eb8472f940d37af8/docs/use_cases/agents/baby_agi_with_agent.ipynb

import os
from pathlib import Path
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, TypedDict
from agi.components.task_execution_agent import Tools
from lib.prompts import Prompts

from colorama import Fore, Style, init
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
from langchain.chains.base import Chain

from langchain.llms import BaseLLM

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

tools_obj = Tools(args.personality)

tools = tools_obj.get_tools()
prompts = Prompts(args.personality)



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
    def _single_step(self):
        self.print_task_list()

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

# REF: https://github.com/hwchase17/langchain/blob/master/docs/use_cases/agents/baby_agi.ipynb
# REF: https://github.com/hwchase17/langchain/blob/66786b0f0fb91f346c00d860eb8472f940d37af8/docs/use_cases/agents/baby_agi_with_agent.ipynb

import os
from pathlib import Path
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, TypedDict
from agi.agents.task_execution_agent import Tools
from lib.prompts import Prompts

from colorama import Fore, Style, init
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
from langchain.chains.base import Chain

from langchain.llms import BaseLLM
from lib.sql import SuperAgent

from pydantic import BaseModel, Field
import argparse
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor

# Initialize colorama
init(autoreset=True)

# Set up the argument parser
parser = argparse.ArgumentParser(description="Configure the BabyAGI agent.")
parser.add_argument(
    "--config",
    type=Path,
    default=Path("config/default.yaml"),
    help="Path to the config file for smsAGI.",
)
parser.add_argument(
    "--temperature", type=float, default=0, help="Temperature for the OpenAI model."
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

# Iterate over all super agents and run the task prioritization and task execution agents
llm = ChatOpenAI(
    temperature=args.temperature,
)
all_agents_cache = {}
while True:
    all_agents = SuperAgent.get_all_agents()
    for agent in all_agents:
        # Check if the agent is already in the cache
        if agent.id in all_agents_cache:
            # Load the agents from the cache
            task_prioritization_agent, task_execution_agent, memory = all_agents_cache[
                agent.id
            ]
        else:
            # Initialize the agents
            memory = VectorStoreMemory(
                embedding_size=args.embedding_size,
            )
            task_prioritization_agent = TaskPrioritizationAgent(
                agent=agent, config=args.config, memory=memory
            )
            task_execution_agent = TaskExecutionAgent(
                agent=agent, config=args.config, memory=memory
            )
            all_agents_cache[agent.id] = (
                task_prioritization_agent,
                task_execution_agent,
                memory,
            )

        # Run the agents
        task_prioritization_agent.run()
        task_execution_agent.run()

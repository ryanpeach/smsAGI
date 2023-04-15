# REF: https://github.com/hwchase17/langchain/blob/master/docs/use_cases/agents/baby_agi.ipynb
# REF: https://github.com/hwchase17/langchain/blob/66786b0f0fb91f346c00d860eb8472f940d37af8/docs/use_cases/agents/baby_agi_with_agent.ipynb

import os
from pathlib import Path
import uuid
from loguru import logger
from collections import deque
from typing import Any, Dict, Generator, List, Optional, TypedDict
from agi.agents.task_execution_agent import TaskExecutionAgent, Tools
from lib.agents.task_prioritization_agent import TaskPrioritizationAgent
from lib.prompts import Prompts
from sqlalchemy.orm import sessionmaker, Session
from colorama import Fore, Style, init
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
from langchain.chains.base import Chain

from langchain.llms import BaseLLM
from lib.sql import SuperAgent, TaskListItem, create_engine_from_env

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
engine = create_engine_from_env()
Session = sessionmaker(bind=engine)
while True:
    # Then run the task execution agent with the top priority task
    with Session() as session:
        agent = SuperAgent.get_random_super_agent(session=session)
        top_priority_task_item: Optional[TaskListItem] = TaskListItem.get_top_task_list_item(session=session, agent=agent)
        if top_priority_task_item is None:
            logger.info(f"Super Agent: ({agent.id},{agent.name}) -> No tasks to execute.")
            continue
        task_execution_agent = TaskExecutionAgent(
            agent=agent, session=session, llm=llm, config=args.config
        ).run(top_priority_task_item)
        session.commit()
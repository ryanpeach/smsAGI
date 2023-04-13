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

from pydantic import BaseModel, Field
import argparse
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor

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


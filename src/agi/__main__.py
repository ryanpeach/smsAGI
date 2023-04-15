# REF: https://github.com/hwchase17/langchain/blob/master/docs/use_cases/agents/baby_agi.ipynb
# REF: https://github.com/hwchase17/langchain/blob/66786b0f0fb91f346c00d860eb8472f940d37af8/docs/use_cases/agents/baby_agi_with_agent.ipynb

import argparse
from pathlib import Path
from time import sleep
from typing import Optional

from loguru import logger
from sqlalchemy.orm import sessionmaker

from agi.agents.task_execution_creation_prioritization_agent import (
    TaskExecutionCreationPrioritizationAgent,
    Tools,
)
from lib.sql import ENGINE, SuperAgent, TaskListItem

# Set up the argument parser
parser = argparse.ArgumentParser(description="Configure the BabyAGI agent.")
parser.add_argument(
    "--config",
    type=Path,
    default=Path("config/default.yaml"),
    help="Path to the config file for smsAGI.",
)
args = parser.parse_args()

# Iterate over all super agents and run the task prioritization and task execution agents
_Session = sessionmaker(bind=ENGINE)
while True:
    # Then run the task execution agent with the top priority task
    with _Session() as session:
        super_agent = SuperAgent.get_random_super_agent(session=session)
        if super_agent is None:
            logger.warning("No super agents to run.")
            sleep(10)
            continue
        if super_agent.wait_for_response:
            logger.info(
                f"Super Agent: ({super_agent.id},{super_agent.name}) -> Waiting for response."
            )
            continue
        top_priority_task_item: Optional[
            TaskListItem
        ] = TaskListItem.get_top_task_list_item(
            session=session, super_agent=super_agent
        )
        if top_priority_task_item is None:
            logger.info(
                f"Super Agent: ({super_agent.id},{super_agent.name}) -> No tasks to execute."
            )
            continue
        TaskExecutionCreationPrioritizationAgent(
            super_agent=super_agent, session=session, config=args.config
        ).run(top_priority_task_item)
        session.commit()

import yaml
from pathlib import Path
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import SerpAPIWrapper
from typing import Optional
import os
from agi.twilio import send_message
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI

class Tools:
    DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config/default.yaml"

    def __init__(self, config: Path = None) -> None:
        """Loads the config file and the default config file."""
        self.config_file = config
        self.config = yaml.safe_load(config.read_text())
        self.default_config = yaml.safe_load(self.DEFAULT_CONFIG.read_text())

    def get_zero_shot_prompt(self) -> PromptTemplate:
        """A prompt that teaches the agi to use the tools."""
        config_tools = self.config.get("tools", {})
        default_tools = self.default_config["tools"]

        prompt = config_tools.get("prompt", {})
        default_prompt = default_tools["prompt"]

        prefix = prompt.get("prefix", default_prompt["prefix"])
        suffix = prompt.get("suffix", default_prompt["suffix"])
        input_variables = prompt.get("input_variables", default_prompt["input_variables"])

        return ZeroShotAgent.create_prompt(
            self.get_tools(),
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
        )

    def _get_search_tool(self) -> Optional[Tool]:
        """The search tool allows the agi to search google."""
        search_tool = self.config.get("tools", {}).get("search", {})
        default_search_tool = self.default_config["tools"]["search"]

        search_tool_enabled = search_tool.get("enabled", default_search_tool["enabled"])
        search_tool_description = search_tool.get("description", default_search_tool["description"])
        if search_tool_enabled:
            if "SERPAPI_API_KEY" not in os.environ:
                raise KeyError("SERPAPI_API_KEY environment variable not set.")

            search = SerpAPIWrapper()
            return Tool(description=search_tool_description, name="search", func=search.run) if search_tool_enabled else None
        return None

    def _get_todo_tool(self) -> Optional[Tool]:
        """The todo tool allows the agi to create todo items."""
        todo_tool = self.config.get("tools", {}).get("todo", {})
        default_todo_tool = self.default_config["tools"]["todo"]

        todo_tool_enabled = todo_tool.get("enabled", default_todo_tool["enabled"])
        todo_tool_description = todo_tool.get("description", default_todo_tool["description"])

        if todo_tool_enabled:
            todo_prompt = todo_tool.get("prompt", {}).get("template", default_todo_tool["prompt"]["template"])
            todo_input_variables = todo_tool.get("prompt", {}).get("input_variables", default_todo_tool["prompt"]["input_variables"])
            todo_prompt_template = PromptTemplate(template=todo_prompt, input_variables=todo_input_variables)
            todo_chain = LLMChain(llm=ChatOpenAI(temperature=0), prompt=todo_prompt_template)
            return Tool(description=todo_tool_description, name="todo", func=todo_chain.run)

        return None

    def _get_send_message_tool(self) -> Optional[Tool]:
        """The send message tool allows the agi to send a message to the user."""
        send_message_tool = self.config.get("tools", {}).get("send_message", {})
        default_send_message_tool = self.default_config["tools"]["send_message"]

        send_message_tool_enabled = send_message_tool.get("enabled", default_send_message_tool["enabled"])
        send_message_tool_description = send_message_tool.get("description", default_send_message_tool["description"])

        return Tool(description=send_message_tool_description, name="send_message", func=send_message) if send_message_tool_enabled else None

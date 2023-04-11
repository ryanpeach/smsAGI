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
    DEFAULT_PERSONALITY = Path(__file__).parent.parent.parent / "personalities/default.yaml"

    def __init__(self, personality: Path) -> None:
        """Loads the personality file and the default personality file."""
        self.personality_file = personality
        self.personality = yaml.safe_load(personality.read_text())
        self.default_personality = yaml.safe_load(self.DEFAULT_PERSONALITY.read_text())

    def get_tools(self) -> list[Tool]:
        """Gets a list of tools from the personality file."""
        tools = [
            self._get_search_tool(),
            self._get_todo_tool(),
            self._get_send_message_tool(),
        ]
        return [tool for tool in tools if tool is not None]

    def get_zero_shot_prompt(self) -> PromptTemplate:
        """A prompt that teaches the agi to use the tools."""
        personality_tools = self.personality.get("tools", {})
        default_tools = self.default_personality["tools"]

        prompt = personality_tools.get("prompt", {})
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
        search_tool = self.personality.get("tools", {}).get("search", {})
        default_search_tool = self.default_personality["tools"]["search"]

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
        todo_tool = self.personality.get("tools", {}).get("todo", {})
        default_todo_tool = self.default_personality["tools"]["todo"]

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
        send_message_tool = self.personality.get("tools", {}).get("send_message", {})
        default_send_message_tool = self.default_personality["tools"]["send_message"]

        send_message_tool_enabled = send_message_tool.get("enabled", default_send_message_tool["enabled"])
        send_message_tool_description = send_message_tool.get("description", default_send_message_tool["description"])

        return Tool(description=send_message_tool_description, name="send_message", func=send_message) if send_message_tool_enabled else None

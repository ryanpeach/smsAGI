import os
from pathlib import Path
from typing import List, Optional

import yaml
from langchain import LLMChain, PromptTemplate, SerpAPIWrapper
from langchain.agents import Tool, ZeroShotAgent
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.openai import ChatOpenAI

from agi.twilio import send_message
from lib.config import Config


class Tools(Config):
    def get_zero_shot_llm(self) -> ChatOpenAI:
        """The zero shot language model."""
        config_tools = self.config.get("tools", {})
        default_tools = self.default_config["tools"]

        prompt = config_tools.get("prompt", {})
        default_prompt = default_tools["prompt"]

        llm = prompt.get("llm", {})
        default_llm = default_prompt["llm"]

        llm_model_name = llm.get("model_name", default_llm["model_name"])
        llm_temperature = llm.get("temperature", default_llm["temperature"])

        return ChatOpenAI(model_name=llm_model_name, temperature=llm_temperature)  # type: ignore

    def get_zero_shot_prompt(self, tools: List[Tool]) -> PromptTemplate:
        """A prompt that teaches the agi to use the tools."""
        config_tools = self.config.get("tools", {})
        default_tools = self.default_config["tools"]

        prompt = config_tools.get("prompt", {})
        default_prompt = default_tools["prompt"]

        prefix = prompt.get("prefix", default_prompt["prefix"])
        suffix = prompt.get("suffix", default_prompt["suffix"])
        input_variables = prompt.get(
            "input_variables", default_prompt["input_variables"]
        )

        return ZeroShotAgent.create_prompt(
            tools=tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
        )

    def get_search_tool(self) -> Optional[Tool]:
        """The search tool allows the agi to search google."""
        search_tool = self.config.get("tools", {}).get("search", {})
        default_search_tool = self.default_config["tools"]["search"]

        search_tool_enabled = search_tool.get("enabled", default_search_tool["enabled"])
        search_tool_description = search_tool.get(
            "description", default_search_tool["description"]
        )
        if search_tool_enabled:
            if "SERPAPI_API_KEY" not in os.environ:
                raise KeyError("SERPAPI_API_KEY environment variable not set.")

            search = SerpAPIWrapper(
                search_engine="google", serpapi_api_key=os.environ["SERPAPI_API_KEY"]
            )
            return (
                Tool(
                    description=search_tool_description, name="search", func=search.run
                )
                if search_tool_enabled
                else None
            )
        return None

    def get_todo_tool(self) -> Optional[Tool]:
        """The todo tool allows the agi to create todo items."""
        todo_tool = self.config.get("tools", {}).get("todo", {})
        default_todo_tool = self.default_config["tools"]["todo"]

        todo_tool_enabled = todo_tool.get("enabled", default_todo_tool["enabled"])
        todo_tool_description = todo_tool.get(
            "description", default_todo_tool["description"]
        )

        if todo_tool_enabled:
            todo_prompt = todo_tool.get("prompt", {}).get(
                "template", default_todo_tool["prompt"]["template"]
            )
            todo_input_variables = todo_tool.get("prompt", {}).get(
                "input_variables", default_todo_tool["prompt"]["input_variables"]
            )
            todo_prompt_template = PromptTemplate(
                template=todo_prompt, input_variables=todo_input_variables
            )
            llm_prompt_template = todo_prompt.get(
                "llm", default_todo_tool["prompt"]["template"]["llm"]
            )
            llm_model_name = llm_prompt_template.get(
                "model_name",
                default_todo_tool["prompt"]["template"]["llm"]["model_name"],
            )
            llm_temperature = llm_prompt_template.get(
                "temperature",
                default_todo_tool["prompt"]["template"]["llm"]["temperature"],
            )
            llm = ChatOpenAI(model_name=llm_model_name, temperature=llm_temperature)  # type: ignore
            todo_chain = LLMChain(llm=llm, prompt=todo_prompt_template)
            return Tool(
                description=todo_tool_description, name="todo", func=todo_chain.run
            )

        return None

    def get_send_message_tool(self) -> Optional[Tool]:
        """The send message tool allows the agi to send a message to the user."""
        send_message_tool = self.config.get("tools", {}).get("send_message", {})
        default_send_message_tool = self.default_config["tools"]["send_message"]

        send_message_tool_enabled = send_message_tool.get(
            "enabled", default_send_message_tool["enabled"]
        )
        send_message_tool_description = send_message_tool.get(
            "description", default_send_message_tool["description"]
        )

        return (
            Tool(
                description=send_message_tool_description,
                name="send_message",
                func=send_message,
            )
            if send_message_tool_enabled
            else None
        )

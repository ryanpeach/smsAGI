class UserAgent:

    def __init__(self, agent: SuperAgent, tools: Tools, session: Session, llm: BaseLLM, vectorstore: VectorStoreMemory, task_list: TaskList):
        self.tools = tools.get_tools()
        self.vectorstore = vectorstore
        self.agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)
        self.agent = agent
        self.goals = Goals(agent=agent, session=session)

    def get_tools(self) -> list[Tool]:
        """Gets a list of tools from the personality file."""
        tools = [
            self.tools._get_todo_tool(),
            self.tools._get_send_message_tool(),
        ]
        return [tool for tool in tools if tool is not None]

    async def arun(self, user_msg: str):
        # Get tasks from the user
        objective = await self.goals.get_prompt()
        context = await self.vectorstore.get_top_tasks(query=objective, k=5)
        return await self.agent_chain.arun(objective=objective, context=context, task=user_msg)
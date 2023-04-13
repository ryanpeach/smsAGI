class UserAgent:

    def __init__(self, agent: SuperAgent, tools: Tools, session: Session, llm: BaseLLM, vectorstore: VectorStoreMemory, task_list: TaskList):
        self.tools = tools.get_tools()
        self.task_list = task_list
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

    def get_tasks_from_user(self):
        # Get tasks from the user
        user_messages = self.qaagent.receive_all_user_responses()
        for user_msg in user_messages:
            self.conversation_buffer.add_to_buffer(user_msg)
            self.agent_chain.run(objective=self.goals.get_prompt(), context=self.vectorstore.get_top_tasks(query=self.agent.objective, k=5), task=user_msg)
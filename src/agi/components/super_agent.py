import threading

class BabyAGI:
    """Controller model for the BabyAGI agent."""

    def __init__(self):
        # Rules for threading:
        # 1. Don't pass mutable arguments to functions or classes that are run in threads.
        # 2. SQLAlchemy is not thread safe, so we need to use a session per thread.
        self.user_agent = UserAgent()
        self.task_creation_chain = TaskCreationChain()
        self.task_prioritization_chain = TaskPrioritizationChain()
        self.task_execution_agent = TaskExecutionAgent()

    def start(self):
        """Run the BabyAGI agent."""
        # See if the user has any comments
        self.user_agent_thread = threading.Thread(target=self.user_agent.get_tasks_from_user)
        self.user_agent_thread.start()

        # Create the tasks
        self.task_creation_thread = threading.Thread(target=self.task_creation_chain.create_next_task)
        self.task_creation_thread.start()

        # Prioritize the tasks
        self.prioritization_thread = threading.Thread(target=self.task_prioritization_chain.prioritize_tasks)
        self.prioritization_thread.start()

        # Execute the tasks
        self.execution_thread = threading.Thread(target=self.task_execution_agent.run)
        self.execution_thread.start()

    def stop(self):
        """Stop the BabyAGI agent."""
        self.user_agent_thread.join()
        self.task_creation_thread.join()
        self.prioritization_thread.join()
        self.execution_thread.join()
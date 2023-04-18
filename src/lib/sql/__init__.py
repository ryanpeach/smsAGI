import datetime
import os
from typing import Callable, Generator, List, Optional

from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.engine import URL, Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship
from sqlalchemy.sql.expression import func

from lib.config.prompts import Prompts

Base = declarative_base()


# Define User table
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    phoneNumber = Column(String, unique=True)
    # Users can have more than one agent
    # But only one primary agent
    primary_agent_id = Column(Integer, ForeignKey("super_agents.id"))
    primary_agent: "SuperAgent" = relationship(
        "SuperAgent", foreign_keys=[primary_agent_id]
    )

    @staticmethod
    def get_from_phone_number(session: Session, phone_number: str) -> "User":
        """Get the user associated with this phone number."""
        out = session.query(User).filter(User.phoneNumber == phone_number).first()
        if isinstance(out, User):
            return out
        raise KeyError("No user found.")


# Define Agent table
class SuperAgent(Base):
    """A super agent is a set of agents that work together to achieve a goal."""

    __tablename__ = "super_agents"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"))
    user: User = relationship("User", foreign_keys=[user_id])
    wait_for_response = Column(Boolean, default=False)

    @staticmethod
    def get_random_super_agent(session: Session) -> Optional["SuperAgent"]:
        """Get a random super agent."""
        out = session.query(SuperAgent).order_by(func.random()).first()
        if isinstance(out, SuperAgent):
            return out
        elif out is None:
            return None
        raise TypeError("Unexpected type.")


# Define Goal table
class Goal(Base):
    __tablename__ = "goals"
    id = Column(Integer, primary_key=True)
    super_agent_id = Column(Integer, ForeignKey("super_agents.id"))
    super_agent: SuperAgent = relationship("SuperAgent")
    objective = Column(String)

    @staticmethod
    def get_goals(session: Session, agent: Optional[SuperAgent]) -> List["Goal"]:
        """Get all goals associated with this super agent."""
        if agent is None:
            return session.query(Goal).all()
        return session.query(Goal).filter(Goal.super_agent_id == agent.id).all()

    @staticmethod
    def get_prompt(goals: List["Goal"], prompts: Prompts) -> str:
        """Get the goals as a bulleted list."""
        out = ""
        for goal in goals:
            out += f"* {goal}"
        prompt_template, _ = prompts.get_goals()
        return prompt_template.format(goals=goals)


# Define ThreadItem table
class ThreadItem(Base):
    __tablename__ = "thread_items"
    id = Column(Integer, primary_key=True)
    super_agent_id = Column(Integer, ForeignKey("super_agents.id"))
    super_agent: SuperAgent = relationship("SuperAgent")
    role = Column(
        String,
        CheckConstraint(
            "role IN ('user', 'assistant', 'system')",
            name="check_role_in_allowed_values",
        ),
        nullable=False,
    )
    content = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    @staticmethod
    def create(
        session: Session, super_agent: SuperAgent, msg: BaseMessage
    ) -> "ThreadItem":
        """Create a new thread item."""
        if isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            raise TypeError(f"Unknown message type. {type(msg)}")
        thread_item = ThreadItem(
            super_agent=super_agent, role=role, content=msg.content
        )
        session.add(thread_item)
        return thread_item

    @staticmethod
    def get_all(
        session: Session, super_agent: Optional[SuperAgent] = None
    ) -> List[BaseMessage]:
        """Get all thread items associated with this super agent."""
        if super_agent is None:
            out = session.query(ThreadItem).order_by(ThreadItem.created_at.asc()).all()
        else:
            out = (
                session.query(ThreadItem)
                .filter(ThreadItem.super_agent_id == super_agent.id)
                .order_by(ThreadItem.created_at.asc())
                .all()
            )
        out_base_msg: List[BaseMessage] = []
        for thread_item in out:
            if thread_item.role == "assistant":
                out_base_msg.append(AIMessage(content=thread_item.content))
            elif thread_item.role == "user":
                out_base_msg.append(HumanMessage(content=thread_item.content))
            elif thread_item.role == "system":
                out_base_msg.append(SystemMessage(content=thread_item.content))
            else:
                raise TypeError(f"Unknown role. {thread_item.role}")
        return out_base_msg


# Define TaskListItem table
class TaskListItem(Base):
    __tablename__ = "task_list_items"
    id = Column(Integer, primary_key=True)
    super_agent_id = Column(Integer, ForeignKey("super_agents.id"))
    super_agent: SuperAgent = relationship("SuperAgent")
    description = Column(String, nullable=False)
    priority = Column(Float, default=0.5, nullable=False)
    earliest_start_time = Column(DateTime, default=None, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, default=None, nullable=True)

    @staticmethod
    def get_random_task_list_item(
        session: Session, k: int = 10, agent: Optional[SuperAgent] = None
    ) -> List["TaskListItem"]:
        """Get a random sample of k tasks."""
        if agent is not None:
            return (
                session.query(TaskListItem)
                .filter(TaskListItem.super_agent == agent)
                .filter(TaskListItem.completed_at != None)
                .order_by(func.random())
                .limit(k)
                .all()
            )
        return (
            session.query(TaskListItem)
            .filter(TaskListItem.completed_at != None)
            .order_by(func.random())
            .limit(k)
            .all()
        )

    @staticmethod
    def get_top_task_list_item(
        session: Session, super_agent: Optional[SuperAgent] = None
    ) -> Optional["TaskListItem"]:
        """Get the task with the highest priority."""
        if super_agent is not None:
            out = (
                session.query(TaskListItem)
                .filter(TaskListItem.super_agent == super_agent)
                .filter(TaskListItem.completed_at != None)
                .order_by(TaskListItem.priority.desc())
                .first()
            )
        else:
            out = (
                session.query(TaskListItem)
                .filter(TaskListItem.completed_at != None)
                .order_by(TaskListItem.priority.desc())
                .first()
            )

        if out is None:
            return None
        elif isinstance(out, TaskListItem):
            return out
        else:
            raise ValueError("Unexpected type returned from query")

    def set_priority(self, priority: float):
        """
        Set the priority of this task.
        Priority must be a float between 0 and 1.
        """
        if priority <= 1 or priority >= 0:
            raise ValueError("Priority must be between 0 and 1")
        self.priority = priority

    def set_completed(self):
        """Set the completed time to now."""
        if self.completed_at is not None:
            raise ValueError("This task has already been completed")
        self.completed_at = datetime.datetime.utcnow()

    @staticmethod
    def task_list_to_table(task_list: List["TaskListItem"]) -> str:
        """Convert a list of tasks to a string as a markdown table."""
        sort_func: Callable[[TaskListItem], float] = lambda x: float(
            x.priority if x.priority is not None else 0.5
        )
        sorted_task_list = sorted(task_list, key=sort_func, reverse=True)
        task_list_str = "| Task | Priority (higher priority) |"
        task_list_str += "| --- | --- |"
        for task in sorted_task_list:
            task_list_str += f"| {task.description} | {task.priority} |"
        return task_list_str


def create_engine_from_env() -> Engine:
    POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]
    POSTGRES_USER = os.environ["POSTGRES_USER"]
    POSTGRES_DB = os.environ["POSTGRES_DB"]
    POSTGRES_URL = os.environ["POSTGRES_URL"]

    url_object = URL.create(
        "postgresql",
        username=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_URL,
        database=POSTGRES_DB,
    )

    return create_engine(url_object)


# Create the tables in the database
ENGINE = create_engine_from_env()
Base.metadata.create_all(ENGINE)

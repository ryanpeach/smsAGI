import json
import threading

from rich import Console
from rich.Markdown import Markdown

from langchain.chat_models.openai import BaseMessage
from lib.redis import connect_to_redis

def pretty_format_message(message: BaseMessage) -> Markdown:
    role = message["role"]
    content = message["content"]

    formatted_message = f"**[{role.name.capitalize()}]**: {content}"
    return Markdown(formatted_message)


class QAClient:
    """
    The model used by the user to get questions from the Auto GPT Instance and answer them.
    This model involved thread safe operations to modify self.unanswered_questions
    """

    def __init__(self) -> None:
        self.redis = connect_to_redis()

    def receive_continuous(
        self, console: Console, console_lock: threading.Lock
    ) -> None:
        """Continuously meant to check for new messages from the Auto GPT Instance and add them to the list of questions that haven't been answered yet. Meant to run in a thread."""
        # Try to get a message from the queue
        while True:
            if self.redis.llen("touser") > 0:
                message = self.redis.rpop("touser")
                msg = pretty_format_message(BaseMessage("assistant", message))
                with console_lock:
                    # with open(LOG_PATH, "a") as f:
                    #     f.write(json.dumps(msg) + "\n")
                    console.print(msg)

    def send_message(self, message: str) -> None:
        self.redis.lpush("togpt", message)

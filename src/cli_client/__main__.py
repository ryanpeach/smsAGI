import argparse
import datetime
import json
import os
import threading
from pathlib import Path
from typing import List
from uuid import uuid4

from colorama import init
from rich import print
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from cli_client.redis import QAClient

# Create an argparse that has the boolean parameter --speak
parser = argparse.ArgumentParser(description="Process arguments.")
parser.add_argument("--speak", action="store_true", help="Enable Speak Mode")
args = parser.parse_args()

# Get the current timestamp
now = datetime.datetime.now()

# Format the timestamp as a string
timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")

# Create a Path object with the timestamp
path = Path("outputs/logs")
assert path.is_dir(), "The logs directory doesn't exist."
LOG_PATH = path / f"{timestamp}.jsonl"
assert not path.is_file(), "Somehow the log file already exists."


def main():
    init(autoreset=True)  # Initialize colorama

    qa_client = QAClient()

    console = Console()
    console_lock = threading.Lock()

    receive_thread = threading.Thread(
        target=qa_client.receive_continuous, args=(console, console_lock)
    )
    receive_thread.daemon = True
    receive_thread.start()

    console.print("Welcome to the Auto GPT Client!")
    console.print(
        "Type your messages below and press enter to send them to the Auto GPT Instance."
    )
    console.print('You can also type "exit" to exit the client.')
    console.print("")

    while True:
        message = console.input("")
        if message:
            if message == "exit":
                break
            qa_client.send_message(message)
            with console_lock:
                msg = create_chat_message("user", message)
                with open(LOG_PATH, "a") as f:
                    f.write(json.dumps(msg))

    console.print("Exiting client...")


if __name__ == "__main__":
    main()

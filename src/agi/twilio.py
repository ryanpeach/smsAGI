# Download the helper library from https://www.twilio.com/docs/python/install
import os
from twilio.rest import Client
from agi.redis import QAAgent
from langchain.agents import tool

# Set environment variables for your credentials
# Read more at http://twil.io/secure
account_sid = os.environ["TWILIO_ACCOUNT_SID"]
auth_token = os.environ["TWILIO_AUTH_TOKEN"]
client = Client(account_sid, auth_token)
qaagent = QAAgent()


@tool
def send_message(message: str, wait_for_response: bool) -> str:
    """
    Sends a message to the user and returns a message to the gpt agent.
    If wait_for_response is True, the gpt agent will wait for a response from the user, and the response message will be from the user.
    If wait_for_response is False, the gpt agent will not wait for a response from the user, and the response message will be a generic system message.
    """
    client.messages.create(
        body=message,
        from_=os.environ["TWILIO_FROM_PHONE_NUMBER"],
        to=os.environ["TWILIO_TO_PHONE_NUMBER"],
    )
    if not wait_for_response:
        return "You have sent the message to the user. You may or may not receive a response. You may ask other questions without waiting for a response. You may also send other messages to the user without waiting for a response. "
    else:
        return qaagent.wait()

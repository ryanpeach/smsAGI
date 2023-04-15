# Download the helper library from https://www.twilio.com/docs/python/install
import os

from langchain.agents import tool
from langchain.schema import AIMessage
from sqlalchemy.orm import Session
from twilio.rest import Client

from lib.sql import SuperAgent, ThreadItem

# Set environment variables for your credentials
# Read more at http://twil.io/secure
account_sid = os.environ["TWILIO_ACCOUNT_SID"]
auth_token = os.environ["TWILIO_AUTH_TOKEN"]
client = Client(account_sid, auth_token)

WAITING_FOR_USER_RESPONSE = "Waiting for user response..."


@tool
def send_message(
    *, session: Session, super_agent: SuperAgent, message: str, wait_for_response: bool
) -> str:
    """
    Sends a message to the user and returns a message to the gpt agent.
    If wait_for_response is True, the gpt agent will wait for a response from the user, and the response message will be from the user.
    If wait_for_response is False, the gpt agent will not wait for a response from the user, and the response message will be a generic system message.
    """
    message = str(message)
    if wait_for_response:
        message += " (Agent will wait for a response from the user.)"
    client.messages.create(
        body=message,
        from_=os.environ["TWILIO_FROM_PHONE_NUMBER"],
        to=os.environ["TWILIO_TO_PHONE_NUMBER"],
    )
    ThreadItem.create(
        session=session, super_agent=super_agent, msg=AIMessage(content=message)
    )
    if not wait_for_response:
        return "You have sent the message to the user. You may or may not receive a response. You may ask other questions without waiting for a response. You may also send other messages to the user without waiting for a response. "
    else:
        super_agent.wait_for_response = True
        return WAITING_FOR_USER_RESPONSE

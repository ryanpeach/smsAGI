import os

from flask import Flask, request
from sqlalchemy.orm import sessionmaker
from twilio.rest import Client

from lib.sql import Goal, SuperAgent, ThreadItem, User, create_engine_from_env
from server.agents.user_agent import UserAgent

# On startup, we need to create a couple objects
engine = create_engine_from_env()
_Session = sessionmaker(bind=engine)

app = Flask(__name__)
CONFIG = None


@app.route("/health")
async def health():
    return {
        "statusCode": 200,
        "body": "Healthy!",
    }


@app.route("/sms", methods=["POST"])
async def sms_reply():
    # Get the message from the request
    incoming_msg = request.values.get("Body", "").lower()
    phone_number = incoming_msg["From"]
    user = User.get_from_phone_number(phone_number)
    with _Session() as session:
        ThreadItem(super_agent=user.primary_agent, content=incoming_msg, role="user")
        user_agent = UserAgent(
            super_agent=user.primary_agent, session=session, config=CONFIG
        )
        await user_agent.arun(incoming_msg)

        # Reset wait_for_response upon receiving a message
        user.primary_agent.wait_for_response = False

        session.commit()

    return {
        "statusCode": 200,
        "body": "Message received!",
    }


if __name__ == "__main__":
    # On startup, we need to create a couple objects if they don't exist
    user_phone_number = os.environ["TWILIO_TO_PHONE_NUMBER"]
    with _Session() as session:
        user = User.get_from_phone_number(
            session=session, phone_number=user_phone_number
        )
        if user is None:
            user = User(name="admin", phone_number=user_phone_number)
            super_agent = SuperAgent(name="admin", user=user)
            goal = Goal(
                super_agent=super_agent, objective="Ask the user what they want of you."
            )
            user.primary_agent = super_agent
            session.add(user)
            session.add(super_agent)
        session.commit()
    app.run(debug=True)

from flask import Flask, request
from server.redis import QAClient
from twilio.rest import Client

app = Flask(__name__)


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
    phone_number = request.values.get("From", "")
    user = User.get_from_phone_number(phone_number)
    user_agent = UserAgent(agent=user.primary_agent)
    await user_agent.arun(incoming_msg)

    return {
        "statusCode": 200,
        "body": "Message sent!",
    }


if __name__ == "__main__":
    app.run(debug=True)

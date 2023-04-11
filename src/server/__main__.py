from flask import Flask, request
from server.redis import QAClient
from twilio.rest import Client

app = Flask(__name__)


@app.route("/health")
def health():
    return {
        "statusCode": 200,
        "body": "Healthy!",
    }


@app.route("/sms", methods=["POST"])
def sms_reply():
    # Get the message from the request
    incoming_msg = request.values.get("Body", "").lower()

    qaclient = QAClient()
    qaclient.send_message(incoming_msg)

    return {
        "statusCode": 200,
        "body": "Message sent!",
    }


if __name__ == "__main__":
    app.run(debug=True)

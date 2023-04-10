import os
import argparse
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from smsgpt.babyagi import BabyAGI, OpenAI, FAISS, InMemoryDocstore, OpenAIEmbeddings
import faiss

# Set up the argument parser
parser = argparse.ArgumentParser(description="Configure the BabyAGI agent.")
parser.add_argument(
    "--sms",
    type=str,
    default="",
    help="Phone number to send SMS messages to. Otherwise, just use the CLI client.",
)
parser.add_argument(
    "--objective",
    type=str,
    default="Write a weather report for SF today",
    help="Initial objective for BabyAGI.",
)
parser.add_argument(
    "--temperature", type=float, default=0, help="Temperature for the OpenAI model."
)
parser.add_argument(
    "--max_iterations",
    type=int,
    default=3,
    help="Maximum number of iterations for the BabyAGI agent.",
)
parser.add_argument(
    "--embedding_size",
    type=int,
    default=1536,
    help="Size of the embeddings for the FAISS index.",
)
parser.add_argument(
    "--verbose", action="store_true", help="Enable verbose output for BabyAGI."
)
args = parser.parse_args()

app = Flask(__name__)

# Initialize the BabyAGI agent using the parsed arguments
index = faiss.IndexFlatL2(args.embedding_size)
embeddings_model = OpenAIEmbeddings()
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
llm = OpenAI(temperature=args.temperature)
baby_agi = BabyAGI.from_llm(
    llm, vectorstore, verbose=args.verbose, max_iterations=args.max_iterations, logger=twilio_client
)


@app.route("/sms", methods=["POST"])
def sms_reply():
    # Get the message from the request
    incoming_msg = request.values.get("Body", "").lower()

    # Create a response object
    resp = MessagingResponse()

    # Run the BabyAGI agent with the incoming message as the objective
    baby_agi.add_task({"objective": incoming_msg})

    return ""


if __name__ == "__main__":
    with baby_agi:
        app.run(debug=True)

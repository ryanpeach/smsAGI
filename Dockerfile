FROM ngrok:debian

# Install python
RUN apt-get update && apt-get install -y python3 python3-pip

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Run the flask api server
COPY smsgpt .
COPY .env .
CMD ["python", "-m", "smsgpt"]
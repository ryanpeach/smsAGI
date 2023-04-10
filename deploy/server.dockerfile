FROM python:3.10-slim-buster

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Run the server
COPY src/server .
CMD ["python", "-m", "src/server"]
FROM python:3.10-slim-buster

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Run the server
COPY config config
COPY src/lib lib
COPY src/server server
ENV PYTHONPATH=.
CMD ["python", "-m", "server"]

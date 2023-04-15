FROM python:3.10-slim-buster

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Run the agi agent
COPY config config
COPY src/lib lib
COPY src/agi agi
ENV PYTHONPATH=.
CMD ["python", "-m", "agi"]

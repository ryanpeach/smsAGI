FROM python:3.10-slim-buster

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Run the agi agent
COPY src/agi .
CMD ["python", "-m", "src/agi"]
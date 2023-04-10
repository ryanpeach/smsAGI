# SMS GPT

**DRAFT: Not currently working.**

An implementation of BabyGPT using Langchain that can communicate via SMS to the user.

# Setup

## OpenAI

1. Sign up for an OpenAI account at https://beta.openai.com/ and create an API key.
2. Create a `.env` file in the root directory of the project and add the following:

```
OPENAI_API_KEY=<your OpenAI API key>
```

## Ngrok

1. Download and install ngrok from https://ngrok.com/download.
2. Create a `.env` file in the root directory of the project and add the following:

```
NGROK_AUTH_TOKEN=<your ngrok auth token>
```

## Twilio

1. Sign up for a Twilio account at https://www.twilio.com/try-twilio.
2. Verify your phone number and get a Twilio phone number from the Twilio Console.
3. Create a `.env` file in the root directory of the project and add the following:
4. Configure your Twilio phone number to use the webhook.
5. Go to the Twilio Console, find your phone number, and set the Messaging webhook URL to the ngrok URL you copied earlier, followed by /sms. For example: https://your-ngrok-url.ngrok.io/sms.

```
TWILIO_ACCOUNT_SID=<your Twilio account SID>
TWILIO_AUTH_TOKEN=<your Twilio auth token>
TWILIO_PHONE_NUMBER=<your Twilio phone number>
```

## Docker

Now you are ready to build and run!

On Linux:

```bash
docker build -t smsgpt .
docker run --net=host -it smsgpt http 80
```

On Windows or Mac:

```bash
docker build -t smsgpt .
docker run -it smsgpt http host.docker.internal:80
```

# Contributing

Please install the requirements:

`pip install -r requirements.txt -r dev-requirements.txt`

Please set up and run pre-commit hooks before contributing:

`pre-commit install`

We are also `mypy` compliant, so please run `mypy` before committing:

`mypy .`

We will work on testing, run `pytest` to run the tests.

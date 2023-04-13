# smsAGI

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
2. Create a `ngrok.yml` file in the root directory of the project. See https://ngrok.com/docs/ngrok-agent/config/ and `ngrok.yml.template` for more information.
3. If you pay for a premium ngrok account, you can use a persistent URL by adding a `hostname` to your `ngrok.yml` file


## Twilio

1. Sign up for a Twilio account at https://www.twilio.com/try-twilio.
2. Verify your phone number and get a Twilio phone number from the Twilio Console.
3. Create a `.env` file in the root directory of the project and add the following:
4. Configure your Twilio phone number to use the webhook.
5. Go to the Twilio Console, find your phone number, and set the Messaging webhook URL to the ngrok URL you copied earlier, followed by /sms. For example: https://your-ngrok-url.ngrok.io/sms.

```
TWILIO_ACCOUNT_SID=<your Twilio account SID>
TWILIO_AUTH_TOKEN=<your Twilio auth token>
TWILIO_FROM_PHONE_NUMBER=<your Twilio phone number>
TWILIO_TO_PHONE_NUMBER=<your verified phone number>
```

## Serpapi

If you would like to search the internet, please provide a SERPAPI_API_KEY in your `.env` file.

```
SERPAPI_API_KEY=<your Serpapi API key>
```

## Docker Compose

Now you are ready to build and run!

`docker-compose up --build`

# Contributing

Please install the requirements:

`pip install -r requirements.txt -r dev-requirements.txt`

Please set up and run pre-commit hooks before contributing:

`pre-commit install`

We are also `mypy` compliant, so please run `mypy` before committing:

`mypy .`

We will work on testing, run `pytest` to run the tests.

# Debugging

## Tracing

To start tracing, set the `LANGCHAIN_HANDLER=langchain` environment variable.

Then run `langchain-server` to start the server.

Then to run just the AGI, run `PYTHONPATH=src python -m agi`
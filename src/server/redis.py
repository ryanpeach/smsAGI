from lib.redis import connect_to_redis


class QAClient:
    """
    The model used by the user to get questions from the Auto GPT Instance and answer them.
    This model involved thread safe operations to modify self.unanswered_questions
    """

    def __init__(self) -> None:
        self.redis = connect_to_redis()

    def send_message(self, message: str) -> None:
        self.redis.lpush("togpt", message)

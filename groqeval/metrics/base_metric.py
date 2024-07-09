import logging
from groq import Groq

class BaseMetric:
    """
    The Base Metric class.
    """
    def __init__(self, groq_client: Groq, verbose: bool = None):
        self.groq_client = groq_client
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)


    def groq_chat_completion(self, messages, model, temperature=0.5, response_format=None):
        """
        Groq's chat completion API
        """
        chat_completion = self.groq_client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            response_format=response_format
        )
        return chat_completion

    def check_data_types(self, **kwargs):
        """
        Checks for empty strings in the arguments
        """
        for key, value in kwargs.items():
            if key != "verbose":
                if key != "context":
                    if value == "":
                        raise ValueError(f"'{key}' cannot be an empty string.")
                    if not isinstance(value, str):
                        raise TypeError(f"'{key}' must be a string")
                else:
                    if len(value) == 0:
                        raise ValueError(f"'{key}' cannot be an empty list.")
                    if not isinstance(value, list):
                        raise TypeError(f"'{key}' must be a list of strings")
                    else:
                        if not all(isinstance(item, str) for item in value):
                            raise TypeError(f"All items in '{key}' must be strings")



    def score(self):
        raise NotImplementedError("This method should be overridden by subclasses")

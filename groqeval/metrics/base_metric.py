import logging
import statistics
from abc import ABC,abstractmethod
from groq import Groq

class BaseMetric(ABC):
    """
    The Base Metric class.
    """
    def __init__(self, groq_client: Groq, verbose: bool = None):
        self.groq_client = groq_client
        self.aggregation = statistics.mean
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()  # Stream handler to output to the console
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.propagate = False
        
        if verbose:
            self.logger.setLevel(logging.INFO)  # Set to DEBUG to see all levels of logs
            self.logger.info("Verbose Mode is on.")
        else:
            self.logger.setLevel(logging.WARNING)

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
                        
    @property
    @abstractmethod
    def scoring_function(self):
        """
        This property should be implemented by each child class
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def score(self, aggregation = None):
        """
        Aggregation of individual scores and final result.
        """
        if aggregation is not None:
            self.aggregation = aggregation
        scored_output, output_dictionary = self.scoring_function()
        if scored_output.scores:
            average_score = self.aggregation([output.score for output in scored_output.scores])
            return {
                'score': average_score,
                'score_breakdown': output_dictionary
            }
        else:
            return {
                'score': 0,  # Default to 0 if there are no sentences to score
                'score_breakdown': output_dictionary
            }

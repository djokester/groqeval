# groqeval/client.py
import importlib
import pkgutil
from groq import Groq
from .metrics.base_metric import BaseMetric

class GroqEval:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)

    def __call__(self, metric_name, **kwargs):
        try:
            metric_module = importlib.import_module(f"groqeval.metrics.{metric_name}")
            class_name = ''.join(word.capitalize() for word in metric_name.split('_'))
            metric_class = getattr(metric_module, class_name)

            # Check if the class is a subclass of BaseMetric and not BaseMetric itself
            if issubclass(metric_class, BaseMetric) and metric_class is not BaseMetric:
                return metric_class(self.client, **kwargs)
            raise TypeError(f"{class_name} is not a valid metric class")

        except (ImportError, AttributeError, TypeError) as e:
            raise ValueError(f"No valid metric found for: {metric_name}") from e
        
    def list_metrics(self):
        metric_list = []
        # Assuming metrics are in groqeval/metrics directory
        package = 'groqeval.metrics'
        for finder, name, ispkg in pkgutil.iter_modules([package.replace('.', '/')]):
            if not ispkg:
                module = importlib.import_module(f"{package}.{name}")
                for attribute_name in dir(module):
                    attribute = getattr(module, attribute_name)
                    if isinstance(attribute, type) and issubclass(attribute, BaseMetric) and attribute is not BaseMetric:
                        metric_list.append(attribute.__name__)

        return metric_list
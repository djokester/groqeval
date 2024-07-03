import pytest
from groqeval import GroqEval
import os
import inspect
import random
import string
from groq import Groq
from typing import List, Dict, get_origin, get_args

@pytest.fixture(scope="session")
def evaluator():
    """Create a reusable GroqEval object with a real client for use in all tests."""
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("API key must be set as an environment variable 'GROQ_API_KEY'")
    evaluator = GroqEval(api_key=api_key)
    return evaluator

@pytest.fixture()
def metrics_folder():
    return "groqeval/metrics"

@pytest.fixture()
def metrics_module():
    return "groqeval.metrics"

def get_class_args(cls):
    init_signature = inspect.signature(cls.__init__)
    params = init_signature.parameters
    args = {name: param for name, param in params.items() if name not in ['self', "groq_client"]}
    return args

def generate_random_value(param):
    annotation = param.annotation
    if annotation == int or isinstance(param.default, int):
        return random.randint(0, 100)
    elif annotation == float or isinstance(param.default, float):
        return random.uniform(0.0, 100.0)
    elif annotation == str or isinstance(param.default, str):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    elif annotation == bool or isinstance(param.default, bool):
        return random.choice([True, False])
    elif get_origin(annotation) == list or isinstance(param.default, list):
        item_type = get_args(annotation)[0] if get_args(annotation) else None
        if item_type == int:
            return [random.randint(0, 10) for _ in range(5)]
        elif item_type == float:
            return [random.uniform(0.0, 10.0) for _ in range(5)]
        elif item_type == str:
            return [''.join(random.choices(string.ascii_letters + string.digits, k=8)) for _ in range(5)]
        else:
            return [random.randint(0, 10) for _ in range(5)]
    elif get_origin(annotation) == dict or isinstance(param.default, dict):
        return {f'key{i}': random.randint(0, 10) for i in range(5)}
    else:
        return None
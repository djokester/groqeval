import os
import importlib
import pytest
from conftest import get_class_args, generate_random_value

def metricize(file_name: str):
    """
        Converts a file name to a a metric class name
    """
    return ''.join(word.capitalize() for word in file_name.split('_'))

def test_list_metrics(evaluator, metrics_folder):
    files = os.listdir(metrics_folder)
    metrics = sorted([metricize(file[:-3]) for file in files if ".py" in file and file not in ["base_metric.py", "__init__.py"]])
    list_metrics = sorted(evaluator.list_metrics())
    assert metrics == list_metrics

def test_load_metrics(evaluator, metrics_folder, metrics_module):
    files = os.listdir(metrics_folder)
    metric_modules = sorted([file[:-3] for file in files if ".py" in file and file not in ["base_metric.py", "__init__.py"]])
    for module_name in metric_modules:
        module_path = f'{metrics_module}.{module_name}'
        module = importlib.import_module(module_path)
        class_name = metricize(module_name)
        
        class_ = getattr(module, class_name)
        class_args = get_class_args(class_)
        random_args = {name: generate_random_value(param) for name, param in class_args.items()}
        assert type(evaluator(module_name, **random_args)) == class_

def test_load_base_metric(evaluator, metrics_module):
    module_name = "base_metric"
    module_path = f'{metrics_module}.{"base_metric"}'
    module = importlib.import_module(module_path)
    class_name = metricize(module_name)

    class_ = getattr(module, class_name)
    class_args = get_class_args(class_)
    random_args = {name: generate_random_value(param) for name, param in class_args.items()}
    with pytest.raises(TypeError, match=f"{class_name} is not a valid metric class"):
        base_metric = evaluator(module_name, **random_args)
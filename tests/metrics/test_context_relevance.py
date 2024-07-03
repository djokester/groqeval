import pytest
import math

@pytest.mark.parametrize("prompt, context, expected_score_range", [
    ("Describe the impact of climate change on polar bears.", 
     ["The latest smartphone has a battery life of 20 hours. There was a concert last night."],
     (0, 3)),  # Expected to be completely irrelevant

    ("Explain the benefits of a balanced diet.", 
     ["A balanced diet includes a variety of nutrients. Traveling to different places can be exciting."],
     (3, 5)),  # Marginally relevant

    ("What are the economic benefits of renewable energy?", 
     ["Renewable energy sources might be the future. In some areas, renewable energy is still expensive to implement."],
     (5, 7)),  # Moderately relevant

    ("How does AI improve medical diagnostics?", 
     ["AI can analyze medical images with high accuracy. It helps doctors diagnose diseases faster."],
     (7, 10))  # Highly relevant
])
def test_context_relevance_scoring(evaluator, prompt, context, expected_score_range):
    context_relevance = evaluator("context_relevance", context=context, prompt=prompt)
    result = context_relevance.score()
    assert math.ceil(result['score']) >= expected_score_range[0] and math.floor(result['score']) <= expected_score_range[1], f"Score {result['score']} not in range {expected_score_range}"

def test_context_relevance_empty_context(evaluator):
    prompt = "What are the benefits of meditation?"
    context = []
    with pytest.raises(ValueError, match="'context' cannot be an empty list"):
        context_relevance = evaluator("context_relevance", context=context, prompt=prompt)
        context_relevance.score()

def test_context_relevance_empty_prompt(evaluator):
    prompt = ""
    context = ["Meditation helps in reducing stress and improving focus."]
    with pytest.raises(ValueError, match="'prompt' cannot be an empty string"):
        context_relevance = evaluator("context_relevance", context=context, prompt=prompt)
        context_relevance.score()

def test_context_relevance_invalid_context_type(evaluator):
    prompt = "What are the health benefits of yoga?"
    context = "Yoga improves flexibility and reduces stress."  # Non-list context
    with pytest.raises(TypeError, match="'context' must be a list of strings"):
        context_relevance = evaluator("context_relevance", context=context, prompt=prompt)
        context_relevance.score()

def test_context_relevance_invalid_prompt_type(evaluator):
    prompt = 12345  # Non-string prompt
    context = ["Yoga improves flexibility and reduces stress."]
    with pytest.raises(TypeError, match="'prompt' must be a string"):
        context_relevance = evaluator("context_relevance", context=context, prompt=prompt)
        context_relevance.score()

def test_context_relevance_non_string_in_context(evaluator):
    prompt = "What are the benefits of yoga?"
    context = ["Yoga improves flexibility.", 12345]  # Non-string item in context list
    with pytest.raises(TypeError, match="All items in 'context' must be strings"):
        context_relevance = evaluator("context_relevance", context=context, prompt=prompt)
        context_relevance.score()

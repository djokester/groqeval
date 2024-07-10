import pytest
import math
import statistics

@pytest.mark.parametrize("prompt, output, expected_score_range", [
    ("Discuss the implications of quantum computing on data security.", 
     "The weather forecast for today is quite breezy. Yesterday's weather was quite rainy.",
     (0, 3)),  # Expected to be completely irrelevant

    ("How can renewable energy impact global warming?", 
     "Renewable energy sources like solar panels help households reduce electricity bills. Global warming needs serious attention.",
     (3, 5)),  # Marginally relevant

    ("Explain how machine learning algorithms improve customer service.", 
     "Machine learning algorithms can automate responses in customer service, speeding up the process. They are also used in finance for predicting stock trends.",
     (5, 7)),  # Moderately relevant

    ("What role does AI play in healthcare?", 
     "AI aids in diagnosing diseases quickly and with high accuracy using data analysis. It's particularly effective in areas like imaging and patient data management.",
     (7,10))  # Highly relevant
])
def test_answer_relevance_scoring(evaluator, prompt, output, expected_score_range):
    answer_relevance = evaluator("answer_relevance", prompt=prompt, output=output)
    result = answer_relevance.score()
    assert math.ceil(result['score']) >= expected_score_range[0] and math.floor(result['score']) <= expected_score_range[1], f"Score {result['score']} not in range {expected_score_range}"

    max_score = answer_relevance.score(max)['score']
    mean_score = answer_relevance.score(statistics.mean)['score']
    min_score = answer_relevance.score(min)['score']
    
    assert max_score >= mean_score, f"Max score {max_score} is not greater than mean score {mean_score}"
    assert min_score <= mean_score, f"Min score {min_score} is not less than mean score {mean_score}"

def test_answer_relevance_empty_output(evaluator):
    prompt = "What are the benefits of meditation?"
    output = ""
    with pytest.raises(ValueError, match="'output' cannot be an empty string."):
        answer_relevance = evaluator("answer_relevance", prompt=prompt, output=output)
        answer_relevance.score()

def test_answer_relevance_empty_prompt(evaluator):
    prompt = ""
    output = "Meditation helps in reducing stress and improving focus."
    with pytest.raises(ValueError, match="'prompt' cannot be an empty string."):
        answer_relevance = evaluator("answer_relevance", prompt=prompt, output=output)
        answer_relevance.score()

def test_answer_relevance_invalid_output_type(evaluator):
    prompt = "What are the health benefits of yoga?"
    output = 12345  # Non-string output
    with pytest.raises(TypeError, match="'output' must be a string"):
        answer_relevance = evaluator("answer_relevance", prompt=prompt, output=output)
        answer_relevance.score()

def test_answer_relevance_invalid_prompt_type(evaluator):
    prompt = 12345  # Non-string prompt
    output = "Yoga improves flexibility and reduces stress."
    with pytest.raises(TypeError, match="'prompt' must be a string"):
        answer_relevance = evaluator("answer_relevance", prompt=prompt, output=output)
        answer_relevance.score()
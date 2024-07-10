import pytest
import math
import statistics

@pytest.mark.parametrize("context, output, expected_score_range", [
    (["Electric vehicles (EVs) are powered by electricity stored in batteries.", "EVs help reduce carbon emissions."], 
     "Electric vehicles run on gasoline and contribute significantly to carbon emissions.",
     (0, 3)),  # Complete contradiction

    (["Electric vehicles (EVs) are powered by electricity stored in batteries.", "EVs will in time lead to lithum shortages even though they solve carbon emmisions"], 
     "Wind energy has somewhat removed the EVs lithium shortage crisis. EVs over-reliance on batteries led us here",
     (3, 5)),  # Introducing elements not supported by context

    (["Electric vehicles (EVs) are powered by electricity stored in batteries.", "EVs help reduce carbon emissions."], 
     "Electric vehicles help reduce carbon emissions. They are becoming more popular.",
     (5, 7)),  # Partially supported by context

    (["Electric vehicles (EVs) are powered by electricity stored in batteries.", "EVs help reduce carbon emissions."], 
     "Electric vehicles help reduce carbon emissions. They are powered by electricity stored in batteries.",
     (7, 10)),  # Full alignment

    (["Many cancer treatments, such as chemotherapy, are still undergoing research to improve their effectiveness and reduce side effects.",
      "Current treatments have limitations and often cause significant side effects."], 
     "Recent advancements have completely eliminated the side effects of cancer treatments.",
     (0, 5))  # Claims to resolve an ongoing issue
])
def test_hallucination_scoring(evaluator, context, output, expected_score_range):
    hallucination = evaluator("hallucination", context=context, output=output)
    result = hallucination.score()
    assert math.ceil(result['score']) >= expected_score_range[0] and math.floor(result['score']) <= expected_score_range[1], f"Score {result['score']} not in range {expected_score_range}"

    max_score = hallucination.score(max)['score']
    mean_score = hallucination.score(statistics.mean)['score']
    min_score = hallucination.score(min)['score']
    
    assert max_score >= mean_score, f"Max score {max_score} is not greater than mean score {mean_score}"
    assert min_score <= mean_score, f"Min score {min_score} is not less than mean score {mean_score}"

def test_hallucination_empty_context(evaluator):
    context = []
    output = "Electric vehicles help reduce carbon emissions."
    with pytest.raises(ValueError, match="'context' cannot be an empty list"):
        hallucination = evaluator("hallucination", context=context, output=output)
        hallucination.score()

def test_hallucination_empty_output(evaluator):
    context = ["Electric vehicles (EVs) are powered by electricity stored in batteries.", "EVs help reduce carbon emissions."]
    output = ""
    with pytest.raises(ValueError, match="'output' cannot be an empty string"):
        hallucination = evaluator("hallucination", context=context, output=output)
        hallucination.score()

def test_hallucination_invalid_context_type(evaluator):
    context = "Electric vehicles (EVs) are powered by electricity stored in batteries."  # Non-list context
    output = "Electric vehicles help reduce carbon emissions."
    with pytest.raises(TypeError, match="'context' must be a list of strings"):
        hallucination = evaluator("hallucination", context=context, output=output)
        hallucination.score()

def test_hallucination_invalid_output_type(evaluator):
    context = ["Electric vehicles (EVs) are powered by electricity stored in batteries.", "EVs help reduce carbon emissions."]
    output = 12345  # Non-string output
    with pytest.raises(TypeError, match="'output' must be a string"):
        hallucination = evaluator("hallucination", context=context, output=output)
        hallucination.score()

def test_hallucination_non_string_in_context(evaluator):
    context = ["Electric vehicles (EVs) are powered by electricity stored in batteries.", 12345]  # Non-string item in context list
    output = "Electric vehicles help reduce carbon emissions."
    with pytest.raises(TypeError, match="All items in 'context' must be strings"):
        hallucination = evaluator("hallucination", context=context, output=output)
        hallucination.score()

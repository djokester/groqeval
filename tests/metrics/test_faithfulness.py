import pytest
import math

@pytest.mark.parametrize("context, output, expected_score_range", [
    (["Solar energy is becoming increasingly cost-effective, making it a competitive alternative to fossil fuels."], 
     "Solar panels are made from cheese. They can power a city for a year without any sunlight.",
     (0, 3)),  # Completely false or unsupported

    (["Solar energy is becoming increasingly cost-effective, making it a competitive alternative to fossil fuels."], 
     "Wind energy is also an effective renewable resource.",
     (3, 5)),  # Factually true but unsupported by context

    (["Solar energy is becoming increasingly cost-effective, making it a competitive alternative to fossil fuels."], 
     "Solar energy reduces dependency on fossil fuels. The raw materials for solar panels have become cheaper over the years.",
     (5, 7)),  # Partially supported by context

    (["Solar energy is becoming increasingly cost-effective, making it a competitive alternative to fossil fuels.",
      "Recent advancements have significantly increased the efficiency of solar panels."], 
     "Solar panels have become more efficient due to recent advancements. Solar energy reduces dependency on non-renewable energy sources.",
     (7, 10))  # Completely true and supported
])
def test_faithfulness_scoring(evaluator, context, output, expected_score_range):
    faithfulness = evaluator("faithfulness", context=context, output=output)
    result = faithfulness.score()
    assert math.ceil(result['score']) >= expected_score_range[0] and math.floor(result['score']) <= expected_score_range[1], f"Score {result['score']} not in range {expected_score_range}"

def test_faithfulness_empty_context(evaluator):
    context = []
    output = "Solar panels have become more efficient due to recent advancements."
    with pytest.raises(ValueError, match="'context' cannot be an empty list"):
        faithfulness = evaluator("faithfulness", context=context, output=output)
        faithfulness.score()

def test_faithfulness_empty_output(evaluator):
    context = ["Solar energy is becoming increasingly cost-effective, making it a competitive alternative to fossil fuels."]
    output = ""
    with pytest.raises(ValueError, match="'output' cannot be an empty string"):
        faithfulness = evaluator("faithfulness", context=context, output=output)
        faithfulness.score()

def test_faithfulness_invalid_context_type(evaluator):
    context = "Solar energy is becoming increasingly cost-effective, making it a competitive alternative to fossil fuels."  # Non-list context
    output = "Solar panels have become more efficient due to recent advancements."
    with pytest.raises(TypeError, match="'context' must be a list of strings"):
        faithfulness = evaluator("faithfulness", context=context, output=output)
        faithfulness.score()

def test_faithfulness_invalid_output_type(evaluator):
    context = ["Solar energy is becoming increasingly cost-effective, making it a competitive alternative to fossil fuels."]
    output = 12345  # Non-string output
    with pytest.raises(TypeError, match="'output' must be a string"):
        faithfulness = evaluator("faithfulness", context=context, output=output)
        faithfulness.score()

def test_faithfulness_non_string_in_context(evaluator):
    context = ["Solar energy is becoming increasingly cost-effective.", 12345]  # Non-string item in context list
    output = "Solar panels have become more efficient due to recent advancements."
    with pytest.raises(TypeError, match="All items in 'context' must be strings"):
        faithfulness = evaluator("faithfulness", context=context, output=output)
        faithfulness.score()

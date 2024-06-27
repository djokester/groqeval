# groqeval
Use groq for evaluations

### Evaluator 
```python
from groqeval.evaluate import GroqEval
evaluator = GroqEval(api_key=API_KEY)
```
### Hallucinations
The hallucination metric evaluates the alignment between an output and a given context. Specifically, it assesses the extent to which the output introduces information that is not supported by the context, even if such information may be factually accurate elsewhere. This is crucial for ensuring that the generated outputs remain grounded in the provided context and do not mislead or introduce inaccuracies.

#### Calculation
The hallucination metric evaluates the alignment between an output and its context, scoring each context statement on a scale from 1 (complete contradiction) to 10 (full alignment). The metric involves decomposing the context into individual statements, scoring the output's alignment with each statement, and then calculating the average score using the formula $$\[ \frac{1}{n} \sum_{i=1}^{n} \text{score}_i \]$$

where n is the number of context statements. This average score provides a quantitative measure of the output's adherence to the provided context.


#### Usage
```python
context = [
    "The use of electric vehicles (EVs) has been steadily increasing as part of global efforts to reduce carbon emissions.",
    "Electric vehicles help reduce dependency on fossil fuels but face challenges such as limited battery range and long charging times.",
    "Many countries are investing in infrastructure to support EV adoption, which includes building more charging stations."
]

output = "Electric vehicles not only reduce carbon emissions but also significantly lower local air pollutants, which improves urban air quality. The latest advancements in battery technology have begun to address the range anxiety associated with EVs by extending their driving range significantly. Moreover, some regions are seeing a surge in ultra-fast charging stations that can charge an EV battery to 80% in just 20 minutes."

hallucination = evaluator("hallucination", context = context, output = output)

output = hallucination.score()
```

#### Output
```json
{
  "score": 7.333333333333333,
  "score_breakdown": {
    "scores": [
      {
        "string": "The use of electric vehicles (EVs) has been steadily increasing as part of global efforts to reduce carbon emissions.",
        "rationale": "The output aligns with the context as it also mentions the reduction of carbon emissions. The context sets the stage for the benefits of EVs, and the output builds upon that by highlighting their positive impact on urban air quality.",
        "score": 9
      },
      {
        "string": "Electric vehicles help reduce dependency on fossil fuels but face challenges such as limited battery range and long charging times.",
        "rationale": "The output partially aligns with the context as it addresses the challenges of EVs, but the context presents these challenges as ongoing issues, whereas the output implies that they are being addressed. The output's assertion that battery technology has begun to address range anxiety contradicts the context's presentation of limited battery range as a challenge.",
        "score": 4
      },
      {
        "string": "Many countries are investing in infrastructure to support EV adoption, which includes building more charging stations.",
        "rationale": "The output aligns with the context as it mentions the surge in ultra-fast charging stations, which is a form of investment in infrastructure to support EV adoption. The context sets the stage for the growth of EV infrastructure, and the output builds upon that by highlighting the benefits of this growth.",
        "score": 9
      }
    ]
  }
}
```
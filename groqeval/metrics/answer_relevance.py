# groqeval/metrics/answer_relevance.py
import json
from groqeval.models.output import Output, ScoredOutput
from groqeval.metrics.base_metric import BaseMetric

class AnswerRelevance(BaseMetric):
    def __init__(self, groq_client, output, prompt):
        super().__init__(groq_client)
        self.output = output
        self.prompt = prompt

    @property
    def output_decomposition_prompt(self):
        json_representation = json.dumps(Output.model_json_schema(), indent=2)
        return (
            "Please process the following output from a language model and decompose it into individual phrases or chunks. "
            "For each phrase or chunk, evaluate whether it can be considered a statement based on its form as a declarative construct "
            "that communicates information, opinions, or beliefs. A phrase should be marked as a statement (true) if it forms a clear, standalone declaration. "
            "Phrases that are overly vague, questions, or merely connective phrases without any declarative content should be marked as not statements (false). "
            "Return the results in a JSON format. The JSON should have an array of objects, each representing a phrase with two properties: "
            "a 'string' that contains the phrase text, and a 'flag' that is a boolean indicating whether the text is considered a statement (true) or not (false).\n"
            f"Use the following JSON schema for your output: {json_representation}"
        )

    @property
    def relevance_prompt(self):
        return (
            f"Given the prompt: '{self.prompt}', evaluate the relevance of the following statements. "
            "Score each coherent sentence on a scale from 1 to 10, where 1 means the sentence is completely irrelevant to the context and prompt, "
            "and 10 means it is highly relevant. Ensure that the full range of scores is utilized, not just the two extremes, to prevent the scoring from being binary in nature. Include a rationale for each score to explain why the sentence received that rating. "            
            f"Use the following JSON schema for your output: {json.dumps(ScoredOutput.model_json_schema(), indent=2)}"
        )

    def output_decomposition(self):
        messages = [
            {"role": "system", "content": self.output_decomposition_prompt},
            {"role": "user", "content": self.output}
        ]
        print(messages)
        response = self.groq_chat_completion(
            messages=messages,
            model="llama3-70b-8192",
            temperature=0,
            response_format={"type": "json_object"}
        )
        return Output.model_validate_json(response.choices[0].message.content)
    
    def score_relevance(self):
        decomposed_output = self.output_decomposition()
        # Filter out incoherent sentences
        coherent_sentences = [s for s in decomposed_output.sentences if s.flag]
        messages = [
            {"role": "system", "content": self.relevance_prompt},
            {"role": "user", "content": json.dumps({"sentences": [s.string for s in coherent_sentences]}, indent=2)}
        ]
        response = self.groq_chat_completion(
            messages=messages,
            model="llama3-70b-8192",
            temperature=0,
            response_format={"type": "json_object"}
        )
        return ScoredOutput.model_validate_json(response.choices[0].message.content)
    
    def score(self):
        scored_output = self.score_relevance()
        if scored_output.scores:
            average_score = sum([output.score for output in scored_output.scores]) / len(scored_output.scores)
            return {
                'score': average_score,
                'score_breakdown': scored_output
            }
        else:
            return {
                'score': 0,  # Default to 0 if there are no sentences to score
                'score_breakdown': scored_output
            }
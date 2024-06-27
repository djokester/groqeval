# groqeval/metrics/faithfulness.py
import json
from groqeval.models.output import Output, ScoredOutput
from groqeval.metrics.base_metric import BaseMetric

class Faithfulness(BaseMetric):
    def __init__(self, groq_client, context, output):
        super().__init__(groq_client)
        self.context = context
        self.output = output


    @property
    def output_decomposition_prompt(self):
        json_representation = json.dumps(Output.model_json_schema(), indent=2)
        return (
            "Please process the following output from a language model and decompose it into individual phrases or chunks. "
            "For each phrase or chunk, evaluate whether it can be considered a claim based on its form as a declarative construct "
            "that communicates information, opinions, or beliefs. A phrase or chunk should be marked as a claim (true) if it forms a clear, standalone declaration, "
            "conveying a specific assertion or point. Phrases or chunks that are overly vague, purely interrogative, or function as connective phrases without "
            "substantial declarative content should be marked as not claims (false). "
            "Return the results in a JSON format. The JSON should have an array of objects, each representing a phrase or chunk with two properties: "
            "a 'string' that contains the text of the claim, and a 'flag' that is a boolean indicating whether the text is considered a claim (true) or not (false).\n"
            f"Use the following JSON schema for your output: {json_representation}"
        )

    @property
    def format_retrieved_context(self):
        formatted_strings = "\n".join(f"- {s}" for s in self.context)
        return f"The retrieved context includes the following items:\n{formatted_strings}"
    
    @property
    def faithfulness_prompt(self):
        return (
            f"Given the context: '{self.format_retrieved_context}', evaluate the truthfulness of the following claims. "
            "Score each claim on a scale from 1 to 10, where 1 means the claim is completely false or unsupported by the context, "
            "and 10 means the claim is entirely true and supported by the context. Ensure that the full range of scores is utilized, not just the two extremes, "
            "to prevent the scoring from being binary in nature. Make sure that any claim supported in the context should score over 5."
            "Claims that are true but not supporrted by the context should score less than 5 but near to it"
            "Include a rationale for each score to explain why the claim received that rating based on the facts presented in the context. "
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
    

    def score_faithfulness(self):
        decomposed_output = self.output_decomposition()
        # Filter out incoherent sentences
        coherent_sentences = [s for s in decomposed_output.sentences if s.flag]
        messages = [
            {"role": "system", "content": self.faithfulness_prompt},
            {"role": "user", "content": json.dumps({"sentences": [s.string for s in coherent_sentences]}, indent=2)}
        ]
        print(messages)
        response = self.groq_chat_completion(
            messages=messages,
            model="llama3-70b-8192",
            temperature=0,
            response_format={"type": "json_object"}
        )
        return ScoredOutput.model_validate_json(response.choices[0].message.content), json.loads(response.choices[0].message.content)
    
    def score(self):
        scored_output, output_dictionary = self.score_faithfulness()
        if scored_output.scores:
            average_score = sum([output.score for output in scored_output.scores]) / len(scored_output.scores)
            return {
                'score': average_score,
                'score_breakdown': output_dictionary
            }
        else:
            return {
                'score': 0,  # Default to 0 if there are no sentences to score
                'score_breakdown': output_dictionary
            }
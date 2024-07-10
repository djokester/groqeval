# groqeval/metrics/answer_relevance.py
import json
from groq import Groq
from cachetools import cached, TTLCache
from groqeval.models.output import Output, ScoredOutput
from groqeval.metrics.base_metric import BaseMetric

class AnswerRelevance(BaseMetric):
    """
    The Answer Relevance metric evaluates how accurately and closely the responses of 
    a language model align with the specific query or prompt provided. This metric ensures 
    that each part of the output, recognized as coherent statements, is scored for its 
    relevance to the original question, helping to gauge the utility and appropriateness 
    of the model's responses.
    """
    def __init__(self, groq_client: Groq, output: str, prompt: str, **kwargs):
        super().__init__(groq_client, kwargs.get('verbose'))
        self.output = output
        self.prompt = prompt
        self.check_data_types(prompt=prompt, output=output)

    @property
    def output_decomposition_prompt(self):
        """
        Prompt for decomposing the output into sentences.
        """
        json_representation = json.dumps(Output.model_json_schema(), indent=2)
        return (
            "Please process the following output from a language model and "
            "decompose it into individual phrases or chunks. For each phrase or "
            "chunk, evaluate whether it can be considered a statement based on its "
            "form as a declarative construct that communicates information, opinions, "
            "or beliefs. A phrase should be marked as a statement (true) if it forms "
            "a clear, standalone declaration. Phrases that are overly vague, questions, "
            "or merely connective phrases without any declarative content should be marked "
            "as not statements (false). Return the results in a JSON format. The JSON should "
            "have an array of objects, each representing a phrase with two properties: a "
            "'string' that contains the phrase text, and a 'flag' that is a boolean indicating "
            "whether the text is considered a statement (true) or not (false).\nUse the following "
            f"JSON schema for your output: {json_representation}"
        )


    @property
    def relevance_prompt(self):
        """
        Prompt for scoring the relevance of each statement in the output with respect to the prompt.
        """
        return (
            f"Given the prompt: '{self.prompt}', evaluate the relevance of the following statements. "
            "Score each coherent statement on a scale from 1 to 10, where 1 means the statement is completely irrelevant to the prompt, "
            "and 10 means it is highly relevant. Ensure that the full range of scores is utilized, not just the two extremes, "
            "to prevent the scoring from being binary in nature. Make sure that anything relevant to the prompt should score over 5. "
            "Include a rationale for each score to explain why the statement received that rating. "
            f"Use the following JSON schema for your output: {json.dumps(ScoredOutput.model_json_schema(), indent=2)}"
        )


    def output_decomposition(self):
        """
        Decomposes the output into individual phrases or chunks. 
        Each phrase or chunk is evaluated to determine if it can be considered a statement.
        A "statement" is defined as a clear, standalone declarative construct that 
        communicates information, opinions, or beliefs effectively.
        """
        messages = [
            {"role": "system", "content": self.output_decomposition_prompt},
            {"role": "user", "content": self.output}
        ]
        response = self.groq_chat_completion(
            messages=messages,
            model="llama3-70b-8192",
            temperature=0,
            response_format={"type": "json_object"}
        )
        self.logger.info("Decomposition of the Output into Statements: %s", response.choices[0].message.content)
        return Output.model_validate_json(response.choices[0].message.content)

    @cached(cache=TTLCache(maxsize=100, ttl=300))
    def score_relevance(self):
        """
        Each identified statement is then scored on a scale from 1 (completely irrelevant) 
        to 10 (highly relevant) in relation to how well it addresses the prompt.
        """
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
        self.logger.info("Breakdown of the Answer Relevance Score: %s", response.choices[0].message.content)
        return ScoredOutput.model_validate_json(response.choices[0].message.content), json.loads(response.choices[0].message.content)

    @property
    def scoring_function(self):
        return self.score_relevance

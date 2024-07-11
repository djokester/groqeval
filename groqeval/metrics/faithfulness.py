# groqeval/metrics/faithfulness.py
import json
from typing import List
from groq import Groq
from cachetools import cached, TTLCache
from groqeval.models.output import Output, ScoredOutput
from groqeval.metrics.base_metric import BaseMetric

class Faithfulness(BaseMetric):
    """
    Faithfulness measures how well the outputs of a language model adhere to the facts 
    and data presented in the context it was provided. This metric ensures that the generated 
    content is not only relevant but also accurate and truthful with respect to the given context, 
    critical for maintaining the integrity and reliability of the model's responses.
    """
    def __init__(self, groq_client: Groq, context: List[str], output: str, **kwargs):
        super().__init__(groq_client, kwargs.get('verbose'))
        self.context = context
        self.output = output        
        self.check_data_types(context=context, output=output)

    @property
    def output_decomposition_prompt(self):
        """
        Prompt to decompose the language model output into phrases and evaluate for claims.
        """
        json_representation = json.dumps(Output.model_json_schema(), indent=2)
        return (
            "Please process the following output from a language model and decompose it into "
            "individual phrases or chunks. For each phrase or chunk, evaluate whether it can "
            "be considered a claim based on its form as a declarative construct that communicates "
            "information, opinions, or beliefs. A phrase or chunk should be marked as a claim "
            "(true) if it forms a clear, standalone declaration, conveying a specific assertion "
            "or point. Phrases or chunks that are overly vague, purely interrogative, or function "
            "as connective phrases without substantial declarative content should be marked as not "
            "claims (false). Return the results in a JSON format. The JSON should have an array of "
            "objects, each representing a phrase or chunk with two properties: a 'string' that "
            "contains the text of the claim, and a 'flag' that is a boolean indicating whether "
            "the text is considered a claim (true) or not (false). Use the following JSON schema "
            f"for your output: {json_representation}"
        )


    @property
    def format_retrieved_context(self):
        """
        Formats the retrieved context which is a List[str] into a string.
        """
        formatted_strings = "\n".join(f"- {s}" for s in self.context)
        return f"The retrieved context includes the following items:\n{formatted_strings}"

    @property
    def faithfulness_prompt(self):
        """
        Prompt to score each claim made in the output for alignment with the retrieved context.
        """
        json_schema = json.dumps(ScoredOutput.model_json_schema(), indent=2)
        return (
            f"Given the context: '{self.format_retrieved_context}', evaluate the truthfulness "
            "of the following claims. Score each claim on a scale from 1 to 10, where 1 means "
            "the claim is completely false or unsupported by the context, and 10 means the "
            "claim is entirely true and supported by the context. Ensure that the full range "
            "of scores is utilized, not just the two extremes, to prevent the scoring from "
            "being binary in nature. Any claim supported in the context should score over 5. "
            "Claims that are true but not supported by the context should score less than 5 "
            "but near to it. Include a rationale for each score to explain why the claim "
            "received that rating based on the facts presented in the context. Use the "
            f"following JSON schema for your output: {json_schema}"
        )


    def output_decomposition(self):
        """
        Faithfulness is calculated by first decomposing the output into individual 
        phrases or chunks. Each phrase or chunks is then evaluated to determine 
        if it can be considered a claim in the sense that it is a declarative 
        construct that communicates information, opinions, or beliefs. 
        A phrase is marked as a claim if it forms a clear, standalone declaration
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
        self.logger.info("Decomposition of the Output into Claims: \n%s", response.choices[0].message.content)
        return Output.model_validate_json(response.choices[0].message.content)

    @cached(cache=TTLCache(maxsize=100, ttl=300))
    def score_faithfulness(self):
        """
        Claims are then scored on a scale from 1 to 10. 
        A score from 1 to 4 is assigned to claims that 
        are either completely false or unsupported by the context, 
        with claims that are factually true but unsupported by the 
        specific context receiving scores close to 4. 
        A score of 5 or above is reserved for claims that are both 
        factually true and corroborated by the context. 
        """
        decomposed_output = self.output_decomposition()
        # Filter out incoherent sentences
        coherent_sentences = [s for s in decomposed_output.sentences if s.flag]
        messages = [
            {"role": "system", "content": self.faithfulness_prompt},
            {"role": "user", "content": json.dumps({"sentences": [s.string for s in coherent_sentences]}, indent=2)}
        ]
        response = self.groq_chat_completion(
            messages=messages,
            model="llama3-70b-8192",
            temperature=0,
            response_format={"type": "json_object"}
        )
        self.logger.info("Breakdown of the Faithfulness Score: \n%s", response.choices[0].message.content)
        return ScoredOutput.model_validate_json(response.choices[0].message.content), json.loads(response.choices[0].message.content)
    
    @property
    def scoring_function(self):
        return self.score_faithfulness

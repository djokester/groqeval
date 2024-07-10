# groqeval/metrics/context_relevance.py
import json
from typing import List
from groq import Groq
from cachetools import cached, TTLCache
from groqeval.models.context import Context, ScoredContext
from groqeval.metrics.base_metric import BaseMetric

class ContextRelevance(BaseMetric):
    """
    Context Relevance evaluates the effectiveness of the retriever in a 
    RAG pipeline by measuring the relevance of the retrieved context 
    to the input query. This metric ensures that the context provided 
    to the generator is pertinent and likely to enhance the quality and 
    accuracy of the generated responses.
    """
    def __init__(self, groq_client: Groq, context: List[str], prompt: str, **kwargs):
        super().__init__(groq_client, kwargs.get('verbose'))
        self.context = context
        self.prompt = prompt
        self.check_data_types(prompt=prompt, context=context)

    @property
    def context_decomposition_prompt(self):
        """
        Prompt to decompose the context retrieved in response to a given prompt 
        into phrases and evaluate for statements.
        """
        json_representation = json.dumps(Context.model_json_schema(), indent=2)
        return (
            "Please process the following context retrieved in response to a given prompt "
            "and decompose it into individual phrases or chunks. For each phrase or chunk, "
            "evaluate whether it can be considered a statement based on its form as a "
            "declarative construct that communicates information, opinions, or beliefs. A "
            "phrase should be marked as a statement (true) if it forms a clear, standalone "
            "declaration. Phrases that are overly vague, questions, or merely connective "
            "phrases without any declarative content should be marked as not statements "
            "(false). Return the results in a JSON format. The JSON should have an array of "
            "objects, each representing a phrase with two properties: a 'string' that contains "
            "the phrase text, and a 'flag' that is a boolean indicating whether the text is "
            f"considered a statement (true) or not (false). Use the following JSON schema for "
            f"your output: {json_representation}"
        )

    @property
    def relevance_prompt(self):
        """
        Prompt to score how well each statement in the context retrieved
        in response to a given query relates to the query.
        """
        json_schema = json.dumps(ScoredContext.model_json_schema(), indent=2)
        return (
            f"Given the prompt: '{self.prompt}', evaluate the relevance of the following "
            "statements. Score each coherent sentence on a scale from 1 to 10, where 1 means "
            "the sentence is completely irrelevant to the prompt, and 10 means it is highly "
            "relevant. Ensure that the full range of scores is utilized, not just the two "
            "extremes, to prevent the scoring from being binary in nature. Make sure that "
            "anything relevant to the prompt should score over 5. Include a rationale for "
            "each score to explain why the sentence received that rating. Use the following "
            f"JSON schema for your output: {json_schema}"
        )

    @property
    def format_retrieved_context(self):
        """
        Formats the retrieved context which is a List[str] into a string.
        """
        formatted_strings = "\n".join(f"- {s}" for s in self.context)
        return f"The retrieved context includes the following items:\n{formatted_strings}"


    def context_decomposition(self):
        """
        Decomposes the context into individual phrases or chunks. 
        Each phrase or chunk is evaluated to determine if it can be considered a statement.
        A "statement" is defined as a clear, standalone declarative construct that 
        communicates information, opinions, or beliefs effectively.
        """
        messages = [
            {"role": "system", "content": self.context_decomposition_prompt},
            {"role": "user", "content": self.format_retrieved_context}
        ]
        response = self.groq_chat_completion(
            messages=messages,
            model="llama3-70b-8192",
            temperature=0,
            response_format={"type": "json_object"}
        )
        self.logger.info("Decomposition of the Context into Statements: %s", response.choices[0].message.content)
        return Context.model_validate_json(response.choices[0].message.content)

    @cached(cache=TTLCache(maxsize=100, ttl=300))
    def score_relevance(self):
        """
        Each statement of context is evaluated to determine if it can be 
        considered a relevant response to the query. A "relevant response" 
        is defined as a clear, contextual piece of information that directly 
        addresses the query's topic or related aspects. Each identified piece 
        of context is then scored on a scale from 1 (completely irrelevant) 
        to 10 (highly relevant) based on how well it relates to the initial query.
        """
        decomposed_context = self.context_decomposition()
        # Filter out incoherent sentences
        coherent_sentences = [s for s in decomposed_context.sentences if s.flag]
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
        self.logger.info("Breakdown of the Context Relevance Score: %s", response.choices[0].message.content)
        return ScoredContext.model_validate_json(response.choices[0].message.content), json.loads(response.choices[0].message.content)

    @property
    def scoring_function(self):
        return self.score_relevance

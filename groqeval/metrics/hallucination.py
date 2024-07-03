# groqeval/metrics/hallucination.py
import json
from typing import List
from groq import Groq
from groqeval.models.context import Context, ScoredContext
from groqeval.metrics.base_metric import BaseMetric

class Hallucination(BaseMetric):
    """
    The hallucination metric evaluates the alignment between an output and a given context. 
    Specifically, it assesses the extent to which the output introduces information that is 
    not supported by the context, even if such information may be factually accurate elsewhere. 
    This is crucial for ensuring that the generated outputs remain grounded in the provided 
    context and do not mislead or introduce inaccuracies.
    """
    def __init__(self, groq_client: Groq, context: List[str], output: str):
        super().__init__(groq_client)
        self.context = context
        self.output = output
        self.check_data_types(context=context, output=output)


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
            "(false). Return the results in a JSON format. The JSON should have an array "
            "of objects, each representing a phrase with two properties: a 'string' that "
            "contains the phrase text, and a 'flag' that is a boolean indicating whether "
            "the text is considered a statement (true) or not (false). Use the following "
            f"JSON schema for your output: {json_representation}"
        )


    @property
    def hallucination_prompt(self):
        """
        Prompt to evaluate each context for alignment or contradiction 
        with the given output.
        """
        json_schema = json.dumps(ScoredContext.model_json_schema(), indent=2)
        return (
            f"Given the output: '{self.output}', critically evaluate each context "
            "to determine if there are contradictions or alignments with the output. "
            "Assign a score from 1 to 10, where 1 indicates a complete contradiction "
            "(the output directly opposes the context) and 10 indicates full alignment "
            "(the output and context agree completely). Intermediate scores should reflect "
            "the extent to which the output diverges from or conforms to the context's facts "
            "and implications. Score below 5 if the output introduces elements not supported "
            "by the context, even if these are accurate elsewhere. The context is the definitive "
            "source for evaluations. If the output claims to resolve an issue like variability "
            "in renewable energy—which the context still presents as ongoing—this should be seen "
            "as a contradiction, not a partial alignment. Each score must include a rationale that "
            "explicitly states why the output either aligns with or contradicts the context, "
            "highlighting specific discrepancies or agreements. Pay particular attention to "
            "assertions about solutions or improvements that contradict unresolved issues "
            "presented in the context. Scores around 5 should be reserved for outputs that "
            "neither clearly align nor contradict but may introduce unrelated or ambiguous "
            "elements. Ensure evaluations are formatted according to the JSON schema provided: "
            f"{json_schema}"
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
        Decomposes the context into individual statements
        """
        messages = [
            {"role": "system", "content": self.context_decomposition_prompt},
            {"role": "user", "content": self.format_retrieved_context}
        ]
        print(messages)
        response = self.groq_chat_completion(
            messages=messages,
            model="llama3-70b-8192",
            temperature=0,
            response_format={"type": "json_object"}
        )
        return Context.model_validate_json(response.choices[0].message.content)
    
    def score_hallucination(self):
        """
        The hallucination metric evaluates the alignment between an output and its context, 
        scoring each context statement on a scale from 1 (complete contradiction) to 10 (full alignment). 
        """
        decomposed_context = self.context_decomposition()
        # Filter out incoherent sentences
        coherent_sentences = [s for s in decomposed_context.sentences if s.flag]
        messages = [
            {"role": "system", "content": self.hallucination_prompt},
            {"role": "user", "content": json.dumps({"sentences": [s.string for s in coherent_sentences]}, indent=2)}
        ]
        response = self.groq_chat_completion(
            messages=messages,
            model="llama3-70b-8192",
            temperature=0,
            response_format={"type": "json_object"}
        )
        return ScoredContext.model_validate_json(response.choices[0].message.content), json.loads(response.choices[0].message.content)
    
    def score(self):
        scored_context, output_dictionary = self.score_hallucination()
        if scored_context.scores:
            average_score = sum([context.score for context in scored_context.scores]) / len(scored_context.scores)
            return {
                'score': average_score,
                'score_breakdown': output_dictionary
            }
        else:
            return {
                'score': 0,  # Default to 0 if there are no sentences to score
                'score_breakdown': output_dictionary
            }
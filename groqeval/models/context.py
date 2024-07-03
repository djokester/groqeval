# groqeval/models/context.py
from typing import List
from pydantic import BaseModel

class Sentence(BaseModel):
    """
    Basic Construct. Deconstructed Context
    """
    string: str
    flag: bool

class Score(BaseModel):
    """
    Score of an Individual Deconstruct
    """
    string: str
    rationale: str
    score: int

class Context(BaseModel):
    """
    THe Output Class as List of Deconstructions
    """
    sentences: List[Sentence]

class ScoredContext(BaseModel):
    """
    A List of Scored Deconstruction
    """
    scores: List[Score]


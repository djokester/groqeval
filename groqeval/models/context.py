# groqeval/models/context.py
from typing import List, Optional
from pydantic import BaseModel

class Sentence(BaseModel):
    string: str
    flag: bool

class Score(BaseModel):
    string: str
    rationale: str
    score: int

class Context(BaseModel):
    sentences: List[Sentence]

class ScoredContext(BaseModel):
    scores: List[Score]


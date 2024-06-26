# groqeval/models/output_models.py
from typing import List, Optional
from pydantic import BaseModel

class Sentence(BaseModel):
    string: str
    flag: bool

class Score(BaseModel):
    string: str
    rationale: str
    score: int

class Output(BaseModel):
    sentences: List[Sentence]

class ScoredOutput(BaseModel):
    scores: List[Score]


from typing import List, Optional, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.spacy_embed import SpaCyEmbedder
app = FastAPI()
# Sample corpus for both model

corpus = [
"The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
"this is another example sentence",
"we are generating text based on bigram probabilities",
"bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)
class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class TextGenerationRequest(BaseModel):
    start_word: Optional[str] = Field(default="<s>")
    length: int = Field(ge=1, le=50)

spacy_embedder = SpaCyEmbedder("en_core_web_md")  # you installed this

class SpaCyEmbedRequest(BaseModel):
    word: str = Field(..., description="Word to embed with spaCy vectors")
    neighbors: bool = Field(False, description="Also return nearest neighbors?")
    k: int = Field(10, ge=1, le=100, description="Top-K nearest neighbors if enabled")

class Neighbor(BaseModel):
    token: str
    similarity: float

class SpaCyEmbedResponse(BaseModel):
    word: str
    vector: List[float]
    neighbors: Optional[List[Neighbor]] = None


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    text = bigram_model.generate_text(request.start_word or "<s>", request.length)
    return {"generated_text": text}

@app.post("/embed_spacy", response_model=SpaCyEmbedResponse)
def embed_spacy(req: SpaCyEmbedRequest):
    vec = spacy_embedder.vector(req.word)
    if vec.size == 0:
        # Clean 400 instead of 500 for OOV words (unlikely with the lg model’s large vocab)
        raise HTTPException(status_code=400, detail="Word has no vector (OOV for en_core_web_lg).")
    resp = SpaCyEmbedResponse(word=req.word, vector=vec.tolist())
    if req.neighbors:
        nbrs = spacy_embedder.nearest_neighbors(req.word, k=req.k)
        resp.neighbors = [Neighbor(**n) for n in nbrs]
    return resp
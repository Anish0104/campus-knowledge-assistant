from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from search import search

app = FastAPI(title="Rutgers Knowledge Assistant")


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Question cannot be blank.")

        return value


@app.post("/search")
def search_documents(request: SearchRequest):
    results = search(
        question=request.question,
        top_k=request.top_k,
    )

    return {
        "question": request.question,
        "count": len(results),
        "results": results,
    }
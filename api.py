import httpx

from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator

from generate import answer_question



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


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Question cannot be blank.")

        return value


@app.post("/ask")
def ask_documents(request: AskRequest) -> dict:
    try:
        return answer_question(request.question)

    except httpx.TimeoutException as error:
        raise HTTPException(
            status_code=504,
            detail="Answer generation timed out. Please try again.",
        ) from error

    except httpx.RequestError as error:
        raise HTTPException(
            status_code=503,
            detail="Could not reach Ollama. Check that it is running.",
        ) from error

    except httpx.HTTPStatusError as error:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an error.",
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=502,
            detail="The generated response failed validation.",
        ) from error
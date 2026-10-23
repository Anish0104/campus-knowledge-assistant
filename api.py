import os

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from config import GROQ_MODEL, LLM_PROVIDER, OLLAMA_MODEL, PROJECT_ROOT
from generate import answer_question
from search import search


STATIC_DIR = PROJECT_ROOT / "static"

app = FastAPI(title="Margin: Campus Knowledge Assistant")

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


@app.get("/", include_in_schema=False)
def homepage():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    """Report that the API is up. Does not perform inference checks."""
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip().lower()
    model = (
        os.getenv("GROQ_MODEL", GROQ_MODEL)
        if provider == "groq"
        else os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
    )
    return {
        "status": "ok",
        "provider": provider,
        "model": model,
        "ollama_model": OLLAMA_MODEL,
    }


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    campus: str | None = Field(default=None, max_length=100)
    program: str | None = Field(default=None, max_length=100)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Question cannot be blank.")

        return value

    @field_validator("campus", "program")
    @classmethod
    def validate_scope(cls, value: str | None) -> str | None:
        if value is None:
            return None

        value = " ".join(value.split())

        if not value:
            raise ValueError("Scope must be non-blank or null.")

        return value


class SearchRequest(AskRequest):
    top_k: int = Field(default=3, ge=1, le=10)


@app.post("/search")
def search_documents(request: SearchRequest) -> dict:
    results = search(
        request.question,
        top_k=request.top_k,
        campus=request.campus,
        program=request.program,
    )

    return {
        "question": request.question,
        "count": len(results),
        "results": results,
    }


@app.post("/ask")
def ask_documents(request: AskRequest) -> dict:
    try:
        return answer_question(
            request.question,
            campus=request.campus,
            program=request.program,
        )

    except httpx.TimeoutException as error:
        raise HTTPException(
            status_code=504,
            detail="Answer generation timed out. Please try again.",
        ) from error

    except httpx.RequestError as error:
        raise HTTPException(
            status_code=503,
            detail="Could not reach the configured model provider. Check that it is available.",
        ) from error

    except httpx.HTTPStatusError as error:
        status = error.response.status_code
        if status in (401, 403):
            detail = "Model provider authentication failed. Check your API credentials."
        elif status == 429:
            detail = "Model provider rate limit exceeded. Please try again later."
        else:
            detail = f"Model provider returned an error ({status})."
        raise HTTPException(
            status_code=502,
            detail=detail,
        ) from error

    except ValueError as error:
        msg = str(error)
        if "GROQ_API_KEY is not configured" in msg or "LLM_PROVIDER" in msg:
            raise HTTPException(
                status_code=500,
                detail=f"Configuration error: {msg}",
            ) from error
        raise HTTPException(
            status_code=502,
            detail="The generated response failed validation.",
        ) from error

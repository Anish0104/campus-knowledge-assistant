import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from config import OLLAMA_MODEL, PROJECT_ROOT
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
    """Report that the API is up. Does not check Ollama."""
    return {"status": "ok", "ollama_model": OLLAMA_MODEL}


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

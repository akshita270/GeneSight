import uvicorn
from main import app  # noqa: F401 — imported for HF Spaces entry point

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860)

import sys

# Prevent HF Spaces ZeroGPU detector from erroring on a CPU-only space
class _MockSpaces:
    def GPU(self, func=None, **kwargs):
        return func if func is not None else (lambda f: f)
    def __getattr__(self, _):
        return lambda *a, **k: None

sys.modules.setdefault("spaces", _MockSpaces())

import gradio as gr
from main import app  # existing FastAPI app

with gr.Blocks(title="GeneSight API") as demo:
    gr.Markdown(
        "## GeneSight API\n"
        "Backend service — use the frontend at "
        "[gene-sight-seven.vercel.app](https://gene-sight-seven.vercel.app)"
    )

app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860)

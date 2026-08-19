from __future__ import annotations

# ── HfFolder compat ──────────────────────────────────────────────────────────
# Gradio 4.x oauth.py imports HfFolder removed in huggingface_hub>=0.21.0.
import huggingface_hub as _hf
if not hasattr(_hf, "HfFolder"):
    class _HfFolder:
        path_token = None
        @classmethod
        def get_token(cls): return None
        @classmethod
        def save_token(cls, token): pass
        @classmethod
        def delete_token(cls): pass
    _hf.HfFolder = _HfFolder

# ── ZeroGPU compat ───────────────────────────────────────────────────────────
# Free-tier Gradio Spaces run on ZeroGPU. Must import spaces and apply
# @spaces.GPU BEFORE importing Gradio, so the registry is populated when
# Gradio's startup check fires. No try/except — a failure here is a real error.
import spaces

@spaces.GPU
def _zerogpu_noop() -> str:
    """Satisfies ZeroGPU startup check; never actually called by the app."""
    return ""

import gradio as gr
import uvicorn
from main import app as fastapi_app

with gr.Blocks(title="GeneSight API") as demo:
    gr.Markdown(
        "## GeneSight API\n"
        "Backend service — use the frontend at "
        "[gene-sight-seven.vercel.app](https://gene-sight-seven.vercel.app)"
    )
    # Textbox.submit registers _zerogpu_noop as a Gradio event handler —
    # the ZeroGPU startup check scans Gradio's event list for GPU-wrapped fns.
    _txt = gr.Textbox(visible=False)
    _txt.submit(fn=_zerogpu_noop, inputs=[_txt], outputs=[_txt])

fastapi_app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

if __name__ == "__main__":
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860)

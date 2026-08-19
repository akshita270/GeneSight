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

import gradio as gr
import uvicorn
from main import app as fastapi_app

# ── ZeroGPU compat ───────────────────────────────────────────────────────────
# HF Spaces ZeroGPU startup check scans Gradio's event handler list for at
# least one @spaces.GPU-decorated callback. We add a hidden button+handler so
# the check passes. On CPU hardware spaces.GPU is a passthrough no-op.
try:
    import spaces as _spaces
    _has_spaces = True
except Exception:
    _has_spaces = False

with gr.Blocks(title="GeneSight API") as demo:
    gr.Markdown(
        "## GeneSight API\n"
        "Backend service — use the frontend at "
        "[gene-sight-seven.vercel.app](https://gene-sight-seven.vercel.app)"
    )
    # Invisible button/output: only here to satisfy ZeroGPU startup check
    _btn = gr.Button(visible=False)
    _out = gr.Textbox(visible=False)

    if _has_spaces:
        @_spaces.GPU
        def _gpu_noop():
            return ""
    else:
        def _gpu_noop():
            return ""

    _btn.click(fn=_gpu_noop, inputs=[], outputs=[_out])

fastapi_app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

if __name__ == "__main__":
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860)

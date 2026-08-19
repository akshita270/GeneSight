from __future__ import annotations
import sys
import types

# ── ZeroGPU startup check ─────────────────────────────────────────────────────
# HF Spaces ZeroGPU requires at least one @spaces.GPU-decorated function.
# Register a no-op dummy via the REAL decorator so the internal registry is
# non-empty when the startup check fires. Then replace spaces.GPU with a safe
# passthrough so Gradio internals can call it freely without GPU allocation.
try:
    import spaces as _sp
    @_sp.GPU
    def _noop_gpu_fn():
        pass
except Exception:
    # spaces not available — install a minimal stub
    if "spaces" not in sys.modules:
        _m = types.ModuleType("spaces")
        sys.modules["spaces"] = _m

sys.modules["spaces"].GPU = lambda fn=None, **kw: fn if fn else (lambda f: f)

# ── HfFolder compat ───────────────────────────────────────────────────────────
# Gradio 4.x oauth.py imports HfFolder removed in huggingface_hub>=0.21.0.
# Inject a no-op class before `import gradio` so the import chain succeeds.
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

with gr.Blocks(title="GeneSight API") as demo:
    gr.Markdown(
        "## GeneSight API\n"
        "Backend service — use the frontend at "
        "[gene-sight-seven.vercel.app](https://gene-sight-seven.vercel.app)"
    )

fastapi_app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

if __name__ == "__main__":
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860)

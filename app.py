from __future__ import annotations
import sys
import types

# Gradio 4.x + HF ZeroGPU: if no @spaces.GPU functions exist, startup errors.
# Force-replace spaces.GPU with a no-op whether or not spaces was pre-imported.
if "spaces" not in sys.modules:
    _mock_spaces = types.ModuleType("spaces")
    sys.modules["spaces"] = _mock_spaces
_spaces_mod = sys.modules["spaces"]
_spaces_mod.GPU = lambda fn=None, **kw: fn if fn else (lambda f: f)

# Gradio 4.x oauth.py does `from huggingface_hub import HfFolder` which was
# removed in huggingface_hub>=0.21.0. Add a no-op stub; we use Clerk auth.
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

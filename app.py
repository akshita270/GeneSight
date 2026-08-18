import gradio as gr
from main import app  # existing FastAPI app

with gr.Blocks(title="GeneSight API") as demo:
    gr.Markdown(
        "## GeneSight API\n"
        "Backend service — use the frontend at "
        "[gene-sight-seven.vercel.app](https://gene-sight-seven.vercel.app)"
    )

# Mount Gradio onto the FastAPI app; all existing /query /result /stream routes stay intact
app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860)

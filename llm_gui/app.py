import json
import os
import gradio as gr
import pandas as pd
from LLMModel import LLMModel

doc_llm = LLMModel()

try:
    demo.close()
except:
    pass

with gr.Blocks(title="Kenn") as demo:
    with gr.Column():
        gr.Tab("Splore Text Summarization Model GUI", interactive=False)
        with gr.Column("Document Upload"):
            gr.Interface(
                fn=doc_llm.process_file,
                inputs=["file"],
                outputs=gr.Textbox(label="Predictions", min_width=1000, show_copy_button=True, autofocus=True),
                title="Upload Topic Chat Data for Inference",
                description="Upload a file to start inference (supported file type: csv)",
                allow_flagging="auto"
            )
demo.launch()

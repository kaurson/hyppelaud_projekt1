import gradio as gr
from transformers import pipeline
from read_rss import get_news

generator = pipeline('text-generation', model='gpt2')


def generate(text, max_length, num_return_sequences):
    result = generator(text, max_length=int(max_length), num_return_sequences=int(num_return_sequences))
    return result[0]["generated_text"]


examples = get_news()
demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.inputs.Textbox(lines=5, label="Input Text"),
        gr.Number(value=50, label="Text length"),
        gr.Number(value=2, label="Number of segments")],
    outputs=gr.outputs.Textbox(label="Generated Text"),
    examples=examples
)

demo.launch()
# demo.launch(share=True) teeb avaliku lingi

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import gradio as gr

tokenizer = AutoTokenizer.from_pretrained('Davlan/distilbert-base-multilingual-cased-ner-hrl')
model = AutoModelForTokenClassification.from_pretrained('Davlan/distilbert-base-multilingual-cased-ner-hrl')


def ner(texto):
    ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer)
    output = ner_pipeline(texto)
    return {'text': texto, 'entities': output}


demo = gr.Interface(fn=ner, inputs='text', outputs=gr.Highlightedtext())
demo.launch()

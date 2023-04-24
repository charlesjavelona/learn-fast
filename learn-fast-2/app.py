#|export
from fastai.vision.all import *
import gradio as gr

def is_cat(x): return x[0].isupper()

categories = ('Dog', 'Cat')


labels = learn.dls.vocab
def classify_image(img):
    pred, idx, probs = learn.predict(image)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


img = gr.inputs.Image((192, 192))
label = gr.outputs.Label()
examples = ['dog.jpg', 'cat.jpg', 'dunno.jpg']

intf = gr.Interface(fn=classify_image, inputs=img, outputs=label, examples=examples)
intf.launch(inline=False)

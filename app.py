import gradio as gr
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

# Load your model
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact")
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)

def answer_question(question):
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"])
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Question"),
    outputs=gr.Textbox(label="Answer"),
    title="RAG QA System"
)

iface.launch()

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import gradio as gr
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
# Initialize components
def initialize_rag():
    # Load embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Load FAISS index
    vectorstore = FAISS.load_local(
        "faiss_dl_book_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    
    # Groq setup
    groq_api_key = os.environ.get("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")
    llm = ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192",
        groq_api_key=groq_api_key
    )
    
    # Prompt template
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create chain
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# Initialize RAG chain
rag_chain = initialize_rag()

def answer_question(question):
    try:
        return rag_chain.invoke(question)
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Bahishti Zewar Q&A System")
    with gr.Row():
        question = gr.Textbox(label="Enter your question")
        output = gr.Textbox(label="Answer")
    submit_btn = gr.Button("Ask")
    submit_btn.click(fn=answer_question, inputs=question, outputs=output)

if __name__ == "__main__":
    demo.launch()
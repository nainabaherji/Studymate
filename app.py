import streamlit as st
import fitz  # For reading PDFs
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# Load models
st_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_pipeline = pipeline("question-answering")

# PDF Text Extraction
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Split long text into parts
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Convert chunks to vectors
def embed_chunks(chunks):
    return st_model.encode(chunks)

# Find the best match for the question
def search_index(question, chunk_embeddings, chunks):
    question_embedding = st_model.encode([question])
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(chunk_embeddings)
    _, I = index.search(question_embedding, 1)
    return chunks[I[0][0]]

# Streamlit Interface
st.title(" StudyMate: AI PDF Q&A Assistant")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
question = st.text_input("Ask a question:")

if uploaded_file and question:
    with st.spinner("Thinking..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)
        best_chunk = search_index(question, embeddings, chunks)
        answer = qa_pipeline(question=question, context=best_chunk)['answer']
        st.success("Answer:")
        st.write(answer)

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

class QAEngine:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        self.index = None
        self.chunks = []

    def build_vector_store(self, chunks):
        self.chunks = chunks
        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def answer_question(self, question: str) -> str:
        question_embedding = self.model.encode([question])
        D, I = self.index.search(np.array(question_embedding), k=3)
        context = " ".join([self.chunks[i] for i in I[0]])
        result = self.qa_pipeline(question=question, context=context)
        return result["answer"]

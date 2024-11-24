from uuid import uuid4
from langchain_chroma import Chroma

from src.config.config import embeddings

class VectorStore:
    def __init__(self):
        
        self.embeddings = embeddings
        self.vectorstore = self.create_vector_store
    

    def get_context(self, query: str):
        context = self.vector_store.similarity_search(
            query,
            k=1
        )
        return context


    def create_vector_store(self, collection_name="karl", persist_directory="./chroma_langchain_db"):
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )
        return vector_store


    def add_documents_to_vector_store(self, texts):
        uuids = [str(uuid4()) for _ in range(len(texts))]
        self.vector_store.add_documents(documents=texts, ids=uuids)



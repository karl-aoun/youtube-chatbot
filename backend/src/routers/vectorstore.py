from fastapi import APIRouter, HTTPException

from src.vectorstore_manager.vectorstore_manager import VectorStore

vectorstore_manager = VectorStore()

vectorstore_router = APIRouter()

@vectorstore_router.post("get_context")
def get_vectorstore_context(query: str):
    vectorstore_manager.get_context(query=query)
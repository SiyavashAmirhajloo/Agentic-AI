# build_db.py
from src.embedding_db import VectorDB
from src.embedding_models import MiniEmbeddingModel

if __name__ == "__main__":
    print("Building RAG database from 'documents/' folder...")
    VectorDB(
        directory="documents",
        vector_file="database.npy",
        embedding_model=MiniEmbeddingModel()
    )
    print("Database built: database.npy + database_chunks.pkl")
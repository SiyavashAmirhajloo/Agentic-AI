# src/embedding_db.py
import os
import numpy as np
import requests
from bs4 import BeautifulSoup
import PyPDF2
from src.embedding_models import BaseEmbeddingModel, MiniEmbeddingModel
import pickle


class VectorDB:
    def __init__(
        self,
        directory: str = "documents",
        vector_file: str = "database.npy",
        max_words_per_chunk: int = 4000,
        embedding_model: BaseEmbeddingModel = MiniEmbeddingModel()  # Local model only
    ):
        """
        Initializes the vector database using local embeddings (no OpenAI/Groq API needed).
        """
        self.directory = directory
        self.vector_file = vector_file
        self.chunks_file = os.path.splitext(vector_file)[0] + "_chunks.pkl"
        self.max_words_per_chunk = max_words_per_chunk
        self.embedding_model = embedding_model

        # Only build if embeddings don't already exist
        if os.path.exists(self.vector_file) and os.path.exists(self.chunks_file):
            print(f"[VectorDB] Loading existing embeddings from {self.vector_file}")
            self.embeddings = np.load(self.vector_file)
            with open(self.chunks_file, 'rb') as f:
                self.chunks = pickle.load(f)
            print(f"[VectorDB] Loaded {len(self.chunks)} pre-computed chunks and embeddings.")
            return

        print(f"[VectorDB] Building new database from files in '{directory}/'")
        docs = self.read_text_files()
        if not docs:
            raise ValueError(f"No .txt files found in {directory}/")

        self.chunks = self.embedding_model.split_documents(docs)
        print(f"[VectorDB] Split into {len(self.chunks)} chunks")

        with open(self.chunks_file, 'wb') as f:
            pickle.dump(self.chunks, f)
        print(f"[VectorDB] Chunks saved to {self.chunks_file}")

        print("[VectorDB] Generating embeddings (this may take a minute on CPU)...")
        embeddings_data = self.embedding_model.get_embeddings_batch(self.chunks)
        self.embeddings = np.array(embeddings_data)  # Shape: (n_chunks, dim)

        print(f"[VectorDB] Generated {len(self.embeddings)} embeddings of dimension {self.embeddings.shape[1]}")
        self.store_embeddings()

    @staticmethod
    def scrape_website(url: str, output_file: str):
        """Download and save webpage or PDF content.
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '')

            if 'application/pdf' in content_type or url.endswith('.pdf'):
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                # Extract text
                try:
                    with open(output_file, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text = "\n".join(page.extract_text() or "" for page in reader.pages)
                    txt_path = os.path.splitext(output_file)[0] + ".txt"
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    print(f"[scrape] PDF text saved to {txt_path}")
                except Exception as e:
                    print(f"[scrape] Could not extract PDF text: {e}")

            elif 'text/html' in content_type:
                soup = BeautifulSoup(response.text, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"[scrape] HTML text saved to {output_file}")
            else:
                print(f"[scrape] Unsupported content type: {content_type}")

        except Exception as e:
            print(f"[scrape] Failed to fetch {url}: {e}")

    def read_text_files(self) -> list[str]:
        """Read all .txt files in the documents directory."""
        docs = []
        for fname in sorted(os.listdir(self.directory)):
            if fname.endswith(".txt"):
                path = os.path.join(self.directory, fname)
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    docs.append(content)
                    print(f"[VectorDB] Read {fname} ({len(content.split())} words)")
        return docs

    def store_embeddings(self):
        """Save embeddings to disk."""
        np.save(self.vector_file, self.embeddings)
        print(f"[VectorDB] Embeddings saved to {self.vector_file}")

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b) + 1e-8)
        return np.dot(a_norm, b_norm)

    @staticmethod
    def get_top_k(
        npy_file: str,
        embedding_model: BaseEmbeddingModel,
        query: str,
        k: int = 5,
        verbose: bool = False
    ) -> tuple[list[str], list[float]]:
        """
        Retrieve top-k most similar chunks for a query.
        """
        embeddings = np.load(npy_file)
        chunks_path = os.path.splitext(npy_file)[0] + "_chunks.pkl"
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)

        query_vec = np.array(embedding_model.get_embedding(query))

        similarities = [VectorDB.cosine_similarity(query_vec, emb) for emb in embeddings]
        top_indices = np.argsort(similarities)[-k:][::-1]

        top_chunks = [chunks[i] for i in top_indices]
        top_scores = [similarities[i] for i in top_indices]

        if verbose:
            print(f"\n[VectorDB] Top {k} results for query: \"{query}\"\n")
            for i, (chunk, score) in enumerate(zip(top_chunks, top_scores), 1):
                print(f"{i}. [Score: {score:.4f}]\n{chunk[:300]}{'...' if len(chunk) > 300 else ''}\n{'-'*60}")

        return top_chunks, top_scores


# —————— Run once to build the DB —————
if __name__ == "__main__":
    # Uses only local MiniLM model → no API key needed
    db = VectorDB(
        directory="documents",
        vector_file="database.npy",
        embedding_model=MiniEmbeddingModel()
    )

    # Test query
    results, scores = VectorDB.get_top_k("database.npy", MiniEmbeddingModel(), "reinforcement learning", k=3, verbose=True)
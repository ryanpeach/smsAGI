import faiss
from typing import List
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from langchain.docstore import InMemoryDocstore
from langchain.agents import Task
from langchain.embeddings import OpenAIEmbeddings

class VectorStoreMemory:
    def __init__(self, embedding_size: int):
        # Define your embedding model
        self.embeddings_model = OpenAIEmbeddings()
        # Initialize the vectorstore as empty
        self.in_memory_docstore = InMemoryDocstore({})
        self.index = faiss.IndexFlatL2(embedding_size)
        self.vectorstore = FAISS(self.embeddings_model.embed_query, self.index, self.in_memory_docstore, {})

    def get_top_tasks(self, query: str, k: int) -> List[Task]:
        """Get the top k tasks based on the query."""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        if not results:
            return []
        sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))
        return [item.metadata["task"] for item in sorted_results]
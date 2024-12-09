import numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, docs: list[str]):
        self.docs = docs

        self.minilm = SentenceTransformer('all-MiniLM-L6-v2')

        print('Encoding docs')
        self.embeddings = self.minilm.encode(docs)
        print('Docs encoded')

    def get_docs(self, query, n=3) -> list[str]:
        semantic_scores = self._get_minilm_scores(query)

        sorted_indices = np.argsort(semantic_scores)

        actual_n = -min(n, len(self.docs) - 1)

        return [self.docs[i] for i in sorted_indices[:actual_n]]
    
    def _get_minilm_scores(self, query):
        query_embedding = self.minilm.encode(query) 
        
        return self.minilm.similarity(query_embedding, self.embeddings)[0]

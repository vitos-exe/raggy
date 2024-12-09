import logging

import torch
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, docs: list[str]):
        self.docs = docs

        logging.info('Loading semantic retriever')
        self.minilm = SentenceTransformer('all-MiniLM-L6-v2')

        logging.info('Embedding documents')
        self.embeddings = self.minilm.encode(docs)

    def get_docs(self, query, n=3) -> list[str]:
        semantic_scores = self._get_minilm_scores(query)
        logging.info(f'Semantic scores: {semantic_scores}')

        sorted_indices = torch.argsort(semantic_scores, descending=True)

        return [self.docs[i] for i in sorted_indices[:n]]
    
    def _get_minilm_scores(self, query):
        query_embedding = self.minilm.encode(query)

        return self.minilm.similarity(query_embedding, self.embeddings)[0]

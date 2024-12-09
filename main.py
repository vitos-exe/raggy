import random
import logging

import gradio
from chonkie import WordChunker
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

from bot import RAGQuestionAnsweringBotHuggingFace
from constants import *
from retriever import Retriever

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logging.info('Starting application')

    logging.info('Loading RAG document dataset')
    ds = load_dataset(DOCUMENT_DATASET)
    docs = ds['train']['text']

    chunkier = WordChunker(
        tokenizer=Tokenizer.from_pretrained(TOKENIZER_NAME),
        chunk_size=256,
        chunk_overlap=64,
    )

    logging.info('Chunking documents')
    chunked_docs = []
    for doc in tqdm(docs):
        chunks = chunkier.chunk(doc)
        chunked_docs.extend([c.text for c in chunks])

    chunked_docs = random.sample(chunked_docs, len(chunked_docs) // 40)

    retriever = Retriever(chunked_docs)
    bot = RAGQuestionAnsweringBotHuggingFace(retriever)

    logging.info('Launching Gradio')
    demo = gradio.Interface(bot.answer_question, inputs='text', outputs='text')
    demo.launch()

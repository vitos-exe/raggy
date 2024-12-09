import random

import gradio
from chonkie import WordChunker
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

from bot import RAGQuestionAnsweringBot
from constants import *
from retriever import Retriever

if __name__ == "__main__":
    ds = load_dataset(DOCUMENT_DATASET)
    docs = ds['train']['text']

    tokenizer = Tokenizer.from_pretrained(TOKENIZER_NAME)
    chunkier = WordChunker(
        tokenizer=tokenizer,
        chunk_size=512,
        chunk_overlap=128,
    )

    chunked_docs = []
    for doc in tqdm(docs):
        chunks = chunkier.chunk(doc)
        chunked_docs.extend([c.text for c in chunks])

    chunked_docs = random.sample(chunked_docs, 5000)

    retriever = Retriever(docs)
    bot = RAGQuestionAnsweringBot(retriever)

    demo = gradio.Interface(bot.answer_question, inputs='text', outputs='text')
    demo.launch()

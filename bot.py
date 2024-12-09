import litellm

from constants import *
from retriever import Retriever


class RAGQuestionAnsweringBot:
    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    def answer_question(self, question: str):
        print(f'Answering question: {question}\n')

        context = self.retriever.get_docs(question)

        messages = [
            SYSTEM_MESSAGE,
            {'role': 'user', 'content': f'Context:\n{context}\nQuestion: {question}'}
        ]

        answer = litellm.completion(
            model=MODEL_NAME,
            api_base=API_BASE,
            messages=messages
        )

        return answer.json()['choices'][0]['message']['content']

import litellm

from constants import *
from retriever import Retriever

from transformers import pipeline

class RAGQuestionAnsweringBotHuggingFace:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.pipeline = pipeline('text-generation', model='distilgpt2')

    def answer_question(self, question: str):
        documents = self.retriever.get_docs(question)
        context = ''
        for i, doc in enumerate(documents, start=1):
            context += f"""
            DOCUMENT {i}:
            {doc}

            """

        with open('temp.txt', 'w') as file:
            file.write(context)

        message = f'''
        {SYSTEM_MESSAGE['content']}
        
        DOCUMENTS:

        {context}
        
        QUESTION:
        {question}
        '''

        answer = self.pipeline(message, max_length=1024)

        return answer[0]['generated_text']

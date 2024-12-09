DOCUMENT_DATASET = 'm-ric/huggingface_doc'
TOKENIZER_NAME = 'gpt2'
DOCS_QUANTITY = 1000
MODEL_NAME = 'ollama/llama3.2'
OLLAMA_API_BASE = 'http://localhost:11434'
SYSTEM_MESSAGE = {
    'role': 'system',
    'content': '''
    INSTRUCTIONS:
    Answer the users QUESTION using the DOCUMENTS text provided.
    Keep your answer ground in the facts of the DOCUMENT.
    If the DOCUMENTS doesn’t contain the facts to answer the QUESTION return "I don't know"
    '''
}

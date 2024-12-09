DOCUMENT_DATASET = 'm-ric/huggingface_doc'
TOKENIZER_NAME = 'gpt2'
MODEL_NAME = 'ollama/llama3.2'
API_BASE = 'http://localhost:11434'
SYSTEM_MESSAGE = {
    'role': 'system',
    'content':
    '''
    You are given some context and also a question. Answer the question using given context.
    '''
}

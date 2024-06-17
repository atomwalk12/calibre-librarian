from clients.llama3 import messages_to_prompt_v3_instruct
from llama_index.core import set_global_tokenizer
from transformers import AutoTokenizer


#set_global_tokenizer(
#    AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct").encode
#)

CONFIG = {
    'HF': {
            'model_name': "meta-llama/Meta-Llama-3-8B-Instruct",
            'timeout': 120,
            'messages_to_prompt': messages_to_prompt_v3_instruct,
            'task': 'complete'
    },
    'Ollama': {
        'model': 'llama3:70b-instruct',
        #'model': 'llama3:8b-instruct-fp16',
        'request_timeout': 600
    }
}
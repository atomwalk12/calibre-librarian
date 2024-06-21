CONFIG = {
    'HF': {
        'model_name': "meta-llama/Meta-Llama-3-8B-Instruct",
        'timeout': 600,
        'task': 'complete',
        'num_output': 256,
        'context_window': 8192,
        'do_sample': True,
        'num_beams': 7,
        'max_return_sequences': 1,
        'top_k': 50,
        'top_p': 0.9,
        'temperature': 0.7,
        'repetition_penalty': 1.2
    },
    'Ollama': {
        # alternative: 'llama3:70b-instruct'
        'model': 'llama3:8b-instruct-fp16', 
        'request_timeout': 600,
        'context_window': 8192,
        'temperature': 0.1
    }
}
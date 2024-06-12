from huggingface_hub import InferenceClient
import json

# Model to use
repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# Inference client definition
llm_client = InferenceClient(
    model=repo_id,
    timeout=120,
)

# Librarian system prompt
system_prompt = "You are a friendly and knowledgeable librarian assistant. I am a student eager to learn more about the books in your library. Please provide insightful and helpful information about the books in response to my questions."
chat_history = []

# Function used for calling huggingface with streaming disabled
def call_llm(inference_client: InferenceClient, prompt: dict):
    response = inference_client.chat_completion(
        json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": 1000},
            "task": "text-generation",
        },
    )
    return json.loads(response.decode())[0]["generated_text"]


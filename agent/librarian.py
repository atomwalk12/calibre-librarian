from huggingface_hub import InferenceClient
from llama_index.core import SimpleDirectoryReader
import json


class Librarian():
    # Model to use
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Track chat history
    chat_history = []

    # Librarian system prompt
    system_prompt = """
        You are a friendly and knowledgeable librarian assistant. I am a student eager to learn more about the books in your library. 
        Please provide insightful, concise and helpful information about the books in response to my questions.
    """


    def __init__(self, lib_path):
        # Load all books from the library
        self.books = self.load_books(lib_path)

        # Inference client definition
        self.llm_client = InferenceClient(
            model=self.repo_id,
            timeout=120,
        )

    def call_llm(self, inference_client: InferenceClient, prompt: dict):
        """Function used for calling huggingface with streaming disabled"""
        response = inference_client.chat_completion(
            json={
                "inputs": prompt,
                "parameters": {"max_new_tokens": 1000},
                "task": "text-generation",
            },
        )
        return json.loads(response.decode())[0]["generated_text"]


    def load_books(self, path):
        """Used to recursively load all books found in the library"""
        loader = SimpleDirectoryReader(
            input_dir=path,
            recursive=True,
            required_exts=[".epub"],
        )

        documents = loader.load_data()
        return documents

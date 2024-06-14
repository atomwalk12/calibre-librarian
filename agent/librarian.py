from huggingface_hub import InferenceClient
from llama_index.core import SimpleDirectoryReader
import json
import logging


class Librarian:
    # Track chat history
    chat_history = []

    # Librarian system prompt
    system_prompt = """
        You are a friendly and knowledgeable librarian assistant. I am a student eager to learn more about the books in your library. 
        Please provide insightful, concise and helpful information about the books in response to my questions.
    """

    def __init__(self, name, lib_path, model_id):
        """
        Initializes the Librarian, which represents a wrapper around the LLM.

        :param name: User's name.
        :param lib_path: The path where the books are stored.
        :param model_id: The Huggingface model id.
        """
        # User's name
        self.name = name

        # Load all books from the library
        self.books = self.load_books(lib_path)

        # Logger
        self.logger = logging.getLogger("librarian_logger")

        # Inference client definition
        self.llm_client = InferenceClient(
            model=model_id,
            timeout=120,
        )

    def stream_response(self):
        """
        The response from the LLM is streamed to the Gradio model via the yield keyword.
        """
        logging.info(f"Chat history: {self.chat_history}")
        full_message = ""
        for token in self.llm_client.chat_completion(
            self.chat_history, max_tokens=100, stream=True
        ):
            # Some magic used by the Gradio library to stream response to GUI
            full_message += token.choices[0].delta.content
            yield full_message

        return full_message

    def generate_response(self, inference_client: InferenceClient, prompt: dict):
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
        # loader = SimpleDirectoryReader(
        #     input_dir=path,
        #     recursive=True,
        #     required_exts=[".epub"],
        # )

        # documents = loader.load_data()
        documents = []
        return documents

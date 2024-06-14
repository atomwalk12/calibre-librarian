import json
import logging
import os
from huggingface_hub import InferenceClient
from llama_index.core import StorageContext, load_index_from_storage, ServiceContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceInferenceAPI



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

        # Logger
        self.logger = logging.getLogger("librarian_logger")

        # Inference client definition
        self.llm_client = HuggingFaceInferenceAPI(
            model_name=model_id,
            timeout=120,
        )

        # Store in vector store
        self.query_engine = self.create_index(lib_path)



    def stream_response(self):
        """
        The response from the LLM is streamed to the Gradio model via the yield keyword.
        """
        logging.info(f"Chat history: {self.chat_history}")
        full_message = ""
        for token in self.llm_client.chat_completion(
            self.chat_history, max_tokens=500, stream=True
        ):
            # Some magic used by the Gradio library to stream response to GUI
            full_message += token.choices[0].delta.content
            yield full_message

        return full_message

    def generate_response(self, inference_client: InferenceClient, prompt: dict):
        """
        Function used for calling huggingface with streaming disabled.
        """
        response = inference_client.chat_completion(
            json={
                "inputs": prompt,
                "parameters": {"max_new_tokens": 1000},
                "task": "text-generation",
            },
        )
        return json.loads(response.decode())[0]["generated_text"]

    def load_books(self, path):
        """
        Used to recursively load all books found in the library.
        """
        loader = SimpleDirectoryReader(
            input_dir="./books",
            recursive=True,
            required_exts=[".epub"],
        )

        documents = loader.load_data()
        return documents

    def create_index(self, lib_path):
        """
        Used to create the index for RAG retrieval.
        """
        # Use a quick embedding model
        embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        service_context = ServiceContext.from_defaults(llm_predictor=self.llm_client, embed_model=embedding_model) 

        # Use a light and fast vector store
        if os.path.isdir('./index'):
            # Load all books from the library
            

            # rebuild storage context
            storage_context = StorageContext.from_defaults(persist_dir="./index")

            # load index
            index = load_index_from_storage(storage_context=storage_context, service_context=service_context, embedding_model='local')
        else:
            # Load the books
            books = self.load_books(lib_path)

            # create index
            index = VectorStoreIndex.from_documents(
                books,
                embed_model=embedding_model,
                service_context=service_context
            )

            # persist index for subsequent faster retrieval
            index.storage_context.persist(persist_dir='./index',)

        query_engine = index.as_query_engine(llm=self.llm_client)
        return query_engine

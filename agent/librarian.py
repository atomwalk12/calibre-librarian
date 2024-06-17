import json
import logging
import os
from typing import List
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)
from huggingface_hub import InferenceClient
from llama_index.core import StorageContext, load_index_from_storage, ServiceContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from clients.inference import inference_factory


class Librarian:
    # Track chat history
    chat_history: List[ChatMessage] = []

    # Librarian system prompt
    system_prompt = """
        You are a friendly and knowledgeable librarian assistant. I am a student eager to learn more about the books in your library. 
        In your library you can find: The Brothers Karamazov by Fyodor Dostoyevsky, Pride and Prejudice by Jane Austen, Anna Karenina by Leo Tolstoy, War and Peace by Leo Tolstoy and Hamlet by William Shakespeare.
        Please provide insightful, concise and helpful information about the books in response to my questions.

        What are the books stored in the library?
    """

    def __init__(self, inference_client, lib_path):
        """
        Initializes the Librarian, which represents a wrapper around the LLM.

        :param name: User's name.
        :param lib_path: The path where the books are stored.
        :param model_id: The Huggingface model id.
        """

        # Add system prompt to history
        self.history_add(ChatMessage(content=self.system_prompt, role=MessageRole.SYSTEM))

        # Logger
        self.logger = logging.getLogger("librarian_logger")

        # Inference client definition
        self.llm_client = inference_factory(inference_client)

        # Store in vector store
        self.query_engine = self.create_index(lib_path)
        self.agent = self.create_agent(self.query_engine)



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


    def use_query_engine(self, user_message: ChatMessage):
        """
        Used via the query engine to retrieve information from the library.
        """

        # Add user message to history
        # self.history_add(user_message)

        # Generate the response
        logging.info(f"{self.chat_history}")
        
        # Todo remove these lines
        # response = self.query_engine.query(user_message.content)
        # response = self.llm_client.chat(self.chat_history)

        response = self.agent.stream_chat(user_message.content)
        response.print_response_stream()
        logging.info(f"""Question: {user_message.content}\n
                     Response: {response=}\n
                     Chat history: {self.chat_history=}\n""")

        # Add assistant message to history
        self.history_add(ChatMessage(content=response, role=MessageRole.ASSISTANT))
        
        # The response for the query imbued with RAG context
        return response


    def history_add(self, message):
        """
        Add a given message to the chat history.
        """
        self.chat_history.append(message)


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
            input_dir=path,
            recursive=True,
            required_exts=[".epub"],
        )
        
        documents = loader.load_data()
        
        logging.info(f"Loaded documents: {documents}")
        return documents

    def create_index(self, lib_path):
        """
        Used to create the index for RAG retrieval.
        """
        # Use a quick embedding model
        embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        service_context = ServiceContext.from_defaults(llm=self.llm_client, embed_model=embedding_model) 

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

        query_engine = index.as_query_engine(llm=self.llm_client, similarity_top_k=5, verbose=True)
        return query_engine


    def create_agent(self, query_engine):
        """
        Creates a ReAct agent with the aim to use RAG retrieval tools to answer questions.
        This way the data is bundled across different query engines, yielding focused context.
        """
        # Define the engines with a short description describing when to use the tool
        query_engine_tools = [
            QueryEngineTool(
                query_engine=query_engine,
                metadata=ToolMetadata(
                    name="Hamlet",
                    description=(
                        "Provides information about the book Hamlet by William Shakespeare. "
                        "Use a detailed plain text question as input to the tool."
                    ),
                ), 
            ),
        ]

        # Generates the agent to use tools and the desired llm client
        agent = ReActAgent.from_tools(
            query_engine_tools,
            llm=self.llm_client,
            verbose=True,
            request_timeout=600
        )
        return agent
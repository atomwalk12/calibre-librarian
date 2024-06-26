import logging
import os
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import List

from lib.inference import inference_factory
from llama_index.core import (
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.agent import ReActAgent
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)
from llama_index.core.node_parser.text import SemanticSplitterNodeParser
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from lib.utils import find_pdfs, split_text, trim_to_num_sentences


class Librarian:
    """
    Librarian acts as a wrapper around the agent used to retrieve data (main class).
    """

    chat_history: List[ChatMessage] = []

    def __init__(self, inference_client, lib_path, extension):
        """
        Initializes the Librarian, which represents a wrapper around the LLM.

        :param inference_client: The client type. Can be Ollama, HF or OpenAI.
        :param lib_path: The path where the books are stored.
        :param extension: Extension for the books.
        :param model_id: The Huggingface model id.
        """

        # When using HF need to trim output to a given number of sentences (using text generation)
        self.using_hf = inference_client == "HF"

        # Retrieve top results from the index
        self.similarity_top_k = 3

        # Logger
        self.logger = logging.getLogger("librarian_logger")

        # Inference client definition
        self.llm_client = inference_factory(inference_client)
        self.lib_path = lib_path

        # Load books
        self.books = find_pdfs(self.lib_path, extension)

        # Store in vector store
        query_engines = self.create_index(lib_path, extension)
        self.agent = self.create_agent(query_engines)

    def stream_response(self):
        """
        The response from the LLM is streamed to the Gradio model via the yield keyword.
        """
        logging.info(f"Chat history: {self.chat_history}")

        full_message = ""
        for token in self.llm_client.chat_completion(self.chat_history, max_tokens=500):
            # Some magic used by the Gradio library to stream response to GUI
            full_message += token.choices[0].delta.content
            yield full_message

        return full_message
    

    def use_query_engine(self, user_message: str, history: List[str]):
        """
        Used via the query engine to retrieve information from the library.

        param: user_message: str - The user's message to be processed by the LLM
        param: history: list - The current conversation history
        """

        # Synchronize chat history with Gradio history
        while len(self.chat_history) > len(history) * 2:
            self.chat_history.pop()

        # Prompt the agent given the input
        with StringIO() as buffer, redirect_stdout(buffer):
            response = self.agent.chat(user_message, chat_history=self.chat_history)
            text = buffer.getvalue()

        # Print the output of the agent
        print(text)

        # Yield the output to the agent when queried to list available books
        display_tool_user = False
        for tool in response.sources:
            if tool.tool_name == self.display_books.__name__:
                display_tool_user = True
                yield response.response

        # If the agent did not use the display books tool, then a show book results
        if not display_tool_user:
            thoughts, observations = split_text(text)

            citation_idx = 0
            for idx in range(len(thoughts)):

                # Discard messages that were the result of parsing errors
                if idx < len(observations) and "Error" in observations[idx]:
                    continue

                # Print current thought
                yield "\n\n" + thoughts[idx]

                # Print citations
                for _ in range(self.similarity_top_k):

                    # Some tools do not require RAG operations and as a result citation index may be out of bounds
                    if citation_idx >= len(response.source_nodes):
                        break

                    yield "\n\n" + response.source_nodes[citation_idx].node.get_text()
                    citation_idx += 1

                # In some cases an observation may include redundant text due to parsing errors
                if idx >= len(observations):
                    continue

                yield "\n\n" + observations[idx]

            # When using the HF api trim for at most 4 sentences, since chat functionality is not supported
            if self.using_hf:
                yield "\n\n" + trim_to_num_sentences(response.response, 4)
            else:
                yield "\n\n" + response.response


        # Add user and assistant messages to history
        self.history_add(ChatMessage(content=user_message, role=MessageRole.USER))
        self.history_add(
            ChatMessage(content=response.response, role=MessageRole.ASSISTANT)
        )

        # The response for the query imbued with RAG context
        return response

    def history_add(self, message: ChatMessage):
        """
        Add a given message to the chat history.

        param: message (ChatMessage) - The message to add to the chat history
        """
        self.chat_history.append(message)


    def load_book(self, path, ext):
        """
        Used to recursively load all books found in the library.

        param: path (str) - The path to the library of books
        param: ext (str) - The extension of the files to load
        """
        loader = SimpleDirectoryReader(
            input_files=[path],
            required_exts=[ext],
        )

        documents = loader.load_data()

        logging.info(f"Loaded documents: {documents}")
        return documents

    def create_index(self, lib_path:str , ext: str):
        """
        Used to create the index for RAG retrieval.
        """
        # Use a quick embedding model
        embedding_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")

        service_context = ServiceContext.from_defaults(
            llm=self.llm_client,
            embed_model=embedding_model,
            node_parser=SemanticSplitterNodeParser(
                embed_model=HuggingFaceEmbedding(
                    model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"
                ),
            ),
        )

        indexes = []
        # Use a light and fast vector store
        if os.path.isdir("./index"):
            for book in self.books:
                name = Path(book["name"]).stem
                # rebuild storage context
                storage_context = StorageContext.from_defaults(
                    persist_dir=f"./index/{book['author']}/{name}"
                )

                # load index
                index = load_index_from_storage(
                    storage_context=storage_context,
                    service_context=service_context,
                    embedding_model="local",
                )

                # Prepare the index for the citation engine
                indexes.append(index)
        else:
            for book in self.books:
                cur_docs = self.load_book(
                    f"{lib_path}/{book['author']}/{book['name']}", ext
                )

                # create index
                index = VectorStoreIndex.from_documents(
                    cur_docs,
                    embed_model=embedding_model,
                    service_context=service_context,
                )

                # persist index for subsequent faster retrieval
                name = Path(book["name"]).stem
                index.storage_context.persist(
                    persist_dir=f"./index/{book['author']}/{name}"
                )

                # Store the index to create the citation engine
                indexes.append(index)

        query_engines = []

        for index in indexes:
            # Create the dedicated query engine
            query_engine = CitationQueryEngine(
                index.as_retriever(
                    similarity_top_k=self.similarity_top_k,
                    vector_store_query_mode=VectorStoreQueryMode.SVM,
                ),
                citation_chunk_size=1024,
            )

            query_engines.append(query_engine)

        return query_engines

    def create_agent(self, query_engines):
        """
        Creates a ReAct agent with the aim to use RAG retrieval tools to answer questions.
        This way the data is bundled across different query engines, yielding focused context.

        param: query_engines: List of query engines to use to answer questions
        """

        citation_engines = []
        for idx in range(len(query_engines)):
            book = self.books[idx]
            name = Path(book["name"]).stem

            tool = QueryEngineTool(
                query_engine=query_engines[idx],
                metadata=ToolMetadata(
                    name=name.replace(" ", ""),
                    description=(
                        f"Provides information about the book {name} by {book['author']}. "
                        "Use a detailed plain text question as input to the tool."
                    ),
                ),
            )
            citation_engines.append(tool)

        # Define the engines with a short description describing when to use the tool
        query_engine_tools = [
            FunctionTool.from_defaults(
                fn=self.display_books,
                description=(
                    "Provides information about the available books in the library. Only use this tool when asked explicitly about the available books."
                    "There is no input to the tool."
                ),
                return_direct=True,
            )
        ] + citation_engines

        # Generates the agent to use tools and the desired llm client
        agent = ReActAgent.from_tools(
            query_engine_tools,
            llm=self.llm_client,
            verbose=True,
            request_timeout=600,
            max_iterations=25,
        )
        return agent


    def display_books(self):
        """
        Used for retrieving the available books to the user using the format <author: book_name>.
        """
        msg = "Below are the available books in the library:\n\n"
        msg += "\n".join(
            [
                f"Author: {book['author']}, Book name: {Path(book['name']).stem}"
                for book in self.books
            ]
        )
        return msg

import gradio as gr
import argparse
import logging
from agent.librarian import Librarian
from gradio.blocks import Blocks


class GradioGUI(Blocks):
    def __init__(self, name, lib_path, model_id):
        """
        Initializes the GUI, by loading the Librarian LLM.
        """
        self.agent = Librarian(name=name, lib_path=lib_path, model_id=model_id)
        self.logger = logging.getLogger("librarian_logger")

    def predict(self, user_input, _):
        """
        Wrapper around the GUI, called each time a new message is triggered.

        :param user_input: message inserted by the user.
        """
        chat_history = self.agent.chat_history

        # If it's the start of the conversation, include the system prompt
        if chat_history is None or len(chat_history) == 0:
            chat_history.append({"role": "system", "content": self.agent.system_prompt})

        # Append the user message
        chat_history.append({"role": "user", "content": user_input})

        # User friendly way to stream response 
        full_message = ""
        for token in self.agent.stream_response():
            yield token
            full_message += token

        # Append the assistant's response to the chat history
        chat_history.append({"role": "assistant", "content": full_message})

    def launch(self):
        """
        Launch the user interface.
        """
        demo = gr.ChatInterface(self.predict)
        demo.queue().launch()


if __name__ == "__main__":
    # Arguments with sensible default values
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="Razvan")
    parser.add_argument(
        "--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    parser.add_argument("--lib_path", type=str, default="./books")
    args, unknown = parser.parse_known_args()

    # Configure the logger to facilitate development
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler("debug.log")],
    )

    # Serve the server 
    demo = GradioGUI(name=args.name, model_id=args.model_id, lib_path=args.lib_path)
    demo.launch()
 
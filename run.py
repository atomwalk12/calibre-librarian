from lib.librarian import Librarian

import gradio as gr
import argparse
import logging


def predict(user_input, history):
    """
    Wrapper around the GUI, called each time a new message is triggered.

    :param user_input: message inserted by the user.
    :param history: the previous history messages
    """
    # User friendly way to stream response
    msg = ""
    for token in librarian.use_query_engine(user_input, history):
        msg += token
        yield msg

    return msg


# Serve the server
demo = gr.ChatInterface(predict)


# Arguments with sensible default values
parser = argparse.ArgumentParser()
parser.add_argument("--extension", type=str, default=".pdf")
parser.add_argument("--inference-client", type=str, default="HF")
parser.add_argument("--lib-path", type=str, default="./books")
args, unknown = parser.parse_known_args()


# Configure the logger to facilitate development
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("debug.log")],
)


# Create the agent
librarian = Librarian(
    inference_client=args.inference_client,
    lib_path=args.lib_path,
    extension=args.extension,
)

if __name__ == "__main__":
    demo.queue().launch()

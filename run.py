import gradio as gr
import argparse
from agent.librarian import Librarian

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="User")
args, unknown = parser.parse_known_args()

agent = Librarian("./books")
# Gradio
def predict(user_input, _):
    global chat_history

    # If it's the start of the conversation, include the system prompt
    if chat_history is None or len(chat_history) == 0:
        chat_history = [{"role": "system", "content": agent.system_prompt}]
    else:
        chat_history = chat_history.copy()  # Make a copy to avoid modifying the input directly

    # Append the user message
    chat_history.append({"role": "user", "content": user_input})

    # Display librarian's answer
    full_message = ""
    for token in agent.llm_client.chat_completion(chat_history, max_tokens=100, stream=True):
        full_message += token.choices[0].delta.content

        yield full_message

    # Append the assistant's response to the chat history
    chat_history.append({"role": "assistant", "content": full_message})



demo = gr.ChatInterface(predict) 
demo.queue().launch()


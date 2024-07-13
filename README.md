# Librarian LLM Agent
LLM powered agent for answering questions about books stored in a local library. 

Check out the 4-minute video explanation [here](https://www.youtube.com/watch?v=63E1ebuUNcs) and paper [here](docs/report.pdf).


![poster](https://github.com/user-attachments/assets/467c1225-db01-4912-a575-f674269c62d0)

Below is a 1 minute presentation showcasing the agent answering a total of 9 questions about 3 books, 3 questions for each individual book.

https://github.com/user-attachments/assets/00f426fd-d46a-46ad-86d5-8f6622f39d58





## Features
Below are some properties of the project:

 - **Remote and Local LLMs.** The agent works both with local and cloud LLMs.
   - **HuggingFace API.** The cloud models leverage the HuggingFace free web API to answer queries, which means that the library is also usable without local hosting. The library uses by default the Llama 3 8b instruct model.
   - **Ollama.** The local models are hosted via the Ollama backend. The local models can be more powerful which means that the quality of the results tends to be better. Possible Ollama models are [listed here](https://ollama.com/library).
 - **Prompts leveraging book context.** The agent answers a series of questions by retrieving information repeatedly from the corresponding book. The information is used to improve the quality of the answer.
   - **Citations.** They are displayed along the answer, together with partial answers, which are used to provide a more thorough and informed response.
 - **Available books.** It is possible to query the agent for the available books but new ones can be added - this requires to rebuild the index. Currently, the project provides indices for three books:
   - The Little Prince by Antoine de Saint-Exupéry.
   - Ethics by Baruch Spinoza.
   - The Picture of Dorian Gray by Oscar Wilde.
 - **Intuitive GUI.** Using the Gradio library, with the ability to reset history, repeat or edit previous questions.

## Building the project
### Ollama
If you'd like to use the local model, follow the instruction [here](https://github.com/ollama/ollama) to install Ollama. The project is currently configured to work with Llama 3 8b and Llama 3 70b (check [config.py](config.py)).

### Huggingface API
If you would like to use a cloud model, you don't need to do anything else.

### Local environment setup
Build the local python environment. 
```console
> git clone git@github.com:atomwalk12/calibre-librarian.git librarian
> cd librarian
> python -m venv env
> source env/bin/activate
> pip install -r requirements.txt
```
For the Ollama client use:
```console
> python run.py --inference-client Ollama
```
For HuggingFace
```console
> python run.py --inference-client HF
```

The model to use is configurable from [config.py](config.py).

### Adding new books
To add new books, add pdf files to the [books](books/) directory. Each file must be placed in a directory named after the author, with each individual file name corresponding to the title of book.
Once the index is generated, remove the 📁Index folder to regenerate it.

For example the directory structure should look something like this:
```
📁 books
├─ 📚 Antoine de Saint Exupery
│  └─ 🗒️ Little Prince.pdf
├─ 📚 Baruch Spinoza
│  └─ 🗒️ Ethics.pdf
└─ 📚 Oscar Wilde
   └─ 🗒️ The Picture of Dorian Gray.pdf
```

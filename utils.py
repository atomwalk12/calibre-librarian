import re
import os


def split_text(input_text):
    """
    Parsing of the text generated by the LLM, to be printed in the order: thought, context, observations.

    param: input_text: the code 
    """
    # Define the pattern to match 'Thought:' or 'Observation:' followed by text.
    pattern = r"(Thought:(.|\n)*?\})\s*|(Observation:.*?\.)\s*"
    matches = re.findall(pattern, input_text)

    # Since the list previously retrieved is made of tuples, it must be flattened.
    result = [item for sublist in matches for item in sublist if item]

    # Create two arrays with each individual thought and observation.
    observations = []
    thoughts = []
    for item in result:
        if item.startswith("Observation") and "Error" not in item:
            observations.append(item)
        if item.startswith("Thought"):
            thoughts.append(item)
    return thoughts, observations


def find_pdfs(root_dir, extension):
    """
    Finds all available books in the root directory.
    """
    books = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:                            # iterate all files recursively
            if filename.lower().endswith(extension):
                full_path = os.path.join(dirpath, filename)
                parent_dir = os.path.dirname(full_path)

                # Append result in (author, book name) format.
                books.append({"author": os.path.basename(parent_dir), "name": filename})

    return books



def trim_to_num_sentences(text, num_sentences):
    """
    Since LlamaIndex has no support for chat inference via the HuggingFaceInferenceAPI,
    we generate a fixed number of tokens (256) then trim for a given number of sentences. 

    param: text: text generated by the LLM.
    param: num_sentences: the number of sentences to retrieve starting from the beginning.    
    """
    sentences = text.split('.')
    trimmed_text = '.'.join(sentences[:num_sentences]) + '.' if len(sentences) > num_sentences else text
    return trimmed_text
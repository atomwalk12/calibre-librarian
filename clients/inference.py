from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from config import CONFIG
from llama_index.llms.ollama import Ollama



def inference_factory(inference_client):
    """Factory Method"""
    providers = {
        "Ollama": Ollama,  # Placeholder if 'Ollama' service class is to be implemented
        "HF": HuggingFaceInferenceAPI,
        "OpenAI": None,  # Placeholder if 'OpenAI' service class is to be implemented
    }

    provider_class = providers.get(inference_client)
    if provider_class is None:
        raise ValueError(f"No implementation available for provider {inference_client}")

    # Instantiate the provider class with any required parameters
    provider_instance = provider_class(**CONFIG[inference_client])

    return provider_instance

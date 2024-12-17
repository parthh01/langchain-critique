from importlib import metadata

from langchain_critique.chat_models import ChatCritique
from langchain_critique.document_loaders import CritiqueLoader
from langchain_critique.embeddings import CritiqueEmbeddings
from langchain_critique.retrievers import CritiqueRetriever
from langchain_critique.toolkits import CritiqueToolkit
from langchain_critique.tools import CritiqueTool
from langchain_critique.vectorstores import CritiqueVectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatCritique",
    "CritiqueVectorStore",
    "CritiqueEmbeddings",
    "CritiqueLoader",
    "CritiqueRetriever",
    "CritiqueToolkit",
    "CritiqueTool",
    "__version__",
]

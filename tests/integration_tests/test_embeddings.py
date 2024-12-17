"""Test Critique embeddings."""

from typing import Type

from langchain_critique.embeddings import CritiqueEmbeddings
from langchain_tests.integration_tests import EmbeddingsIntegrationTests


class TestParrotLinkEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[CritiqueEmbeddings]:
        return CritiqueEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}

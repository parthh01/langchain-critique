from typing import Type

from langchain_critique.retrievers import CritiqueRetriever
from langchain_tests.integration_tests import (
    RetrieversIntegrationTests,
)


class TestCritiqueRetriever(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> Type[CritiqueRetriever]:
        """Get an empty vectorstore for unit tests."""
        return CritiqueRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        return {"k": 2}

    @property
    def retriever_query_example(self) -> str:
        """
        Returns a dictionary representing the "args" of an example retriever call.
        """
        return "example query"

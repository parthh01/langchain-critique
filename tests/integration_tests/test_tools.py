from typing import Type
import os
import pytest
from langchain_tests.integration_tests import ToolsIntegrationTests
from langchain_critique.tools import CritiqueSearchTool, CritiqueAPIDesignTool

@pytest.mark.skipif(
    "CRITIQUE_API_KEY" not in os.environ,
    reason="Critique API key not found in environment"
)
class TestCritiqueSearchToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[CritiqueSearchTool]:
        return CritiqueSearchTool

    @property
    def tool_constructor_params(self) -> dict:
        return {}  # Will use env var

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "prompt": "What are the latest developments in AI?",
            "source_blacklist": ["unreliable-news.com"],
            "output_format": {"key_points": ["string"]}
        }

class TestCritiqueSearchToolCustomIntegration:
    def test_image_search(self):
        tool = CritiqueSearchTool()
        result = tool.invoke({
            "prompt": "What is this building?",
            "image": "https://example.com/test-image.jpg",
            "output_format": {
                "building_info": {
                    "name": "string",
                    "location": "string",
                    "year_built": "number"
                }
            }
        })
        assert "response" in result
        assert "citations" in result

@pytest.mark.skipif(
    "CRITIQUE_API_KEY" not in os.environ,
    reason="Critique API key not found in environment"
)
class TestCritiqueAPIDesignToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[CritiqueAPIDesignTool]:
        return CritiqueAPIDesignTool

    @property
    def tool_constructor_params(self) -> dict:
        return {}  # Will use env var

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "operation": "create",
            "prompt": "Create an API that takes a company name and returns their ESG score"
        }

class TestCritiqueAPIDesignToolCustomIntegration:
    def test_api_lifecycle(self):
        tool = CritiqueAPIDesignTool()
        # Test implementation here

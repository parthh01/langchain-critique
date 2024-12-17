from typing import Type
import os
import pytest

from langchain_critique.tools import CritiqueSearchTool, CritiqueAPIDesignTool
from langchain_tests.integration_tests import ToolsIntegrationTests


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

    def test_image_search(self):
        tool = self.tool_constructor(**self.tool_constructor_params)
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

    def test_api_lifecycle(self):
        tool = self.tool_constructor(**self.tool_constructor_params)
        
        # Create API
        create_result = tool.invoke({
            "operation": "create",
            "prompt": "Create an API that takes a company name and returns their ESG score"
        })
        api_id = create_result["id"]
        
        # Update API
        update_result = tool.invoke({
            "operation": "update",
            "api_id": api_id,
            "prompt": "Add carbon footprint metrics to the response"
        })
        
        # List APIs
        list_result = tool.invoke({"operation": "list"})
        assert any(api["id"] == api_id for api in list_result)
        
        # Delete API
        delete_result = tool.invoke({
            "operation": "delete",
            "api_id": api_id
        })

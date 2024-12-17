from typing import Type, List
import pytest

from langchain_critique.tools import CritiqueSearchTool, CritiqueAPIDesignTool, CritiqueDynamicAPITool,DynamicSchemaDefinition
from langchain_tests.unit_tests import ToolsUnitTests


class TestCritiqueSearchToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> Type[CritiqueSearchTool]:
        return CritiqueSearchTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": "fake_key"}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "prompt": "What are the latest developments in AI?",
            "source_blacklist": ["unreliable-news.com"],
            "output_format": {"key_points": ["string"]}
        }

class TestCritiqueSearchToolCustomUnit:
    def test_image_validation(self):
        tool = CritiqueSearchTool(api_key="fake_key")
        
        # Test valid image URL
        valid_url = "http://example.com/image.jpg"
        assert tool._validate_image(valid_url) == valid_url
        
        # Test invalid image URL
        with pytest.raises(ValueError):
            tool._validate_image("http://example.com/not-an-image")
            
        # Test valid base64
        valid_base64 = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
        assert tool._validate_image(valid_base64) == valid_base64
        
        # Test invalid base64
        with pytest.raises(ValueError):
            tool._validate_image("not-base64-or-url")

class TestCritiqueAPIDesignToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> Type[CritiqueAPIDesignTool]:
        return CritiqueAPIDesignTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": "fake_key"}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "operation": "create",
            "prompt": "Create an API that takes a company name and returns their ESG score"
        }

# Move non-standard tests to custom test classes
class TestCritiqueAPIDesignToolCustomUnit:
    def test_api_operations(self):
        tool = CritiqueAPIDesignTool(api_key="fake_key")
        
        # Test create operation validation
        with pytest.raises(ValueError, match="Create operation requires 'prompt' parameter"):
            tool.invoke({"operation": "create"})
            
        # Test update operation validation
        with pytest.raises(ValueError, match="Update operation requires 'api_id' parameter"):
            tool.invoke({"operation": "update"})
            
        # Test delete operation validation
        with pytest.raises(ValueError, match="Delete operation requires 'api_id' parameter"):
            tool.invoke({"operation": "delete"})

class TestCritiqueDynamicAPIToolUnit:
    def test_dynamic_schema_creation(self):
        schema_definition = {
            "company_name": DynamicSchemaDefinition(
                type=str,
                description="Name of the company"
            ),
            "metrics": DynamicSchemaDefinition(
                type=list,
                description="List of metrics to include",
                items_type=str
            )
        }
        
        tool = CritiqueDynamicAPITool(
            api_id="test_api",
            name="ESG Score API",
            description="Get ESG scores for companies",
            schema_definition=schema_definition,
            api_key="fake_key"
        )
        
        # Verify schema creation
        assert tool.args_schema is not None
        assert "company_name" in tool.args_schema.model_fields
        assert "metrics" in tool.args_schema.model_fields
        
        # Test schema validation
        valid_input = {
            "company_name": "Test Corp",
            "metrics": ["environmental", "social"]
        }
        result = tool.invoke(valid_input)
        assert result["result"] == "success"
        
        # Test invalid input
        with pytest.raises(ValueError):
            tool.invoke({"company_name": "Test Corp"})  # Missing required metrics

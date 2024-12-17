from typing import Type
import pytest

from langchain_critique.tools import CritiqueSearchTool, CritiqueAPIDesignTool, CritiqueDynamicAPITool
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

    def test_api_operations(self):
        tool = CritiqueAPIDesignTool(api_key="fake_key")
        
        # Test create operation validation
        with pytest.raises(ValueError):
            tool._run(operation="create")  # Missing prompt
            
        # Test update operation validation
        with pytest.raises(ValueError):
            tool._run(operation="update")  # Missing api_id
            
        # Test delete operation validation
        with pytest.raises(ValueError):
            tool._run(operation="delete")  # Missing api_id

class TestCritiqueDynamicAPIToolUnit:
    def test_dynamic_schema_creation(self):
        input_schema = {
            "company_name": {
                "type": str,
                "description": "Name of the company"
            },
            "metrics": {
                "type": List[str],
                "description": "List of metrics to include"
            }
        }
        
        tool = CritiqueDynamicAPITool(
            api_id="test_api",
            name="ESG Score API",
            description="Get ESG scores for companies",
            input_schema=input_schema,
            api_key="fake_key"
        )
        
        # Verify schema creation
        assert hasattr(tool, "args_schema")
        assert "company_name" in tool.args_schema.__fields__
        assert "metrics" in tool.args_schema.__fields__

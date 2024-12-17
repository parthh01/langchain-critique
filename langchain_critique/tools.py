"""Critique tools."""

from typing import Dict, List, Optional, Type, Union
from urllib.parse import urlparse
import base64
import requests
import os
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from datetime import datetime

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, validator, create_model


class CritiqueSearchInput(BaseModel):
    """Input schema for Critique search tool.

    This docstring is not part of what is sent to the model when performing tool
    calling. The Field default values and descriptions are part of what is sent to
    the model when performing tool calling.
    """
    prompt: str = Field(..., description="Search query or question to ask")
    image: Optional[str] = Field(None, description="Optional image URL or base64 string to search with")
    source_blacklist: Optional[List[str]] = Field(
        default=[], 
        description="Optional list of domain names to exclude from search results"
    )
    output_format: Optional[Dict] = Field(
        default={}, 
        description="Optional structured output format specification"
    )

    @validator('image')
    def validate_image(cls, v):
        if v is None:
            return v
        if v.startswith('http'):
            try:
                result = urlparse(v)
                if not all([result.scheme, result.netloc]):
                    raise ValueError("Invalid URL format")
                if not any(v.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                    raise ValueError("URL must point to an image file")
            except Exception as e:
                raise ValueError(f"Invalid image URL: {str(e)}")
        elif v.startswith('data:image'):
            try:
                header, data = v.split(',', 1)
                base64.b64decode(data)
            except Exception:
                raise ValueError("Invalid base64 image format")
        else:
            raise ValueError("Image must be either a URL or base64 encoded string")
        return v


class CritiqueSearchTool(BaseTool):
    """Critique search tool for agentic, grounded search with optional image input.

    Setup:
        Install ``langchain-critique`` and set environment variable ``CRITIQUE_API_KEY``.
        Get your API key at https://critiquebrowser.app/en/flow-api?view=keys

        .. code-block:: bash

            pip install -U langchain-critique
            export CRITIQUE_API_KEY="your-api-key"

    For detailed API documentation, visit: https://critiquebrowser.app/en/flow-api?view=usage

    Key init args:
        api_key: str
            Critique API key. If not provided, will look for CRITIQUE_API_KEY env var.
        base_url: str
            Base URL for Critique API. Defaults to https://api.critiquebrowser.app

    Instantiation:
        .. code-block:: python

            from langchain_critique import CritiqueSearchTool

            tool = CritiqueSearchTool(
                api_key="your-api-key",  # Optional if env var is set
                base_url="https://api.critiquebrowser.app"  # Optional
            )

    Invocation with args:
        .. code-block:: python

            tool.invoke({
                "prompt": "What are the latest developments in AI?",
                "source_blacklist": ["unreliable-news.com"],
                "output_format": {"key_points": ["string"]}
            })

        .. code-block:: python

            {
                "response": "Here are the key developments in AI...",
                "citations": [
                    {"text": "...", "url": "..."}
                ]
            }

    Invocation with image:
        .. code-block:: python

            tool.invoke({
                "prompt": "What is this building?",
                "image": "https://example.com/image.jpg",
                "output_format": {
                    "building_info": {
                        "name": "string",
                        "location": "string",
                        "year_built": "number"
                    }
                }
            })
    """

    name: str = "critique_search"
    description: str = (
        "Search tool that provides grounded, cited responses to queries with optional "
        "image input and structured output format specification."
    )
    args_schema: Type[BaseModel] = CritiqueSearchInput

    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: str = "https://api.critiquebrowser.app",
    ):
        """Initialize the Critique search tool."""
        super().__init__()
        self.api_key = api_key or os.getenv("CRITIQUE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Critique API key must be provided either through api_key parameter "
                "or CRITIQUE_API_KEY environment variable"
            )
        self.base_url = base_url
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }

    def _image_to_base64(self, image_url: str) -> str:
        """Convert image URL to base64 string."""
        response = requests.get(image_url)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch image. Status code: {response.status_code}")
        base64_encoded = base64.b64encode(response.content).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_encoded}"

    def _run(
        self,
        prompt: str,
        image: Optional[str] = None,
        source_blacklist: Optional[List[str]] = None,
        output_format: Optional[Dict] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict:
        """Execute the search."""
        if image and image.startswith('http'):
            image = self._image_to_base64(image)

        payload = {
            "prompt": prompt,
            "source_blacklist": source_blacklist or [],
            "output_format": output_format or {}
        }
        if image:
            payload["image"] = image

        response = requests.post(
            f"{self.base_url}/v1/search",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise ValueError(f"Search failed: {response.text}")
            
        return response.json()


class APIOperation(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    LIST = "list"

class CritiqueAPIDesignInput(BaseModel):
    """Input schema for Critique API design tool."""
    operation: APIOperation = Field(
        ..., 
        description="Operation to perform: 'create', 'update', 'delete', or 'list'"
    )
    prompt: Optional[str] = Field(
        None,
        description="Natural language description of the API to create or update"
    )
    api_id: Optional[str] = Field(
        None,
        description="ID of the API to update or delete"
    )
    schema_updates: Optional[Dict] = Field(
        None,
        description="Updates to apply to an existing API's schema"
    )

class CritiqueAPIDesignTool(BaseTool):
    """Critique API design tool for creating and managing APIs through natural language.

    This tool allows creating, updating, and managing APIs through natural language prompts.
    The created APIs can then be used as dynamic tools in LangChain.

    Setup:
        Install ``langchain-critique`` and set environment variable ``CRITIQUE_API_KEY``.
        Get your API key at https://critiquebrowser.app/en/flow-api?view=keys

        .. code-block:: bash

            pip install -U langchain-critique
            export CRITIQUE_API_KEY="your-api-key"

    For detailed API documentation, visit: https://critiquebrowser.app/en/flow-api?view=usage

    Key init args:
        api_key: str
            Critique API key. If not provided, will look for CRITIQUE_API_KEY env var.
        base_url: str
            Base URL for Critique API. Defaults to https://api.critiquebrowser.app

    Instantiation:
        .. code-block:: python

            from langchain_critique import CritiqueAPIDesignTool

            tool = CritiqueAPIDesignTool(
                api_key="your-api-key"  # Optional if env var is set
            )

    Create API example:
        .. code-block:: python

            tool.invoke({
                "operation": "create",
                "prompt": "Create an API that takes a company name and returns their ESG score"
            })

    Update API example:
        .. code-block:: python

            tool.invoke({
                "operation": "update",
                "api_id": "api_123",
                "prompt": "Add carbon footprint metrics to the response"
            })

    List APIs example:
        .. code-block:: python

            tool.invoke({
                "operation": "list"
            })

    Delete API example:
        .. code-block:: python

            tool.invoke({
                "operation": "delete",
                "api_id": "api_123"
            })
    """

    name: str = "critique_api_design"
    description: str = (
        "Design and manage APIs through natural language. Can create new APIs, "
        "update existing ones, list available APIs, or delete APIs."
    )
    args_schema: Type[BaseModel] = CritiqueAPIDesignInput

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.critiquebrowser.app",
    ):
        """Initialize the Critique API design tool."""
        super().__init__()
        self.api_key = api_key or os.getenv("CRITIQUE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Critique API key must be provided either through api_key parameter "
                "or CRITIQUE_API_KEY environment variable"
            )
        self.base_url = base_url
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }

    def _run(
        self,
        operation: APIOperation,
        prompt: Optional[str] = None,
        api_id: Optional[str] = None,
        schema_updates: Optional[Dict] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict:
        """Execute the API design operation."""
        endpoint = f"{self.base_url}/v1/design-api"
        
        if operation == APIOperation.LIST:
            response = requests.get(endpoint, headers=self.headers)
            
        elif operation == APIOperation.DELETE:
            if not api_id:
                raise ValueError("api_id is required for delete operation")
            response = requests.delete(f"{endpoint}/{api_id}", headers=self.headers)
            
        elif operation == APIOperation.CREATE:
            if not prompt:
                raise ValueError("prompt is required for create operation")
            payload = {
                "original_prompt": prompt,
                "id": None
            }
            response = requests.post(endpoint, headers=self.headers, json=payload)
            
        elif operation == APIOperation.UPDATE:
            if not api_id:
                raise ValueError("api_id is required for update operation")
            payload = {
                "id": api_id,
                "prompt": prompt,
                "schema_updates": schema_updates
            }
            response = requests.post(endpoint, headers=self.headers, json=payload)
            
        if response.status_code != 200:
            raise ValueError(f"API design operation failed: {response.text}")
            
        return response.json()

# Add this class to create dynamic tools from designed APIs
class CritiqueDynamicAPITool(BaseTool):
    """Dynamic tool created from a Critique-designed API.
    
    This tool is created dynamically based on an API designed through the CritiqueAPIDesignTool.
    """
    
    def __init__(
        self,
        api_id: str,
        name: str,
        description: str,
        input_schema: Dict,
        api_key: Optional[str] = None,
        base_url: str = "https://api.critiquebrowser.app",
    ):
        """Initialize the dynamic API tool."""
        super().__init__()
        self.api_id = api_id
        self.name = name
        self.description = description
        self.api_key = api_key or os.getenv("CRITIQUE_API_KEY")
        self.base_url = base_url
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Create dynamic input schema
        self.args_schema = create_model(
            f"CritiqueAPI_{api_id}_Input",
            **{k: (v["type"], Field(..., description=v.get("description", ""))) 
               for k, v in input_schema.items()}
        )

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Dict:
        """Execute the API call."""
        endpoint = f"{self.base_url}/v1/user-defined-service/{self.api_id}"
        response = requests.post(
            endpoint,
            headers=self.headers,
            json=kwargs
        )
        
        if response.status_code != 200:
            raise ValueError(f"API execution failed: {response.text}")
            
        return response.json()

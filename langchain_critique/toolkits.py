"""Critique toolkits."""

from typing import List, Optional, Dict
import requests

from langchain_core.tools import BaseTool, BaseToolkit

from langchain_critique.tools import (
    CritiqueSearchTool,
    CritiqueAPIDesignTool,
    CritiqueDynamicAPITool
)


class CritiqueToolkit(BaseToolkit):
    """Critique toolkit for agentic search and API design.

    Setup:
        Install ``langchain-critique`` and set environment variable ``CRITIQUE_API_KEY``.
        Get your API key at https://critiquebrowser.app/en/flow-api?view=keys

        .. code-block:: bash

            pip install -U langchain-critique
            export CRITIQUE_API_KEY="your-api-key"

    For detailed API documentation, visit: https://critiquebrowser.app/en/flow-api?view=usage

    Key init args:
        api_key: Optional[str]
            Critique API key. If not provided, will look for CRITIQUE_API_KEY env var.
        base_url: str
            Base URL for Critique API. Defaults to https://api.critiquebrowser.app
        include_apis: bool
            Whether to include dynamically created APIs as tools. Defaults to True.

    Instantiate:
        .. code-block:: python

            from langchain_critique import CritiqueToolkit

            toolkit = CritiqueToolkit(
                api_key="your-api-key",  # Optional if env var is set
                include_apis=True  # Include dynamic API tools
            )

    Tools:
        .. code-block:: python

            toolkit.get_tools()

        Returns a list containing:
        - CritiqueSearchTool: For performing grounded searches with optional image input
        - CritiqueAPIDesignTool: For creating and managing APIs through natural language
        - Dynamic API tools: If include_apis=True, includes tools for all created APIs
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.critiquebrowser.app",
        include_apis: bool = True,
    ):
        """Initialize the toolkit."""
        self.api_key = api_key
        self.base_url = base_url
        self.include_apis = include_apis

    def _get_dynamic_api_tools(self) -> List[BaseTool]:
        """Get tools for all created APIs."""
        if not self.include_apis:
            return []
            
        # Get list of available APIs
        api_design_tool = CritiqueAPIDesignTool(
            api_key=self.api_key,
            base_url=self.base_url
        )
        apis = api_design_tool._run(operation="list")
        
        # Create a tool for each API
        tools = []
        for api in apis:
            tools.append(
                CritiqueDynamicAPITool(
                    api_id=api["id"],
                    name=api["name"],
                    description=api["description"],
                    input_schema=api["input_schema"],
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            )
        return tools

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        tools = [
            CritiqueSearchTool(
                api_key=self.api_key,
                base_url=self.base_url
            ),
            CritiqueAPIDesignTool(
                api_key=self.api_key,
                base_url=self.base_url
            )
        ]
        
        # Add dynamic API tools if enabled
        tools.extend(self._get_dynamic_api_tools())
        
        return tools

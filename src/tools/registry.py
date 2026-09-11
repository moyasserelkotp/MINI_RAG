from typing import Dict, List, Type
from .base import BaseTool


class ToolRegistry:
    """Registry for managing and accessing Agent Tools."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        
    def register(self, tool: BaseTool):
        """Register a tool instance."""
        if tool.name in self._tools:
            raise ValueError(f"Tool with name {tool.name} is already registered.")
        self._tools[tool.name] = tool
        
    def get_tool(self, name: str) -> BaseTool:
        """Retrieve a tool by name."""
        if name not in self._tools:
            raise KeyError(f"Tool {name} not found in registry.")
        return self._tools[name]
        
    def get_all_tools(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())
        
    def get_tools_schema(self) -> List[Dict]:
        """Get schemas for all registered tools, formatted for LLM prompts."""
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_schema
            })
        return schemas

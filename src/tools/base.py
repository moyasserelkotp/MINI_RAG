from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """Abstract base class for all Agent Tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the tool, used by the LLM to select it."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Detailed description of what the tool does and when to use it."""
        pass
    
    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """JSON schema defining the expected arguments for the tool."""
        pass
        
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with the provided arguments."""
        pass

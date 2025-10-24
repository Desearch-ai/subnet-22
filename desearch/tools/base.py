from abc import abstractmethod, ABC
from typing import List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from desearch.protocol import ScraperTextRole


class BaseTool(ABC):
    tool_id: str
    slug: Optional[str] = None
    tool_manager: Any = None

    @abstractmethod
    async def _arun(self, *args, **kwargs) -> Any:
        """Asynchronous run method for the tool."""
        pass

    @abstractmethod
    async def send_event(self, send, response_streamer, data):
        pass


class BaseToolkit(BaseModel):
    toolkit_id: str
    name: str
    description: str
    slug: str
    is_active: bool = Field(default=True)
    tool_manager: Any = None

    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        pass

    @abstractmethod
    async def summarize(
        self, prompt, model, data, system_message
    ) -> Tuple[Any, ScraperTextRole]:
        pass

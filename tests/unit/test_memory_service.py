import pytest
from unittest.mock import MagicMock, AsyncMock
from services.memory_service import MemoryService

@pytest.fixture
def memory_service():
    db_client = AsyncMock()
    vectordb_client = MagicMock()
    generation_client = MagicMock()
    embedding_client = MagicMock()
    embedding_client.embedding_size = 1536
    template_parser = MagicMock()
    
    class Settings:
        PRIMARY_LANG = "en"
        
    return MemoryService(
        db_client=db_client,
        vectordb_client=vectordb_client,
        generation_client=generation_client,
        embedding_client=embedding_client,
        template_parser=template_parser,
        app_settings=Settings(),
        initialized_collections=set()
    )

def test_entity_collection_name(memory_service):
    name = memory_service.get_entity_collection_name("project_abc")
    assert name == "entities_project_abc_1536"

@pytest.mark.asyncio
async def test_condense_query_no_messages(memory_service):
    # If no history, query remains unchanged without calling LLM
    result = await memory_service.condense_query(query="What is Giza?", session_messages=[])
    assert result == "What is Giza?"
    memory_service.generation_client.generate_text.assert_not_called()

@pytest.mark.asyncio
async def test_condense_query_with_messages(memory_service):
    class FakeMsg:
        def __init__(self, role, text):
            self.role = role
            self.text = text
            
    messages = [
        FakeMsg(role="user", text="I want to visit Egypt"),
        FakeMsg(role="assistant", text="Egypt is a great destination.")
    ]
    
    memory_service.template_parser.get.return_value = "Condense: What is there to do?"
    memory_service.generation_client.construct_prompt.return_value = "system prompt"
    memory_service.generation_client.generate_text.return_value = "What activities are in Egypt?"
    
    result = await memory_service.condense_query(query="What is there to do?", session_messages=messages)
    assert result == "What activities are in Egypt?"

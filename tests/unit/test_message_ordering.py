import pytest
from datetime import datetime, timezone
import asyncio
from unittest.mock import MagicMock
from models.MessageModel import MessageModel
from models.db_schemes.chat_message import ChatMessage

@pytest.mark.asyncio
async def test_get_messages_by_session_ordering():
    # Setup mock DB client and collection
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    
    # Create the model
    model = MessageModel(db_client=mock_db)
    
    # Create test messages (simulating DB docs)
    # The newest message should have the latest timestamp
    now = datetime.now(timezone.utc)
    docs = [
        {"session_id": "s1", "role": "user", "text": "newest", "created_at": now},
        {"session_id": "s1", "role": "assistant", "text": "middle", "created_at": now},
        {"session_id": "s1", "role": "user", "text": "oldest", "created_at": now},
    ]
    
    # Setup cursor mock to simulate MongoDB returning newest first (sort created_at -1)
    class AsyncMockCursor:
        def __init__(self, items):
            self.items = items
        def sort(self, *args, **kwargs): return self
        def skip(self, *args, **kwargs): return self
        def limit(self, *args, **kwargs): return self
        async def __aiter__(self):
            for item in self.items:
                yield item

    mock_cursor = AsyncMockCursor(docs)
    mock_collection.find.return_value = mock_cursor
    
    # Run the method
    messages = await model.get_messages_by_session("s1", limit=3)
    
    # Verify the results are reversed (oldest -> newest)
    assert len(messages) == 3
    assert messages[0].text == "oldest"
    assert messages[1].text == "middle"
    assert messages[2].text == "newest"
